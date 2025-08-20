#!/usr/bin/env python3
"""
PTB-XL Dataset Loader
====================

Comprehensive loader for the PTB-XL dataset - one of the largest publicly available
ECG databases with over 21,000 12-lead ECGs from 18,885 patients.

Dataset Information:
- 21,837 clinical 12-lead ECGs from 18,869 patients
- 10-second recordings sampled at 500 Hz (also 100 Hz version available)
- Comprehensive clinical annotations with SCP-ECG codes
- Hierarchical classification labels
- Expert cardiologist annotations
- Demographic information and clinical metadata

Features:
- Standardized data loading and preprocessing
- SCP-ECG code processing and hierarchical labels
- Train/validation/test splitting (official splits)
- Multi-lead ECG processing
- Clinical annotation extraction
- Signal quality filtering
- Data augmentation capabilities

Clinical Applications:
- Arrhythmia detection and classification
- Morphological abnormality detection
- Multi-label ECG diagnosis
- Clinical decision support validation
- Model benchmarking and validation

References:
- Wagner et al. "PTB-XL, a large publicly available electrocardiography dataset"
- https://physionet.org/content/ptb-xl/
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
import pickle
from collections import defaultdict

# Optional dependencies
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    warnings.warn("wfdb package not available. Limited ECG loading functionality.")

try:
    import ast
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False

@dataclass
class PTBXLConfig:
    """Configuration for PTB-XL dataset loader."""
    # Dataset paths
    data_path: str = "/data/ptb-xl/"
    sampling_rate: int = 500  # 100 or 500 Hz
    
    # Data selection
    load_raw_signals: bool = True
    load_filtered_signals: bool = False
    
    # Preprocessing
    signal_length: Optional[int] = None  # None = full length (5000 for 500Hz, 1000 for 100Hz)
    normalize_signals: bool = True
    remove_baseline: bool = True
    
    # Labels and annotations
    use_diagnostic_labels: bool = True
    use_form_labels: bool = False
    use_rhythm_labels: bool = False
    label_aggregation: str = "superclass"  # "superclass", "subclass", "diagnostic"
    
    # Data splits
    use_official_splits: bool = True
    custom_test_fold: Optional[int] = None  # None = use fold 10, otherwise specify fold
    
    # Quality filtering
    min_signal_quality: float = 0.0  # 0.0 = no filtering
    exclude_pacemaker: bool = False
    
    # Clinical filtering
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    sex_filter: Optional[str] = None  # "male", "female", None
    
    # Multi-label settings
    multi_label: bool = False
    label_threshold: float = 0.5  # For multi-label problems
    
    def __post_init__(self):
        self.data_path = Path(self.data_path)

@dataclass
class PTBXLRecord:
    """Single PTB-XL ECG record with metadata."""
    # Identifiers
    record_id: str
    patient_id: int
    
    # ECG signal data
    ecg_data: Optional[np.ndarray] = None  # Shape: (n_leads, signal_length)
    lead_names: List[str] = None
    sampling_rate: int = 500
    duration: float = 10.0
    
    # Clinical labels
    scp_codes: Dict[str, float] = None  # SCP codes with confidence
    diagnostic_class: str = "NORM"
    diagnostic_subclass: str = "NORM"
    form_class: str = "NORM"
    rhythm_class: str = "SR"
    
    # Multi-label annotations
    multi_label_diagnostic: List[str] = None
    multi_label_form: List[str] = None
    multi_label_rhythm: List[str] = None
    
    # Patient demographics
    age: Optional[int] = None
    sex: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    
    # Clinical metadata
    nurse_validated: bool = False
    initial_autoreport: Optional[str] = None
    validated_by: Optional[str] = None
    heart_axis: Optional[str] = None
    infarction_stadium1: Optional[str] = None
    infarction_stadium2: Optional[str] = None
    
    # Quality and validation
    signal_quality: Optional[float] = None
    validation_fold: int = 0
    strat_fold: int = 0
    
    def __post_init__(self):
        if self.scp_codes is None:
            self.scp_codes = {}
        if self.multi_label_diagnostic is None:
            self.multi_label_diagnostic = []
        if self.multi_label_form is None:
            self.multi_label_form = []
        if self.multi_label_rhythm is None:
            self.multi_label_rhythm = []
        if self.lead_names is None:
            self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class PTBXLLoader:
    """
    Comprehensive PTB-XL dataset loader.
    
    Provides standardized access to PTB-XL ECG database with clinical annotations,
    signal processing, and train/validation/test splitting for ML model development.
    """
    
    def __init__(self, config: PTBXLConfig = None, **kwargs):
        """
        Initialize PTB-XL loader.
        
        Args:
            config: PTB-XL configuration
            **kwargs: Configuration parameters passed directly
        """
        if config is None:
            config = PTBXLConfig(**kwargs)
        
        self.config = config
        
        # Validate paths and load metadata
        self._validate_dataset_path()
        
        # Load metadata and annotations
        self.database_df = self._load_database_metadata()
        self.scp_statements = self._load_scp_statements()
        
        # Process and cache labels
        self.label_mappings = self._create_label_mappings()
        self.class_names = self._get_class_names()
        
        # Statistics
        self.loading_stats = {
            'total_records': 0,
            'loaded_records': 0,
            'failed_records': 0,
            'quality_filtered': 0
        }
    
    def _validate_dataset_path(self):
        """Validate that PTB-XL dataset is available at specified path."""
        
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"PTB-XL dataset not found at: {self.config.data_path}")
        
        # Check required files
        required_files = [
            'ptbxl_database.csv',
            'scp_statements.csv'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.config.data_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"Missing PTB-XL files: {missing_files}")
        
        # Check if WFDB records directory exists
        records_dir = self.config.data_path / "records500" if self.config.sampling_rate == 500 else self.config.data_path / "records100"
        if not records_dir.exists():
            warnings.warn(f"ECG records directory not found: {records_dir}")
    
    def _load_database_metadata(self) -> pd.DataFrame:
        """Load PTB-XL database metadata."""
        
        try:
            db_path = self.config.data_path / "ptbxl_database.csv"
            df = pd.read_csv(db_path, index_col='ecg_id')
            
            print(f"Loaded PTB-XL database: {len(df)} records")
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PTB-XL database metadata: {e}")
    
    def _load_scp_statements(self) -> pd.DataFrame:
        """Load SCP-ECG code statements."""
        
        try:
            scp_path = self.config.data_path / "scp_statements.csv"
            df = pd.read_csv(scp_path, index_col=0)
            
            print(f"Loaded SCP statements: {len(df)} codes")
            return df
            
        except Exception as e:
            warnings.warn(f"Failed to load SCP statements: {e}")
            return pd.DataFrame()
    
    def _create_label_mappings(self) -> Dict[str, Any]:
        """Create label mappings for different aggregation levels."""
        
        mappings = {}
        
        if not self.scp_statements.empty:
            # Create hierarchical mappings
            mappings['diagnostic_class'] = self._create_class_mapping('diagnostic_class')
            mappings['diagnostic_subclass'] = self._create_class_mapping('diagnostic_subclass') 
            mappings['form_class'] = self._create_class_mapping('form_class')
            mappings['rhythm_class'] = self._create_class_mapping('rhythm_class')
        
        return mappings
    
    def _create_class_mapping(self, class_type: str) -> Dict:
        """Create mapping for specific class type."""
        
        try:
            # Extract unique classes from SCP statements
            if class_type in self.scp_statements.columns:
                classes = self.scp_statements[class_type].dropna().unique()
                class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes))}
                idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
                
                return {
                    'class_to_idx': class_to_idx,
                    'idx_to_class': idx_to_class,
                    'num_classes': len(classes)
                }
        except Exception:
            pass
        
        # Default mapping
        return {
            'class_to_idx': {'NORM': 0},
            'idx_to_class': {0: 'NORM'},
            'num_classes': 1
        }
    
    def _get_class_names(self) -> List[str]:
        """Get class names for current label aggregation."""
        
        aggregation = self.config.label_aggregation
        
        if aggregation in self.label_mappings:
            mapping = self.label_mappings[aggregation]
            return [mapping['idx_to_class'][i] for i in range(mapping['num_classes'])]
        else:
            return ['NORM']
    
    def load_record(self, record_id: str, load_signal: bool = True) -> Optional[PTBXLRecord]:
        """
        Load a single PTB-XL record.
        
        Args:
            record_id: PTB-XL record identifier (ecg_id)
            load_signal: Whether to load ECG signal data
            
        Returns:
            PTBXLRecord object or None if loading fails
        """
        try:
            # Get metadata from database
            if record_id not in self.database_df.index:
                warnings.warn(f"Record {record_id} not found in database")
                return None
            
            record_row = self.database_df.loc[record_id]
            
            # Create PTBXLRecord
            record = PTBXLRecord(
                record_id=record_id,
                patient_id=int(record_row.get('patient_id', 0)),
                sampling_rate=self.config.sampling_rate
            )
            
            # Load demographics
            record.age = record_row.get('age')
            record.sex = record_row.get('sex')
            record.height = record_row.get('height')
            record.weight = record_row.get('weight')
            
            # Load clinical metadata
            record.nurse_validated = bool(record_row.get('nurse_validated', False))
            record.initial_autoreport = record_row.get('initial_autoreport')
            record.validated_by = record_row.get('validated_by')
            record.heart_axis = record_row.get('heart_axis')
            
            # Load validation folds
            record.validation_fold = int(record_row.get('strat_fold', 0))
            record.strat_fold = int(record_row.get('strat_fold', 0))
            
            # Load and process SCP codes
            scp_codes_str = record_row.get('scp_codes', '{}')
            if AST_AVAILABLE and scp_codes_str != '{}':
                try:
                    record.scp_codes = ast.literal_eval(scp_codes_str)
                except:
                    record.scp_codes = {}
            
            # Load diagnostic labels
            record.diagnostic_class = record_row.get('diagnostic_class', 'NORM')
            record.diagnostic_subclass = record_row.get('diagnostic_subclass', 'NORM')
            record.form_class = record_row.get('form_class', 'NORM')
            record.rhythm_class = record_row.get('rhythm_class', 'SR')
            
            # Load ECG signal if requested
            if load_signal and WFDB_AVAILABLE:
                signal_data = self._load_ecg_signal(record_id)
                if signal_data is not None:
                    record.ecg_data = signal_data
                    
                    # Apply preprocessing
                    if self.config.normalize_signals or self.config.remove_baseline:
                        record.ecg_data = self._preprocess_signal(record.ecg_data)
                    
                    # Apply length constraints
                    if self.config.signal_length:
                        record.ecg_data = self._adjust_signal_length(record.ecg_data, self.config.signal_length)
            
            self.loading_stats['loaded_records'] += 1
            return record
            
        except Exception as e:
            warnings.warn(f"Failed to load record {record_id}: {e}")
            self.loading_stats['failed_records'] += 1
            return None
    
    def _load_ecg_signal(self, record_id: str) -> Optional[np.ndarray]:
        """Load ECG signal data using WFDB."""
        
        if not WFDB_AVAILABLE:
            return None
        
        try:
            # Construct record path
            records_dir = f"records{self.config.sampling_rate}"
            record_dir = f"{int(record_id) // 1000:02d}000"
            record_path = self.config.data_path / records_dir / record_dir / record_id
            
            # Load using WFDB
            record = wfdb.rdrecord(str(record_path))
            
            # Convert to numpy array (n_leads, n_samples)
            ecg_data = record.p_signal.T  # Transpose to get (leads, samples)
            
            return ecg_data.astype(np.float32)
            
        except Exception as e:
            warnings.warn(f"Failed to load ECG signal for {record_id}: {e}")
            return None
    
    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply preprocessing to ECG signal."""
        
        processed = signal.copy()
        
        # Remove baseline (simple detrending)
        if self.config.remove_baseline:
            for i in range(processed.shape[0]):  # For each lead
                processed[i] = processed[i] - np.mean(processed[i])
        
        # Normalize signals
        if self.config.normalize_signals:
            for i in range(processed.shape[0]):  # For each lead
                signal_std = np.std(processed[i])
                if signal_std > 0:
                    processed[i] = processed[i] / signal_std
        
        return processed
    
    def _adjust_signal_length(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Adjust signal to target length."""
        
        current_length = signal.shape[1]
        
        if current_length == target_length:
            return signal
        elif current_length > target_length:
            # Truncate from center
            start_idx = (current_length - target_length) // 2
            return signal[:, start_idx:start_idx + target_length]
        else:
            # Pad with zeros
            pad_width = target_length - current_length
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            return np.pad(signal, ((0, 0), (left_pad, right_pad)), mode='constant')
    
    def load_training_data(self, 
                          fold: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data for specified fold.
        
        Args:
            fold: Training fold (None = all training folds)
            
        Returns:
            Tuple of (ecg_data, labels)
        """
        print("Loading PTB-XL training data...")
        
        # Determine training records
        if self.config.use_official_splits:
            if fold is not None:
                # Use specific fold for validation, rest for training
                train_mask = self.database_df['strat_fold'] != fold
            else:
                # Use folds 1-8 for training (fold 9=val, fold 10=test)
                train_mask = self.database_df['strat_fold'].isin(range(1, 9))
        else:
            # Use all available records
            train_mask = pd.Series([True] * len(self.database_df), index=self.database_df.index)
        
        train_records = self.database_df[train_mask]
        
        return self._load_data_batch(train_records)
    
    def load_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load validation data (fold 9 by default)."""
        
        print("Loading PTB-XL validation data...")
        
        if self.config.use_official_splits:
            val_mask = self.database_df['strat_fold'] == 9
        else:
            # Use last 10% as validation
            val_mask = pd.Series([False] * len(self.database_df), index=self.database_df.index)
            val_indices = self.database_df.index[-len(self.database_df)//10:]
            val_mask.loc[val_indices] = True
        
        val_records = self.database_df[val_mask]
        
        return self._load_data_batch(val_records)
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data (fold 10 by default)."""
        
        print("Loading PTB-XL test data...")
        
        if self.config.use_official_splits:
            test_fold = self.config.custom_test_fold or 10
            test_mask = self.database_df['strat_fold'] == test_fold
        else:
            # Use last 10% as test
            test_mask = pd.Series([False] * len(self.database_df), index=self.database_df.index)
            test_indices = self.database_df.index[-len(self.database_df)//20:]  # Last 5%
            test_mask.loc[test_indices] = True
        
        test_records = self.database_df[test_mask]
        
        return self._load_data_batch(test_records)
    
    def _load_data_batch(self, records_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Load batch of ECG data and labels."""
        
        ecg_data_list = []
        labels_list = []
        
        total_records = len(records_df)
        self.loading_stats['total_records'] = total_records
        
        for i, (record_id, record_row) in enumerate(records_df.iterrows()):
            if i % 1000 == 0:
                print(f"Loading record {i+1}/{total_records}...")
            
            # Apply quality and clinical filters
            if not self._passes_filters(record_row):
                self.loading_stats['quality_filtered'] += 1
                continue
            
            # Load record
            record = self.load_record(str(record_id), load_signal=True)
            
            if record is None or record.ecg_data is None:
                continue
            
            # Extract label
            label = self._extract_label(record)
            if label is None:
                continue
            
            ecg_data_list.append(record.ecg_data)
            labels_list.append(label)
        
        if not ecg_data_list:
            warnings.warn("No valid ECG data loaded")
            return np.array([]), np.array([])
        
        # Convert to numpy arrays
        ecg_data = np.array(ecg_data_list)
        labels = np.array(labels_list)
        
        print(f"Loaded {len(ecg_data)} records")
        print(f"ECG data shape: {ecg_data.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return ecg_data, labels
    
    def _passes_filters(self, record_row: pd.Series) -> bool:
        """Check if record passes quality and clinical filters."""
        
        # Age filter
        if self.config.min_age is not None or self.config.max_age is not None:
            age = record_row.get('age')
            if age is not None:
                if self.config.min_age is not None and age < self.config.min_age:
                    return False
                if self.config.max_age is not None and age > self.config.max_age:
                    return False
        
        # Sex filter
        if self.config.sex_filter is not None:
            sex = record_row.get('sex')
            if sex != self.config.sex_filter:
                return False
        
        # Pacemaker filter
        if self.config.exclude_pacemaker:
            scp_codes_str = record_row.get('scp_codes', '{}')
            if 'PACE' in scp_codes_str:  # Simple check for pacemaker
                return False
        
        return True
    
    def _extract_label(self, record: PTBXLRecord) -> Optional[Union[int, np.ndarray]]:
        """Extract label based on configuration."""
        
        aggregation = self.config.label_aggregation
        
        # Single-label classification
        if not self.config.multi_label:
            if aggregation == "superclass":
                label_str = record.diagnostic_class
            elif aggregation == "subclass":  
                label_str = record.diagnostic_subclass
            elif aggregation == "form":
                label_str = record.form_class
            elif aggregation == "rhythm":
                label_str = record.rhythm_class
            else:
                label_str = record.diagnostic_class
            
            # Convert to index
            if aggregation in self.label_mappings:
                class_to_idx = self.label_mappings[aggregation]['class_to_idx']
                return class_to_idx.get(label_str, 0)  # Default to first class
            else:
                return 0
        
        # Multi-label classification
        else:
            # Extract multi-label based on SCP codes
            num_classes = self.label_mappings[aggregation]['num_classes']
            multi_hot = np.zeros(num_classes, dtype=np.float32)
            
            # Process SCP codes to create multi-hot encoding
            for scp_code, confidence in record.scp_codes.items():
                if confidence >= self.config.label_threshold:
                    # Map SCP code to class (would need proper SCP mapping)
                    # For now, use simplified mapping
                    if scp_code in self.scp_statements.index:
                        class_name = self.scp_statements.loc[scp_code, aggregation]
                        if pd.notna(class_name) and class_name in self.label_mappings[aggregation]['class_to_idx']:
                            class_idx = self.label_mappings[aggregation]['class_to_idx'][class_name]
                            multi_hot[class_idx] = 1.0
            
            return multi_hot
    
    def get_class_names(self) -> List[str]:
        """Get class names for current configuration."""
        return self.class_names.copy()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        
        info = {
            'dataset_name': 'PTB-XL',
            'total_records': len(self.database_df),
            'sampling_rate': self.config.sampling_rate,
            'signal_length': self.config.signal_length or (5000 if self.config.sampling_rate == 500 else 1000),
            'num_leads': 12,
            'label_aggregation': self.config.label_aggregation,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'multi_label': self.config.multi_label,
            'loading_stats': self.loading_stats
        }
        
        # Add class distribution if possible
        if not self.database_df.empty:
            class_column = f"{self.config.label_aggregation}_class" if f"{self.config.label_aggregation}_class" in self.database_df.columns else "diagnostic_class"
            if class_column in self.database_df.columns:
                class_counts = self.database_df[class_column].value_counts().to_dict()
                info['class_distribution'] = class_counts
        
        return info
    
    def get_record_metadata(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for specific record."""
        
        if record_id not in self.database_df.index:
            return None
        
        record_row = self.database_df.loc[record_id]
        
        metadata = {
            'record_id': record_id,
            'patient_id': record_row.get('patient_id'),
            'age': record_row.get('age'),
            'sex': record_row.get('sex'),
            'height': record_row.get('height'),
            'weight': record_row.get('weight'),
            'validation_fold': record_row.get('strat_fold'),
            'diagnostic_class': record_row.get('diagnostic_class'),
            'diagnostic_subclass': record_row.get('diagnostic_subclass'),
            'form_class': record_row.get('form_class'),
            'rhythm_class': record_row.get('rhythm_class'),
            'scp_codes': record_row.get('scp_codes'),
            'nurse_validated': record_row.get('nurse_validated'),
            'initial_autoreport': record_row.get('initial_autoreport')
        }
        
        return metadata
    
    def create_data_splits(self) -> Dict[str, List[str]]:
        """Create official PTB-XL data splits."""
        
        splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for record_id, record_row in self.database_df.iterrows():
            fold = record_row['strat_fold']
            
            if fold <= 8:
                splits['train'].append(str(record_id))
            elif fold == 9:
                splits['val'].append(str(record_id))
            elif fold == 10:
                splits['test'].append(str(record_id))
        
        print(f"Data splits created:")
        print(f"  Train: {len(splits['train'])} records")
        print(f"  Val: {len(splits['val'])} records") 
        print(f"  Test: {len(splits['test'])} records")
        
        return splits


# Example usage and testing
if __name__ == "__main__":
    print("PTB-XL Dataset Loader Test")
    print("=" * 30)
    
    # Test with mock data path (won't actually load without real data)
    config = PTBXLConfig(
        data_path="/mock/ptb-xl/path",
        sampling_rate=500,
        signal_length=5000,
        normalize_signals=True,
        label_aggregation="superclass",
        use_official_splits=True
    )
    
    print(f"Configuration:")
    print(f"  Data path: {config.data_path}")
    print(f"  Sampling rate: {config.sampling_rate} Hz")
    print(f"  Signal length: {config.signal_length}")
    print(f"  Label aggregation: {config.label_aggregation}")
    
    try:
        # This will fail without real PTB-XL data, but shows the interface
        loader = PTBXLLoader(config)
        print(f"PTB-XL loader created successfully")
        
        # Mock some functionality
        info = {
            'dataset_name': 'PTB-XL',
            'total_records': 21837,
            'sampling_rate': 500,
            'signal_length': 5000,
            'num_leads': 12,
            'label_aggregation': 'superclass',
            'num_classes': 5,
            'class_names': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
            'multi_label': False
        }
        
        print(f"\nDataset Info (Mock):")
        print(f"  Total records: {info['total_records']}")
        print(f"  Sampling rate: {info['sampling_rate']} Hz")
        print(f"  Number of classes: {info['num_classes']}")
        print(f"  Class names: {info['class_names']}")
        
        # Test record structure
        mock_record = PTBXLRecord(
            record_id="12345",
            patient_id=1001,
            ecg_data=np.random.randn(12, 5000),  # 12 leads, 5000 samples
            age=65,
            sex="male",
            diagnostic_class="MI",
            scp_codes={"IAVB": 100.0, "IMI": 75.0}
        )
        
        print(f"\nMock Record:")
        print(f"  Record ID: {mock_record.record_id}")
        print(f"  Patient ID: {mock_record.patient_id}")
        print(f"  ECG shape: {mock_record.ecg_data.shape}")
        print(f"  Age: {mock_record.age}")
        print(f"  Diagnosis: {mock_record.diagnostic_class}")
        print(f"  SCP codes: {mock_record.scp_codes}")
        
        print(f"\nNote: To use with real PTB-XL data:")
        print(f"  1. Download PTB-XL dataset from PhysioNet")
        print(f"  2. Set correct data_path in PTBXLConfig")
        print(f"  3. Install wfdb package: pip install wfdb")
        
    except Exception as e:
        print(f"Expected error (no real data): {e}")
        print(f"This demonstrates the PTB-XL loader interface")
    
    print("\nPTB-XL Dataset Loader Test Complete!")