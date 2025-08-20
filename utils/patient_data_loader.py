"""
Patient Data Loaders for CoT-RAG Stage 2
Handles loading and preprocessing of clinical data for RAG population.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from core.stage2_rag import PatientData, create_patient_data

@dataclass
class ECGData:
    """Container for ECG signal data."""
    signal: np.ndarray = None
    sampling_rate: int = 500
    duration: float = 10.0
    leads: List[str] = None
    annotations: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.leads is None:
            # Standard 12-lead ECG
            self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if self.annotations is None:
            self.annotations = {}

class PatientDataLoader:
    """
    Loads patient data from various sources for CoT-RAG Stage 2 processing.
    
    Supports:
    - Clinical text files
    - ECG signal data 
    - Structured patient records (JSON/CSV)
    - MIMIC-IV style data
    - PTB-XL ECG dataset format
    """
    
    def __init__(self, data_directory: str = "data"):
        """
        Initialize patient data loader.
        
        Args:
            data_directory: Base directory for patient data files
        """
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(exist_ok=True)
        
        # Track loaded patients
        self.loaded_patients = {}
    
    def load_patient_from_files(self, patient_id: str, 
                               clinical_text_file: str = None,
                               ecg_data_file: str = None,
                               demographics_file: str = None,
                               query_description: str = "") -> PatientData:
        """
        Load patient data from individual files.
        
        Args:
            patient_id: Unique patient identifier
            clinical_text_file: Path to clinical notes file
            ecg_data_file: Path to ECG data file
            demographics_file: Path to demographics JSON file
            query_description: Specific clinical query
            
        Returns:
            PatientData: Structured patient data object
        """
        # Load clinical text
        clinical_text = ""
        if clinical_text_file:
            try:
                with open(clinical_text_file, 'r', encoding='utf-8') as f:
                    clinical_text = f.read()
            except Exception as e:
                print(f"Warning: Could not load clinical text from {clinical_text_file}: {e}")
        
        # Load ECG data
        ecg_data = None
        if ecg_data_file:
            ecg_data = self._load_ecg_data(ecg_data_file)
        
        # Load demographics
        demographics = {}
        if demographics_file:
            try:
                with open(demographics_file, 'r') as f:
                    demographics = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load demographics from {demographics_file}: {e}")
        
        patient_data = PatientData(
            patient_id=patient_id,
            clinical_text=clinical_text,
            ecg_data=ecg_data,
            demographics=demographics,
            query_description=query_description
        )
        
        self.loaded_patients[patient_id] = patient_data
        return patient_data
    
    def load_from_mimic_style(self, patient_id: str, 
                             note_events_file: str = None,
                             admissions_file: str = None,
                             query_description: str = "") -> PatientData:
        """
        Load patient data from MIMIC-IV style CSV files.
        
        Args:
            patient_id: Patient identifier
            note_events_file: NOTEEVENTS.csv style file
            admissions_file: ADMISSIONS.csv style file
            query_description: Clinical query
            
        Returns:
            PatientData: Structured patient data
        """
        clinical_text = ""
        demographics = {}
        
        # Load clinical notes
        if note_events_file and Path(note_events_file).exists():
            try:
                notes_df = pd.read_csv(note_events_file)
                patient_notes = notes_df[notes_df['SUBJECT_ID'] == int(patient_id)]
                
                # Combine all notes for this patient
                note_texts = []
                for _, note in patient_notes.iterrows():
                    category = note.get('CATEGORY', 'Unknown')
                    text = note.get('TEXT', '')
                    if text:
                        note_texts.append(f"[{category}] {text}")
                
                clinical_text = "\n\n".join(note_texts)
                
            except Exception as e:
                print(f"Warning: Could not process MIMIC note events: {e}")
        
        # Load demographics from admissions
        if admissions_file and Path(admissions_file).exists():
            try:
                admissions_df = pd.read_csv(admissions_file)
                patient_admissions = admissions_df[admissions_df['SUBJECT_ID'] == int(patient_id)]
                
                if not patient_admissions.empty:
                    latest_admission = patient_admissions.iloc[-1]
                    demographics = {
                        'gender': latest_admission.get('GENDER', 'Unknown'),
                        'admission_type': latest_admission.get('ADMISSION_TYPE', 'Unknown'),
                        'diagnosis': latest_admission.get('DIAGNOSIS', 'Unknown'),
                        'ethnicity': latest_admission.get('ETHNICITY', 'Unknown')
                    }
                    
            except Exception as e:
                print(f"Warning: Could not process MIMIC admissions: {e}")
        
        return PatientData(
            patient_id=patient_id,
            clinical_text=clinical_text,
            demographics=demographics,
            query_description=query_description
        )
    
    def load_from_ptb_xl(self, patient_id: str, ptb_xl_dir: str,
                        include_statements: bool = True) -> PatientData:
        """
        Load patient data from PTB-XL dataset format.
        
        Args:
            patient_id: ECG record ID in PTB-XL
            ptb_xl_dir: PTB-XL dataset directory
            include_statements: Whether to include diagnostic statements
            
        Returns:
            PatientData: Patient data with ECG and clinical information
        """
        ptb_dir = Path(ptb_xl_dir)
        
        # Load metadata
        metadata_file = ptb_dir / "ptbxl_database.csv"
        statements_file = ptb_dir / "scp_statements.csv"
        
        clinical_text = ""
        demographics = {}
        ecg_data = None
        
        try:
            # Load PTB-XL database
            if metadata_file.exists():
                ptb_df = pd.read_csv(metadata_file, index_col='ecg_id')
                
                if int(patient_id) in ptb_df.index:
                    record = ptb_df.loc[int(patient_id)]
                    
                    # Extract demographics
                    demographics = {
                        'age': record.get('age', 'Unknown'),
                        'sex': record.get('sex', 'Unknown'),
                        'height': record.get('height', 'Unknown'),
                        'weight': record.get('weight', 'Unknown')
                    }
                    
                    # Extract diagnostic information
                    scp_codes = eval(record.get('scp_codes', '{}'))  # Dictionary of SCP codes
                    diagnostic_superclass = record.get('diagnostic_superclass', '')
                    
                    # Build clinical description
                    clinical_parts = []
                    if diagnostic_superclass:
                        clinical_parts.append(f"Diagnostic Category: {diagnostic_superclass}")
                    
                    if scp_codes:
                        clinical_parts.append(f"SCP Diagnostic Codes: {scp_codes}")
                    
                    # Load SCP statements if available
                    if include_statements and statements_file.exists():
                        statements_df = pd.read_csv(statements_file, index_col=0)
                        
                        for scp_code, likelihood in scp_codes.items():
                            if scp_code in statements_df.index:
                                statement = statements_df.loc[scp_code, 'description']
                                clinical_parts.append(f"{scp_code}: {statement} (likelihood: {likelihood})")
                    
                    clinical_text = "\n".join(clinical_parts)
                    
                    # Load ECG signal if available
                    signal_file = ptb_dir / "records500" / f"{patient_id:05d}_hr"
                    if signal_file.exists():
                        ecg_data = self._load_wfdb_signal(str(signal_file))
                        
        except Exception as e:
            print(f"Warning: Could not load PTB-XL data for patient {patient_id}: {e}")
        
        return PatientData(
            patient_id=patient_id,
            clinical_text=clinical_text,
            ecg_data=ecg_data,
            demographics=demographics,
            query_description="ECG analysis and diagnostic classification"
        )
    
    def create_synthetic_patient(self, patient_id: str, 
                                condition: str = "atrial_fibrillation",
                                severity: str = "moderate") -> PatientData:
        """
        Create synthetic patient data for testing purposes.
        
        Args:
            patient_id: Patient identifier
            condition: Primary condition to simulate
            severity: Condition severity
            
        Returns:
            PatientData: Synthetic patient data
        """
        # Synthetic clinical scenarios
        scenarios = {
            "atrial_fibrillation": {
                "clinical_text": f"""
Patient presents with irregular heart rhythm and palpitations.
ECG shows irregularly irregular rhythm with absence of P waves.
RR intervals are variable. Heart rate approximately 110-130 bpm.
Patient reports episodes of rapid heart rate over the past 2 weeks.
No chest pain or shortness of breath at rest.
Current medications: None relevant to arrhythmia.
""",
                "demographics": {"age": 72, "sex": "M", "weight": 85},
                "query": "Evaluate irregular rhythm and determine appropriate treatment"
            },
            
            "anterior_mi": {
                "clinical_text": f"""
67-year-old male presents with acute chest pain, onset 2 hours ago.
Pain described as crushing, substernal, radiating to left arm.
ECG shows ST-elevation in leads V1-V4 consistent with anterior STEMI.
Troponin elevated. Patient appears diaphoretic and anxious.
No prior history of coronary artery disease.
""",
                "demographics": {"age": 67, "sex": "M", "weight": 78},
                "query": "Acute coronary syndrome evaluation and management"
            },
            
            "heart_block": {
                "clinical_text": f"""
85-year-old female with history of syncope and dizziness.
ECG shows prolonged PR interval ({severity} heart block).
Patient reports several episodes of near-fainting.
Heart rate 45-50 bpm. Blood pressure stable.
Currently on metoprolol for hypertension.
""",
                "demographics": {"age": 85, "sex": "F", "weight": 62},
                "query": "Evaluate conduction abnormalities and syncope risk"
            }
        }
        
        scenario = scenarios.get(condition, scenarios["atrial_fibrillation"])
        
        # Add severity modifiers
        if severity == "severe":
            scenario["clinical_text"] += f"\nCondition severity: {severity}. Immediate intervention may be required."
        elif severity == "mild":
            scenario["clinical_text"] += f"\nCondition severity: {severity}. Conservative management considered."
        
        return PatientData(
            patient_id=patient_id,
            clinical_text=scenario["clinical_text"].strip(),
            demographics=scenario["demographics"],
            query_description=scenario["query"],
            additional_data={"synthetic": True, "condition": condition, "severity": severity}
        )
    
    def _load_ecg_data(self, file_path: str) -> ECGData:
        """Load ECG data from various formats."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            return self._load_ecg_json(file_path)
        elif file_path.suffix.lower() in ['.csv', '.tsv']:
            return self._load_ecg_csv(file_path)
        elif file_path.suffix.lower() == '.npy':
            return self._load_ecg_numpy(file_path)
        else:
            # Try WFDB format
            return self._load_wfdb_signal(str(file_path))
    
    def _load_ecg_json(self, file_path: Path) -> ECGData:
        """Load ECG data from JSON format."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            signal = np.array(data.get('signal', []))
            if signal.size == 0:
                signal = None
            
            return ECGData(
                signal=signal,
                sampling_rate=data.get('sampling_rate', 500),
                duration=data.get('duration', 10.0),
                leads=data.get('leads', None),
                annotations=data.get('annotations', {})
            )
        except Exception as e:
            print(f"Warning: Could not load ECG JSON: {e}")
            return ECGData()
    
    def _load_ecg_csv(self, file_path: Path) -> ECGData:
        """Load ECG data from CSV format."""
        try:
            df = pd.read_csv(file_path)
            
            # Assume first row might be lead names
            if all(isinstance(col, str) for col in df.columns):
                leads = list(df.columns)
                signal = df.values.T  # Transpose to get leads x samples
            else:
                leads = None
                signal = df.values.T
            
            return ECGData(
                signal=signal,
                leads=leads
            )
        except Exception as e:
            print(f"Warning: Could not load ECG CSV: {e}")
            return ECGData()
    
    def _load_ecg_numpy(self, file_path: Path) -> ECGData:
        """Load ECG data from NumPy format."""
        try:
            signal = np.load(file_path)
            return ECGData(signal=signal)
        except Exception as e:
            print(f"Warning: Could not load ECG NumPy: {e}")
            return ECGData()
    
    def _load_wfdb_signal(self, record_path: str) -> ECGData:
        """Load ECG signal using WFDB format (if wfdb package available)."""
        try:
            import wfdb
            
            record = wfdb.rdrecord(record_path)
            
            return ECGData(
                signal=record.p_signal.T,  # Transpose to get leads x samples
                sampling_rate=record.fs,
                duration=record.sig_len / record.fs,
                leads=record.sig_name,
                annotations={'units': record.units, 'comments': record.comments}
            )
        except ImportError:
            print("Warning: wfdb package not available for WFDB signal loading")
            return ECGData()
        except Exception as e:
            print(f"Warning: Could not load WFDB signal: {e}")
            return ECGData()
    
    def get_patient_list(self) -> List[str]:
        """Get list of loaded patient IDs."""
        return list(self.loaded_patients.keys())
    
    def save_patient_data(self, patient_data: PatientData, output_file: str):
        """Save patient data to JSON file."""
        data = {
            'patient_id': patient_data.patient_id,
            'clinical_text': patient_data.clinical_text,
            'demographics': patient_data.demographics,
            'query_description': patient_data.query_description,
            'additional_data': patient_data.additional_data
        }
        
        # Handle ECG data serialization
        if patient_data.ecg_data:
            data['ecg_metadata'] = {
                'sampling_rate': patient_data.ecg_data.sampling_rate,
                'duration': patient_data.ecg_data.duration,
                'leads': patient_data.ecg_data.leads,
                'annotations': patient_data.ecg_data.annotations,
                'has_signal': patient_data.ecg_data.signal is not None
            }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Patient data saved to: {output_file}")

# Convenience functions
def load_sample_patients(data_source: str = "synthetic", count: int = 3) -> List[PatientData]:
    """
    Load sample patient data for testing.
    
    Args:
        data_source: Source type ("synthetic", "mimic", "ptb_xl")
        count: Number of patients to load
        
    Returns:
        List of PatientData objects
    """
    loader = PatientDataLoader()
    patients = []
    
    if data_source == "synthetic":
        conditions = ["atrial_fibrillation", "anterior_mi", "heart_block"]
        severities = ["mild", "moderate", "severe"]
        
        for i in range(count):
            condition = conditions[i % len(conditions)]
            severity = severities[i % len(severities)]
            patient_id = f"synthetic_{condition}_{i+1:03d}"
            
            patient = loader.create_synthetic_patient(patient_id, condition, severity)
            patients.append(patient)
    
    return patients

def create_test_patient_simple() -> PatientData:
    """Create a simple test patient for quick testing."""
    return create_patient_data(
        patient_id="test_001",
        clinical_text="""
        72-year-old male presents with palpitations and irregular heart rhythm.
        ECG shows atrial fibrillation with rapid ventricular response.
        Heart rate 120-140 bpm, irregularly irregular.
        No chest pain or shortness of breath.
        """,
        query_description="Evaluate atrial fibrillation and management options",
        demographics={"age": 72, "sex": "M", "weight": 85}
    )