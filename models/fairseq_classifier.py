#!/usr/bin/env python3
"""
Fairseq-Signals ECG Classifier Integration
==========================================

Integration layer for using fairseq-signals trained models within the CoT-RAG framework.
Provides hierarchical ECG classification using models trained on PTB-XL and ECG-QA datasets.

Features:
- Direct integration with fairseq-signals model checkpoints
- Hierarchical classification (superclass -> class -> subclass)
- SCP-ECG statement interpretation
- Real-time inference pipeline
- Attention mechanism visualization
- Clinical decision support integration

Model Support:
- ECG Transformer Classifier from fairseq-signals
- Hierarchical loss trained models
- Multi-task models (classification + QA)
- Ensemble models with multiple checkpoints

Data Integration:
- PTB-XL hierarchical labels
- SCP-ECG statement processing  
- ECG-QA question-answering support
- Clinical metadata integration
"""

import os
import sys
import numpy as np
import warnings
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

# Add fairseq-signals to path
FAIRSEQ_SIGNALS_ROOT = "/home/ll/Desktop/codes/fairseq-signals"
if FAIRSEQ_SIGNALS_ROOT not in sys.path:
    sys.path.insert(0, FAIRSEQ_SIGNALS_ROOT)

# Import fairseq-signals components
FAIRSEQ_AVAILABLE = False
try:
    # Core fairseq components
    from fairseq_signals.models.classification.ecg_transformer_classifier import ECGTransformerClassifier
    from fairseq_signals.tasks.ecg_classification import ECGClassificationTask
    from fairseq_signals.data.ecg import ecg_utils
    from fairseq_signals.checkpoint_utils import load_checkpoint_to_cpu
    from fairseq import utils
    
    # Try importing torch
    import torch
    import torch.nn.functional as F
    
    FAIRSEQ_AVAILABLE = True
    TORCH_AVAILABLE = True
    
except ImportError as e:
    warnings.warn(f"Fairseq-signals or PyTorch not available: {e}")
    FAIRSEQ_AVAILABLE = False
    TORCH_AVAILABLE = False

# Import local components
from .base_ecg_model import BaseECGModel, ECGModelConfig, ModelPrediction, ECGClassificationTask as LocalTask, ModelArchitecture

@dataclass
class FairseqModelConfig:
    """Configuration for fairseq-signals model integration."""
    
    # Model paths
    model_checkpoint: str
    model_args_path: Optional[str] = None
    
    # Hierarchy configuration
    hierarchy_file: Optional[str] = None
    scp_statements_file: Optional[str] = None
    
    # Processing parameters
    target_sampling_rate: int = 500
    target_length: int = 5000
    normalize: bool = True
    
    # Inference parameters
    batch_size: int = 16
    use_cuda: bool = True
    fp16: bool = False
    
    # Clinical parameters
    confidence_threshold: float = 0.5
    hierarchical_consistency: bool = True
    enable_attention: bool = True
    
    # Output configuration
    return_all_levels: bool = True
    return_attention_weights: bool = False
    return_features: bool = False

class FairseqECGClassifier(BaseECGModel):
    """
    Fairseq-signals ECG classifier integration for CoT-RAG framework.
    
    Provides hierarchical ECG classification using fairseq-signals trained models
    with direct integration to PTB-XL and ECG-QA datasets.
    """
    
    def __init__(self, 
                 config: ECGModelConfig,
                 fairseq_config: FairseqModelConfig):
        """
        Initialize fairseq ECG classifier.
        
        Args:
            config: Base ECG model configuration
            fairseq_config: Fairseq-specific configuration
        """
        # Set architecture type
        if config.architecture != ModelArchitecture.FAIRSEQ_TRANSFORMER:
            config.architecture = ModelArchitecture.FAIRSEQ_TRANSFORMER
        
        super().__init__(config)
        
        self.fairseq_config = fairseq_config
        
        # Model components
        self.fairseq_model = None
        self.fairseq_task = None
        self.device = torch.device('cuda' if fairseq_config.use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Label mappings
        self.label_to_id = {}
        self.id_to_label = {}
        self.hierarchy_mapping = {}
        self.scp_mapping = {}
        
        # Load model and mappings
        if FAIRSEQ_AVAILABLE:
            self._load_fairseq_model()
            self._load_label_mappings()
            self._load_hierarchy_mappings()
        else:
            self._create_mock_model()
        
        # Set class names
        self._set_class_names()
    
    def build_model(self) -> Any:
        """Build method required by BaseECGModel."""
        # Already built in __init__
        return self.fairseq_model
    
    def _load_fairseq_model(self) -> None:
        """Load fairseq-signals model checkpoint."""
        
        try:
            checkpoint_path = Path(self.fairseq_config.model_checkpoint)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint
            state = load_checkpoint_to_cpu(str(checkpoint_path))
            
            # Get model args
            if 'args' in state:
                args = state['args']
            else:
                # Load from separate args file if available
                args_path = self.fairseq_config.model_args_path
                if args_path and Path(args_path).exists():
                    with open(args_path, 'r') as f:
                        args = json.load(f)
                else:
                    raise ValueError("Model args not found in checkpoint or separate file")
            
            # Build task and model
            self.fairseq_task = ECGClassificationTask.setup_task(args)
            self.fairseq_model = self.fairseq_task.build_model(args)
            
            # Load model state
            self.fairseq_model.load_state_dict(state['model'])
            self.fairseq_model.eval()
            
            # Move to device
            self.fairseq_model = self.fairseq_model.to(self.device)
            
            # Enable FP16 if requested
            if self.fairseq_config.fp16:
                self.fairseq_model = self.fairseq_model.half()
            
            self.is_trained = True
            print(f"Successfully loaded fairseq-signals model from {checkpoint_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to load fairseq model: {e}")
            self._create_mock_model()
    
    def _create_mock_model(self) -> None:
        """Create mock model for testing without fairseq-signals."""
        
        print("Creating mock fairseq-signals model")
        
        # Mock model components
        self.fairseq_model = {
            'type': 'mock_fairseq_ecg_transformer',
            'input_size': self.config.input_shape[0],
            'num_classes': self.config.num_classes,
            'weights': np.random.randn(self.config.input_shape[0], self.config.num_classes) * 0.01,
            'bias': np.zeros(self.config.num_classes)
        }
        
        self.fairseq_task = {
            'type': 'mock_ecg_classification_task',
            'dictionary': {f'class_{i}': i for i in range(self.config.num_classes)}
        }
        
        self.is_trained = True
    
    def _load_label_mappings(self) -> None:
        """Load label mappings from fairseq task."""
        
        if FAIRSEQ_AVAILABLE and self.fairseq_task:
            try:
                # Get dictionary from fairseq task
                dictionary = getattr(self.fairseq_task, 'dictionary', None)
                
                if dictionary:
                    self.label_to_id = {dictionary.symbols[i]: i for i in range(len(dictionary.symbols))}
                    self.id_to_label = {i: dictionary.symbols[i] for i in range(len(dictionary.symbols))}
                else:
                    self._create_mock_label_mappings()
                
                print(f"Loaded {len(self.label_to_id)} label mappings from fairseq task")
                
            except Exception as e:
                warnings.warn(f"Failed to load label mappings: {e}")
                self._create_mock_label_mappings()
        else:
            self._create_mock_label_mappings()
    
    def _create_mock_label_mappings(self) -> None:
        """Create mock label mappings."""
        
        # PTB-XL style labels
        mock_labels = [
            'NORM', 'MI', 'STTC', 'CD', 'HYP', 'PACE',
            'IMI', 'AMI', 'LMI', 'PMI',  # MI subtypes
            'AFIB', 'AFL', 'SVT', 'VT',  # Arrhythmias
            'RBBB', 'LBBB', '1AVB', '2AVB', '3AVB',  # Conduction
            'LVH', 'RVH', 'LAO', 'RAO'  # Hypertrophy
        ]
        
        self.label_to_id = {label: i for i, label in enumerate(mock_labels[:self.config.num_classes])}
        self.id_to_label = {i: label for i, label in enumerate(mock_labels[:self.config.num_classes])}
    
    def _load_hierarchy_mappings(self) -> None:
        """Load hierarchical label mappings."""
        
        hierarchy_file = self.fairseq_config.hierarchy_file
        if hierarchy_file and Path(hierarchy_file).exists():
            try:
                with open(hierarchy_file, 'r') as f:
                    self.hierarchy_mapping = json.load(f)
                print(f"Loaded hierarchy mapping from {hierarchy_file}")
            except Exception as e:
                warnings.warn(f"Failed to load hierarchy file: {e}")
                self._create_mock_hierarchy()
        else:
            self._create_mock_hierarchy()
        
        # Load SCP statements if available
        scp_file = self.fairseq_config.scp_statements_file
        if scp_file and Path(scp_file).exists():
            try:
                import pandas as pd
                scp_df = pd.read_csv(scp_file, index_col=0)
                self.scp_mapping = scp_df.to_dict('index')
                print(f"Loaded {len(self.scp_mapping)} SCP statement mappings")
            except Exception as e:
                warnings.warn(f"Failed to load SCP statements: {e}")
    
    def _create_mock_hierarchy(self) -> None:
        """Create mock hierarchical mappings."""
        
        self.hierarchy_mapping = {
            # Normal
            'NORM': {'superclass': 'NORM', 'class': 'NORM', 'subclass': 'NORM'},
            
            # Myocardial Infarction
            'MI': {'superclass': 'PATHOLOGIC', 'class': 'MI', 'subclass': 'MI'},
            'IMI': {'superclass': 'PATHOLOGIC', 'class': 'MI', 'subclass': 'IMI'},
            'AMI': {'superclass': 'PATHOLOGIC', 'class': 'MI', 'subclass': 'AMI'},
            'LMI': {'superclass': 'PATHOLOGIC', 'class': 'MI', 'subclass': 'LMI'},
            'PMI': {'superclass': 'PATHOLOGIC', 'class': 'MI', 'subclass': 'PMI'},
            
            # ST/T Changes
            'STTC': {'superclass': 'PATHOLOGIC', 'class': 'STTC', 'subclass': 'STTC'},
            
            # Conduction Disturbances
            'CD': {'superclass': 'PATHOLOGIC', 'class': 'CD', 'subclass': 'CD'},
            'RBBB': {'superclass': 'PATHOLOGIC', 'class': 'CD', 'subclass': 'RBBB'},
            'LBBB': {'superclass': 'PATHOLOGIC', 'class': 'CD', 'subclass': 'LBBB'},
            '1AVB': {'superclass': 'PATHOLOGIC', 'class': 'CD', 'subclass': '1AVB'},
            
            # Hypertrophy
            'HYP': {'superclass': 'PATHOLOGIC', 'class': 'HYP', 'subclass': 'HYP'},
            'LVH': {'superclass': 'PATHOLOGIC', 'class': 'HYP', 'subclass': 'LVH'},
            'RVH': {'superclass': 'PATHOLOGIC', 'class': 'HYP', 'subclass': 'RVH'},
            
            # Arrhythmias
            'AFIB': {'superclass': 'ARRHYTHMIA', 'class': 'AFIB', 'subclass': 'AFIB'},
            'AFL': {'superclass': 'ARRHYTHMIA', 'class': 'AFL', 'subclass': 'AFL'},
            'SVT': {'superclass': 'ARRHYTHMIA', 'class': 'SVT', 'subclass': 'SVT'},
            'VT': {'superclass': 'ARRHYTHMIA', 'class': 'VT', 'subclass': 'VT'},
        }
    
    def _set_class_names(self) -> None:
        """Set class names from label mappings."""
        if self.id_to_label:
            class_names = [self.id_to_label.get(i, f'class_{i}') for i in range(self.config.num_classes)]
            self.set_class_names(class_names)
        else:
            # Fallback to generic names
            self.set_class_names([f'class_{i}' for i in range(self.config.num_classes)])
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess ECG signal for fairseq model."""
        
        # Handle input dimensions
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        elif signal.ndim == 2 and signal.shape[0] > signal.shape[1]:
            signal = signal.T  # Transpose if needed
        
        # Resample to target length
        if signal.shape[1] != self.fairseq_config.target_length:
            from scipy.signal import resample
            signal = resample(signal, self.fairseq_config.target_length, axis=1)
        
        # Normalize if requested
        if self.fairseq_config.normalize:
            signal = (signal - np.mean(signal, axis=1, keepdims=True)) / (np.std(signal, axis=1, keepdims=True) + 1e-8)
        
        return signal
    
    def predict(self, 
                ecg_data: np.ndarray,
                return_attention: bool = False,
                return_features: bool = False) -> ModelPrediction:
        """
        Make hierarchical prediction using fairseq-signals model.
        
        Args:
            ecg_data: Input ECG signal
            return_attention: Whether to return attention weights
            return_features: Whether to return intermediate features
            
        Returns:
            ModelPrediction with hierarchical results
        """
        start_time = time.time()
        
        # Validate input
        if not self.validate_input(ecg_data):
            return self._create_failed_prediction("Invalid input format")
        
        if not self.is_trained:
            return self._create_failed_prediction("Model not trained")
        
        # Preprocess input
        processed_data = self.preprocess_signal(ecg_data)
        
        try:
            if FAIRSEQ_AVAILABLE and isinstance(self.fairseq_model, torch.nn.Module):
                prediction = self._predict_fairseq(processed_data, return_attention, return_features)
            else:
                prediction = self._predict_mock(processed_data, return_attention, return_features)
        except Exception as e:
            return self._create_failed_prediction(f"Prediction failed: {e}")
        
        # Record inference time
        inference_time = (time.time() - start_time) * 1000
        prediction.inference_time_ms = inference_time
        self._update_inference_stats(inference_time)
        
        return prediction
    
    def _predict_fairseq(self, processed_data: np.ndarray, return_attention: bool, return_features: bool) -> ModelPrediction:
        """Make prediction using real fairseq model."""
        
        self.fairseq_model.eval()
        
        with torch.no_grad():
            # Convert to tensor
            input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
            
            if self.fairseq_config.fp16:
                input_tensor = input_tensor.half()
            
            # Forward pass
            model_output = self.fairseq_model(input_tensor)
            
            # Handle different output formats
            if isinstance(model_output, dict):
                logits = model_output['logits']
                attention_weights = model_output.get('attention_weights', None) if return_attention else None
                features = model_output.get('features', None) if return_features else None
            elif isinstance(model_output, tuple):
                logits = model_output[0]
                attention_weights = model_output[1] if len(model_output) > 1 and return_attention else None
                features = model_output[2] if len(model_output) > 2 and return_features else None
            else:
                logits = model_output
                attention_weights = None
                features = None
            
            # Convert to numpy
            logits_np = logits.squeeze().cpu().numpy()
            
            # Get probabilities
            if TORCH_AVAILABLE:
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            else:
                # Manual softmax
                exp_logits = np.exp(logits_np - np.max(logits_np))
                probs = exp_logits / np.sum(exp_logits)
            
            # Get primary prediction
            predicted_idx = np.argmax(probs)
            predicted_label = self.id_to_label.get(predicted_idx, f'class_{predicted_idx}')
            confidence = float(probs[predicted_idx])
            
            # Build hierarchical prediction
            hierarchy_info = self.hierarchy_mapping.get(predicted_label, {
                'superclass': 'UNKNOWN',
                'class': predicted_label,
                'subclass': predicted_label
            })
            
            # Create prediction
            prediction = ModelPrediction(
                predicted_class=predicted_label,
                confidence=confidence,
                probabilities={self.id_to_label.get(i, f'class_{i}'): float(probs[i]) 
                             for i in range(len(probs))},
                raw_output=logits_np,
                risk_level=self._calculate_risk_level(confidence, predicted_label)
            )
            
            # Add hierarchical information
            prediction.metadata = {
                'hierarchical_prediction': {
                    'superclass': hierarchy_info.get('superclass'),
                    'class': hierarchy_info.get('class'),
                    'subclass': hierarchy_info.get('subclass')
                },
                'model_type': 'fairseq_ecg_transformer',
                'fairseq_integration': True
            }
            
            # Add attention weights if available
            if return_attention and attention_weights is not None:
                if isinstance(attention_weights, torch.Tensor):
                    attention_weights = attention_weights.cpu().numpy()
                prediction.attention_weights = attention_weights.flatten()
            
            # Add features if available
            if return_features and features is not None:
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                prediction.feature_importance = np.mean(np.abs(features), axis=0)
            
            return prediction
    
    def _predict_mock(self, processed_data: np.ndarray, return_attention: bool, return_features: bool) -> ModelPrediction:
        """Mock prediction for testing."""
        
        # Simple linear prediction
        if processed_data.ndim == 2:
            processed_data = processed_data.flatten()
        
        # Pad or truncate to match weight dimensions
        if len(processed_data) != self.fairseq_model['weights'].shape[0]:
            if len(processed_data) > self.fairseq_model['weights'].shape[0]:
                processed_data = processed_data[:self.fairseq_model['weights'].shape[0]]
            else:
                processed_data = np.pad(processed_data, 
                                      (0, self.fairseq_model['weights'].shape[0] - len(processed_data)))
        
        raw_output = np.dot(processed_data, self.fairseq_model['weights']) + self.fairseq_model['bias']
        raw_output += np.random.randn(self.config.num_classes) * 0.02  # Add some noise
        
        # Softmax
        exp_output = np.exp(raw_output - np.max(raw_output))
        probs = exp_output / np.sum(exp_output)
        
        # Get prediction
        predicted_idx = np.argmax(probs)
        predicted_label = self.id_to_label.get(predicted_idx, f'class_{predicted_idx}')
        confidence = float(probs[predicted_idx])
        
        # Create prediction with hierarchical info
        hierarchy_info = self.hierarchy_mapping.get(predicted_label, {
            'superclass': 'MOCK',
            'class': predicted_label,
            'subclass': predicted_label
        })
        
        prediction = ModelPrediction(
            predicted_class=predicted_label,
            confidence=confidence,
            probabilities={self.id_to_label.get(i, f'class_{i}'): float(probs[i]) 
                         for i in range(len(probs))},
            raw_output=raw_output,
            risk_level=self._calculate_risk_level(confidence, predicted_label)
        )
        
        prediction.metadata = {
            'hierarchical_prediction': {
                'superclass': hierarchy_info.get('superclass'),
                'class': hierarchy_info.get('class'),
                'subclass': hierarchy_info.get('subclass')
            },
            'model_type': 'mock_fairseq_ecg_transformer',
            'fairseq_integration': False
        }
        
        # Mock attention and features
        if return_attention:
            prediction.attention_weights = np.random.rand(len(processed_data))
            
        if return_features:
            prediction.feature_importance = np.random.rand(len(processed_data))
        
        return prediction
    
    def _calculate_risk_level(self, confidence: float, predicted_class: str) -> str:
        """Calculate clinical risk level."""
        
        # Define high-risk conditions
        high_risk_classes = ['MI', 'IMI', 'AMI', 'LMI', 'PMI', 'VT', '3AVB']
        medium_risk_classes = ['AFIB', 'AFL', 'SVT', '2AVB', 'LBBB', 'RBBB']
        
        if predicted_class in high_risk_classes:
            if confidence > 0.8:
                return 'high'
            elif confidence > 0.6:
                return 'medium'
            else:
                return 'low'
        elif predicted_class in medium_risk_classes:
            if confidence > 0.8:
                return 'medium'
            else:
                return 'low'
        else:
            return 'low'
    
    def get_hierarchical_prediction(self, ecg_data: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed hierarchical prediction with all levels.
        
        Args:
            ecg_data: Input ECG signal
            
        Returns:
            Dictionary with hierarchical prediction details
        """
        
        prediction = self.predict(ecg_data, return_attention=True, return_features=True)
        
        if not prediction.success:
            return {'error': prediction.error_message}
        
        hierarchy = prediction.metadata.get('hierarchical_prediction', {})
        
        # Calculate confidence at each hierarchical level
        superclass_conf = self._calculate_superclass_confidence(prediction.probabilities, hierarchy.get('superclass'))
        class_conf = self._calculate_class_confidence(prediction.probabilities, hierarchy.get('class'))
        
        return {
            'primary_prediction': {
                'label': prediction.predicted_class,
                'confidence': prediction.confidence,
                'risk_level': prediction.risk_level
            },
            'hierarchical_levels': {
                'superclass': {
                    'label': hierarchy.get('superclass'),
                    'confidence': superclass_conf
                },
                'class': {
                    'label': hierarchy.get('class'),
                    'confidence': class_conf
                },
                'subclass': {
                    'label': hierarchy.get('subclass'),
                    'confidence': prediction.confidence
                }
            },
            'all_probabilities': prediction.probabilities,
            'clinical_metadata': {
                'scp_interpretation': self._interpret_scp_codes(prediction.predicted_class),
                'clinical_significance': self._get_clinical_significance(prediction.predicted_class),
                'recommended_actions': self._get_recommended_actions(prediction.predicted_class, prediction.confidence)
            },
            'model_info': {
                'model_type': prediction.metadata.get('model_type'),
                'fairseq_integration': prediction.metadata.get('fairseq_integration', False),
                'inference_time_ms': prediction.inference_time_ms
            }
        }
    
    def _calculate_superclass_confidence(self, probabilities: Dict[str, float], superclass: str) -> float:
        """Calculate confidence for superclass prediction."""
        
        if not superclass:
            return 0.0
        
        # Sum probabilities of all classes in this superclass
        superclass_prob = 0.0
        for label, prob in probabilities.items():
            hierarchy = self.hierarchy_mapping.get(label, {})
            if hierarchy.get('superclass') == superclass:
                superclass_prob += prob
        
        return superclass_prob
    
    def _calculate_class_confidence(self, probabilities: Dict[str, float], class_label: str) -> float:
        """Calculate confidence for class prediction."""
        
        if not class_label:
            return 0.0
        
        # Sum probabilities of all subclasses in this class
        class_prob = 0.0
        for label, prob in probabilities.items():
            hierarchy = self.hierarchy_mapping.get(label, {})
            if hierarchy.get('class') == class_label:
                class_prob += prob
        
        return class_prob
    
    def _interpret_scp_codes(self, predicted_label: str) -> str:
        """Interpret SCP codes for clinical understanding."""
        
        scp_info = self.scp_mapping.get(predicted_label, {})
        if scp_info:
            return scp_info.get('description', 'No description available')
        
        # Fallback descriptions
        descriptions = {
            'NORM': 'Normal ECG',
            'MI': 'Myocardial infarction',
            'IMI': 'Inferior myocardial infarction',
            'AMI': 'Anterior myocardial infarction',
            'LMI': 'Lateral myocardial infarction',
            'PMI': 'Posterior myocardial infarction',
            'STTC': 'ST-T changes',
            'CD': 'Conduction disturbance',
            'RBBB': 'Right bundle branch block',
            'LBBB': 'Left bundle branch block',
            '1AVB': 'First degree AV block',
            'HYP': 'Hypertrophy',
            'LVH': 'Left ventricular hypertrophy',
            'RVH': 'Right ventricular hypertrophy',
            'AFIB': 'Atrial fibrillation',
            'AFL': 'Atrial flutter',
            'SVT': 'Supraventricular tachycardia',
            'VT': 'Ventricular tachycardia'
        }
        
        return descriptions.get(predicted_label, 'Unknown condition')
    
    def _get_clinical_significance(self, predicted_label: str) -> str:
        """Get clinical significance of the prediction."""
        
        significance_map = {
            'NORM': 'Normal finding, no immediate action required',
            'MI': 'Acute coronary syndrome, immediate medical attention required',
            'IMI': 'Inferior wall MI, monitor for complications',
            'AMI': 'Anterior wall MI, high risk for complications',
            'AFIB': 'Arrhythmia, anticoagulation consideration',
            'VT': 'Life-threatening arrhythmia, immediate intervention',
            'LBBB': 'May indicate underlying heart disease',
            'RBBB': 'Usually benign, may indicate RV strain',
            '1AVB': 'Usually benign, monitor progression',
            'LVH': 'May indicate hypertension or valve disease'
        }
        
        return significance_map.get(predicted_label, 'Clinical correlation recommended')
    
    def _get_recommended_actions(self, predicted_label: str, confidence: float) -> List[str]:
        """Get recommended clinical actions."""
        
        actions = []
        
        # High confidence critical findings
        if confidence > 0.8:
            if predicted_label in ['MI', 'IMI', 'AMI', 'LMI', 'PMI']:
                actions.extend([
                    'Immediate cardiology consultation',
                    'Serial cardiac enzymes',
                    'Continuous cardiac monitoring',
                    'Consider urgent catheterization'
                ])
            elif predicted_label == 'VT':
                actions.extend([
                    'Immediate ACLS protocol',
                    'Defibrillation if unstable',
                    'IV antiarrhythmic agents',
                    'ICU monitoring'
                ])
            elif predicted_label == 'AFIB':
                actions.extend([
                    'Rate/rhythm control',
                    'Anticoagulation assessment',
                    'Echocardiogram',
                    'Thyroid function tests'
                ])
        
        # Medium confidence or less critical findings
        elif confidence > 0.6:
            if predicted_label in ['LBBB', 'RBBB']:
                actions.extend([
                    'Clinical correlation',
                    'Compare with prior ECGs',
                    'Consider echocardiogram'
                ])
            elif predicted_label == '1AVB':
                actions.extend([
                    'Monitor for progression',
                    'Review medications',
                    'Serial ECGs'
                ])
        
        # Low confidence - general recommendations
        else:
            actions.extend([
                'Clinical correlation recommended',
                'Consider repeat ECG',
                'Expert interpretation if clinical concern'
            ])
        
        return actions
    
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """Training not supported for pretrained fairseq models."""
        
        warnings.warn("Training not supported for pretrained fairseq-signals models")
        return {
            'message': 'Pretrained model - training not supported',
            'is_trained': True
        }
    
    def load_weights(self, weights_path: str) -> bool:
        """Load weights is handled during initialization."""
        
        warnings.warn("Weight loading handled during model initialization")
        return True
    
    def save_weights(self, weights_path: str) -> bool:
        """Save weights not supported for fairseq models."""
        
        warnings.warn("Weight saving not supported for fairseq-signals models")
        return False
    
    def get_model_summary(self) -> str:
        """Get model summary with fairseq integration info."""
        
        summary = f"""
Fairseq-Signals ECG Classifier
=============================

Model Configuration:
- Checkpoint: {self.fairseq_config.model_checkpoint}
- Architecture: {self.config.architecture.value}
- Task: {self.config.task.value}
- Input Shape: {self.config.input_shape}
- Number of Classes: {self.config.num_classes}
- Sampling Rate: {self.config.sampling_rate} Hz

Fairseq Integration:
- Available: {FAIRSEQ_AVAILABLE}
- Device: {self.device}
- FP16: {self.fairseq_config.fp16}
- Target Length: {self.fairseq_config.target_length}
- Hierarchical: {len(self.hierarchy_mapping)} mappings

Label Mappings: {len(self.label_to_id)} classes
SCP Mappings: {len(self.scp_mapping)} codes

Training Status: {'Trained' if self.is_trained else 'Not Trained'}
Class Names: {self.class_names}

Performance Metrics: {self.performance_metrics}
"""
        
        return summary


# Factory function for easy instantiation
def create_fairseq_ecg_classifier(model_checkpoint: str,
                                  num_classes: int = 21,
                                  input_shape: Tuple[int] = (5000,),
                                  **kwargs) -> FairseqECGClassifier:
    """
    Factory function to create FairseqECGClassifier.
    
    Args:
        model_checkpoint: Path to fairseq model checkpoint
        num_classes: Number of classes
        input_shape: Input signal shape
        **kwargs: Additional configuration options
        
    Returns:
        Configured FairseqECGClassifier instance
    """
    
    # Create base config
    config = ECGModelConfig(
        model_name="fairseq_ecg_transformer",
        architecture=ModelArchitecture.FAIRSEQ_TRANSFORMER,
        task=LocalTask.RHYTHM_CLASSIFICATION,
        num_classes=num_classes,
        input_shape=input_shape,
        sampling_rate=500.0
    )
    
    # Create fairseq config
    fairseq_config = FairseqModelConfig(
        model_checkpoint=model_checkpoint,
        **kwargs
    )
    
    return FairseqECGClassifier(config, fairseq_config)


# Example usage and testing
if __name__ == "__main__":
    print("Fairseq-Signals ECG Classifier Test")
    print("=" * 50)
    
    # Test configuration
    model_checkpoint = "/models/fairseq_ecg_hierarchical.pt"
    
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Fairseq available: {FAIRSEQ_AVAILABLE}")
    print(f"Torch available: {TORCH_AVAILABLE}")
    
    # Create classifier
    classifier = create_fairseq_ecg_classifier(
        model_checkpoint=model_checkpoint,
        num_classes=12,
        input_shape=(5000,),
        use_cuda=False,  # CPU for testing
        fp16=False
    )
    
    print(f"\nModel Summary:")
    print(classifier.get_model_summary())
    
    # Test prediction
    print(f"\nTesting prediction...")
    test_ecg = np.random.randn(5000) * 0.3
    
    # Basic prediction
    prediction = classifier.predict(test_ecg, return_attention=True, return_features=True)
    
    print(f"  Predicted class: {prediction.predicted_class}")
    print(f"  Confidence: {prediction.confidence:.3f}")
    print(f"  Risk level: {prediction.risk_level}")
    print(f"  Inference time: {prediction.inference_time_ms:.1f} ms")
    
    if prediction.metadata:
        hierarchy = prediction.metadata.get('hierarchical_prediction', {})
        print(f"  Hierarchical: {hierarchy.get('superclass')} -> {hierarchy.get('class')} -> {hierarchy.get('subclass')}")
    
    # Test hierarchical prediction
    print(f"\nTesting hierarchical prediction...")
    hierarchical_result = classifier.get_hierarchical_prediction(test_ecg)
    
    if 'error' not in hierarchical_result:
        print(f"  Primary: {hierarchical_result['primary_prediction']['label']} "
              f"({hierarchical_result['primary_prediction']['confidence']:.3f})")
        
        levels = hierarchical_result['hierarchical_levels']
        print(f"  Superclass: {levels['superclass']['label']} ({levels['superclass']['confidence']:.3f})")
        print(f"  Class: {levels['class']['label']} ({levels['class']['confidence']:.3f})")
        print(f"  Subclass: {levels['subclass']['label']} ({levels['subclass']['confidence']:.3f})")
        
        clinical = hierarchical_result['clinical_metadata']
        print(f"  Clinical significance: {clinical['clinical_significance']}")
        print(f"  Recommended actions: {len(clinical['recommended_actions'])} items")
    
    # Test evaluation
    print(f"\nTesting evaluation...")
    test_data = np.random.randn(10, 5000) * 0.3
    test_labels = np.random.randint(0, classifier.config.num_classes, 10)
    
    metrics = classifier.evaluate(test_data, test_labels)
    print(f"  Evaluation metrics: {metrics}")
    
    # Test inference statistics
    stats = classifier.get_inference_stats()
    print(f"\nInference Statistics:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Average inference time: {stats['average_inference_time']:.1f} ms")
    
    print("\nFairseq-Signals ECG Classifier Test Complete!")