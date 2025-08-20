"""
Medical ECG Datasets Module for CoT-RAG Stage 4
==============================================

This module provides comprehensive integration with medical ECG datasets
for training, validation, and clinical testing of ECG classification models.

Supported Datasets:
- PTB-XL: Large ECG database with over 21,000 12-lead ECGs
- MIMIC-IV ECG: Critical care ECG data from MIMIC-IV dataset
- Chapman-Shaoxing: Multi-lead ECG classification dataset
- Synthetic ECG: Generated synthetic ECG data for testing

Components:
- ptb_xl_loader: PTB-XL dataset integration and loading
- mimic_ecg_loader: MIMIC-IV ECG data processing
- chapman_loader: Chapman-Shaoxing dataset handling
- synthetic_ecg_generator: Synthetic ECG data generation

Features:
- Standardized data loading interface
- Clinical annotation processing
- Signal quality assessment
- Multi-lead ECG handling
- Train/validation/test splitting
- Data augmentation capabilities
- Clinical metadata extraction

Usage:
    from datasets import PTBXLLoader, MIMICECGLoader
    
    # Load PTB-XL dataset
    loader = PTBXLLoader(data_path="/path/to/ptbxl")
    train_data, train_labels = loader.load_training_data()
    
    # Load MIMIC-IV ECG data
    mimic_loader = MIMICECGLoader(data_path="/path/to/mimic")
    ecg_data = mimic_loader.load_patient_data(patient_id="12345")
"""

from .ptb_xl_loader import PTBXLLoader, PTBXLConfig, PTBXLRecord
from .mimic_ecg_loader import MIMICECGLoader, MIMICConfig, MIMICRecord
from .chapman_loader import ChapmanLoader, ChapmanConfig
from .synthetic_ecg_generator import SyntheticECGGenerator, SyntheticConfig

__all__ = [
    'PTBXLLoader',
    'PTBXLConfig',
    'PTBXLRecord',
    'MIMICECGLoader', 
    'MIMICConfig',
    'MIMICRecord',
    'ChapmanLoader',
    'ChapmanConfig',
    'SyntheticECGGenerator',
    'SyntheticConfig'
]

__version__ = '1.0.0'
__author__ = 'CoT-RAG Stage 4 Implementation'

# Dataset registry for easy access
DATASET_REGISTRY = {
    'ptb_xl': PTBXLLoader,
    'mimic_iv_ecg': MIMICECGLoader,
    'chapman_shaoxing': ChapmanLoader,
    'synthetic': SyntheticECGGenerator
}

def create_dataset_loader(dataset_name: str, **kwargs):
    """
    Factory function to create dataset loaders.
    
    Args:
        dataset_name: Name of dataset ('ptb_xl', 'mimic_iv_ecg', 'chapman_shaoxing', 'synthetic')
        **kwargs: Dataset-specific configuration parameters
        
    Returns:
        Initialized dataset loader
        
    Example:
        >>> loader = create_dataset_loader('ptb_xl', data_path='/data/ptbxl', sampling_rate=500)
        >>> train_data, train_labels = loader.load_training_data()
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    loader_class = DATASET_REGISTRY[dataset_name]
    return loader_class(**kwargs)