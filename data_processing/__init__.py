"""
Enhanced Data Processing Module for CoT-RAG
==========================================

Comprehensive data processing capabilities for ECG classification and
clinical text analysis, with integration for PTB-XL, ECGQA, and fairseq-signals.

Components:
- ecg_loader: Enhanced ECG data loading and preprocessing
- clinical_loader: Clinical text and metadata processing
- fairseq_loader: Integration with fairseq-signals framework
- hierarchy_builder: Medical taxonomy and hierarchy construction

Features:
- PTB-XL dataset integration with SCP-ECG statements
- ECGQA dataset support for question-answering tasks
- Hierarchical classification support
- fairseq-signals compatibility layer
- Clinical standards compliance (WFDB, DICOM)

Usage:
    from data_processing import PTBXLLoader, ECGQALoader, FairseqECGLoader
    
    # Load PTB-XL with hierarchical labels
    ptbxl_loader = PTBXLLoader()
    data = ptbxl_loader.load_hierarchical_data()
    
    # Load ECGQA for question-answering
    ecgqa_loader = ECGQALoader()
    qa_data = ecgqa_loader.load_qa_pairs()
"""

from .ecg_loader import ECGLoader, PTBXLLoader, ECGQALoader
from .clinical_loader import ClinicalDataLoader, PatientDataLoader
from .fairseq_loader import FairseqECGLoader, FairseqHierarchicalClassifier
from .hierarchy_builder import MedicalHierarchyBuilder, SCPStatementProcessor

__all__ = [
    'ECGLoader',
    'PTBXLLoader', 
    'ECGQALoader',
    'ClinicalDataLoader',
    'PatientDataLoader',
    'FairseqECGLoader',
    'FairseqHierarchicalClassifier',
    'MedicalHierarchyBuilder',
    'SCPStatementProcessor'
]

__version__ = '2.0.0'
__author__ = 'CoT-RAG Enhanced Implementation'