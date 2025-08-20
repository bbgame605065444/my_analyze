"""
ECG Processing Module for CoT-RAG Stage 4
========================================

This module provides ECG signal processing capabilities for the CoT-RAG Stage 4
Medical Domain Integration, including:

- Signal preprocessing and noise reduction
- Multi-lead ECG analysis
- Feature extraction (time and frequency domain)
- Signal quality assessment
- ECG normalization and standardization

Components:
- signal_preprocessing: Core ECG signal cleaning and preprocessing
- feature_extraction: Time/frequency domain feature computation  
- lead_analysis: 12-lead ECG specific processing
- quality_assessment: Signal quality validation

Usage:
    from ecg_processing import ECGSignalProcessor
    
    processor = ECGSignalProcessor()
    processed_ecg = processor.preprocess_signal(raw_ecg_data)
"""

from .signal_preprocessing import ECGSignalProcessor
from .feature_extraction import ECGFeatureExtractor
from .lead_analysis import MultiLeadAnalyzer
from .quality_assessment import SignalQualityAssessor

__all__ = [
    'ECGSignalProcessor',
    'ECGFeatureExtractor', 
    'MultiLeadAnalyzer',
    'SignalQualityAssessor'
]

__version__ = '1.0.0'
__author__ = 'CoT-RAG Stage 4 Implementation'