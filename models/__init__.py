"""
Deep Learning Models Module for CoT-RAG Stage 4
==============================================

This module provides state-of-the-art deep learning models for ECG analysis
in the CoT-RAG Stage 4 Medical Domain Integration, including:

- SE-ResNet (Squeeze-and-Excitation ResNet) for ECG classification
- HAN (Hierarchical Attention Network) for temporal analysis
- Model ensemble management and orchestration
- Base ECG model interface for standardization

Components:
- base_ecg_model: Abstract base class for ECG models
- se_resnet_classifier: SE-ResNet implementation for ECG data
- han_classifier: Hierarchical Attention Network for ECG
- ensemble_manager: Multi-model orchestration system
- model_registry: Model loading and management

Features:
- Production-ready model implementations
- Clinical-grade performance optimization
- Real-time inference capabilities
- Model versioning and A/B testing support
- Interpretability and attention visualization
- Integration with CoT-RAG Stage 3 reasoning

Usage:
    from models import SEResNetClassifier, HANClassifier, EnsembleManager
    
    # Initialize models
    se_resnet = SEResNetClassifier(num_classes=5)
    han_model = HANClassifier(num_classes=5)
    
    # Create ensemble
    ensemble = EnsembleManager([se_resnet, han_model])
    
    # Make predictions
    predictions = ensemble.predict(ecg_data)
"""

from .base_ecg_model import BaseECGModel, ECGModelConfig, ModelPrediction
from .se_resnet_classifier import SEResNetClassifier, SEResNetConfig
from .han_classifier import HANClassifier, HANConfig
from .ensemble_manager import EnsembleManager, EnsembleConfig
from .model_registry import ModelRegistry, ModelInfo

__all__ = [
    'BaseECGModel',
    'ECGModelConfig',
    'ModelPrediction',
    'SEResNetClassifier',
    'SEResNetConfig',
    'HANClassifier', 
    'HANConfig',
    'EnsembleManager',
    'EnsembleConfig',
    'ModelRegistry',
    'ModelInfo'
]

__version__ = '1.0.0'
__author__ = 'CoT-RAG Stage 4 Implementation'

# Model type registry
MODEL_TYPES = {
    'se_resnet': SEResNetClassifier,
    'han': HANClassifier,
    'ensemble': EnsembleManager
}

def create_model(model_type: str, **kwargs):
    """
    Factory function to create ECG models.
    
    Args:
        model_type: Type of model to create ('se_resnet', 'han', 'ensemble')
        **kwargs: Model-specific configuration parameters
        
    Returns:
        Initialized ECG model instance
        
    Example:
        >>> se_resnet = create_model('se_resnet', num_classes=5, input_length=5000)
        >>> han_model = create_model('han', num_classes=5, sequence_length=1000)
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(MODEL_TYPES.keys())}")
    
    model_class = MODEL_TYPES[model_type]
    return model_class(**kwargs)