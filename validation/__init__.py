"""
Clinical Validation Framework for CoT-RAG Stage 4
================================================

This module provides comprehensive clinical validation capabilities for ECG
classification models, ensuring they meet medical standards and regulatory
requirements for clinical deployment.

Components:
- clinical_metrics: Medical performance metrics and validation
- hierarchy_validation: Hierarchical classification validation
- expert_comparison: Comparison against expert annotations
- model_interpretability: Attention visualization and analysis

Features:
- Clinical performance metrics (sensitivity, specificity, PPV, NPV)
- Regulatory compliance validation
- Expert annotation comparison
- Model interpretability and explainability
- Statistical significance testing
- Clinical confidence assessment

Clinical Applications:
- FDA/CE marking validation
- Clinical trial integration
- Expert validation workflows
- Performance monitoring
- Safety assessment

Usage:
    from validation import ClinicalMetrics, ExpertComparison
    
    metrics = ClinicalMetrics()
    results = metrics.evaluate_clinical_performance(predictions, ground_truth)
"""

from .clinical_metrics import ClinicalMetrics, ClinicalValidationResult
from .hierarchy_validation import HierarchyValidator, HierarchicalMetrics
from .expert_comparison import ExpertComparison, ExpertAgreement
from .model_interpretability import InterpretabilityAnalyzer, AttentionAnalysis

__all__ = [
    'ClinicalMetrics',
    'ClinicalValidationResult',
    'HierarchyValidator',
    'HierarchicalMetrics', 
    'ExpertComparison',
    'ExpertAgreement',
    'InterpretabilityAnalyzer',
    'AttentionAnalysis'
]

__version__ = '1.0.0'
__author__ = 'CoT-RAG Stage 4 Implementation'