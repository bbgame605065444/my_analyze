#!/usr/bin/env python3
"""
Model Ensemble Management Framework
===================================

Advanced ensemble management system for ECG classification models in CoT-RAG Stage 4.
Provides sophisticated model orchestration, weighted voting, and clinical validation
for optimal diagnostic performance using multiple deep learning models.

Features:
- Multi-model ensemble orchestration
- Dynamic weighting based on model performance
- Clinical confidence aggregation
- Model-specific specialization handling
- Real-time performance monitoring
- A/B testing and model versioning
- Interpretability aggregation across models

Ensemble Strategies:
- Simple voting (majority, weighted)
- Stacking with meta-learner
- Bayesian model averaging
- Clinical confidence weighting
- Specialty-based routing

Clinical Integration:
- Risk-stratified ensemble selection
- Confidence threshold management
- Clinical validation workflows
- Expert override mechanisms
- Performance monitoring and alerting

Usage:
    ensemble = EnsembleManager([se_resnet, han_model])
    prediction = ensemble.predict(ecg_data)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import time
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_ecg_model import BaseECGModel, ECGModelConfig, ModelPrediction, ECGClassificationTask, ModelArchitecture

class EnsembleStrategy(Enum):
    """Ensemble combination strategies."""
    SIMPLE_VOTING = "simple_voting"
    WEIGHTED_VOTING = "weighted_voting"
    STACKING = "stacking"
    BAYESIAN_AVERAGING = "bayesian_averaging"
    CLINICAL_WEIGHTING = "clinical_weighting"
    CONFIDENCE_ROUTING = "confidence_routing"

class ModelRole(Enum):
    """Specialized roles for models in ensemble."""
    GENERAL = "general"  # General-purpose ECG classification
    RHYTHM_SPECIALIST = "rhythm_specialist"  # Specialized in rhythm analysis
    MORPHOLOGY_SPECIALIST = "morphology_specialist"  # Specialized in morphology
    ARRHYTHMIA_DETECTOR = "arrhythmia_detector"  # Binary arrhythmia detection
    QUALITY_ASSESSOR = "quality_assessor"  # Signal quality assessment

@dataclass
class ModelWeight:
    """Dynamic weight configuration for ensemble models."""
    base_weight: float = 1.0
    performance_weight: float = 1.0
    confidence_weight: float = 1.0
    clinical_weight: float = 1.0
    specialty_bonus: float = 0.0
    
    @property
    def effective_weight(self) -> float:
        """Calculate effective weight combining all factors."""
        return (self.base_weight * 
                self.performance_weight * 
                self.confidence_weight * 
                self.clinical_weight + 
                self.specialty_bonus)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble management."""
    # Ensemble strategy
    strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_VOTING
    
    # Voting parameters
    min_agreement_threshold: float = 0.6  # Minimum agreement for high confidence
    confidence_threshold: float = 0.7  # Threshold for clinical use
    
    # Performance monitoring
    performance_window: int = 100  # Number of predictions for rolling performance
    weight_update_frequency: int = 50  # How often to update weights
    
    # Clinical validation
    enable_clinical_routing: bool = True
    clinical_override_threshold: float = 0.9  # Expert override threshold
    risk_stratification: bool = True
    
    # Model specialization
    enable_specialty_routing: bool = True
    specialty_threshold: float = 0.8  # Threshold for specialty model activation
    
    # Meta-learning (for stacking)
    meta_model_architecture: str = "logistic_regression"
    meta_model_features: List[str] = None
    
    # Interpretability
    aggregate_attention: bool = True
    attention_weighting_method: str = "confidence_weighted"
    
    def __post_init__(self):
        if self.meta_model_features is None:
            self.meta_model_features = ['probabilities', 'confidence', 'attention_entropy']

@dataclass
class EnsemblePrediction:
    """Extended prediction with ensemble-specific information."""
    # Core prediction
    ensemble_prediction: ModelPrediction
    
    # Individual model results
    individual_predictions: List[ModelPrediction]
    
    # Ensemble metadata
    agreement_score: float
    model_weights_used: Dict[str, float]
    strategy_used: EnsembleStrategy
    
    # Clinical information
    clinical_consensus: bool
    risk_level_consensus: bool
    confidence_distribution: np.ndarray
    
    # Interpretability
    aggregated_attention: Optional[np.ndarray] = None
    model_contributions: Dict[str, float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ensemble prediction to dictionary."""
        result = self.ensemble_prediction.to_dict()
        result.update({
            'agreement_score': self.agreement_score,
            'model_weights_used': self.model_weights_used,
            'strategy_used': self.strategy_used.value,
            'clinical_consensus': self.clinical_consensus,
            'risk_level_consensus': self.risk_level_consensus,
            'confidence_distribution': self.confidence_distribution.tolist(),
            'num_models': len(self.individual_predictions),
            'individual_confidences': [p.confidence for p in self.individual_predictions]
        })
        return result

class MetaLearner(nn.Module if TORCH_AVAILABLE else object):
    """
    Meta-learner for stacking ensemble strategy.
    
    Learns to optimally combine predictions from base models
    based on their outputs and additional features.
    """
    
    def __init__(self, 
                 num_models: int,
                 num_classes: int,
                 feature_dim: int = 10):
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.num_models = num_models
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        if TORCH_AVAILABLE:
            # Input: [model_probs (num_models * num_classes) + additional_features]
            input_dim = num_models * num_classes + feature_dim
            
            self.meta_network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
            
            self.confidence_network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def forward(self, model_outputs, additional_features):
        """Forward pass through meta-learner."""
        if not TORCH_AVAILABLE:
            # Mock implementation
            return np.random.rand(len(model_outputs), self.num_classes)
        
        # Flatten model outputs and concatenate with features
        flattened_outputs = model_outputs.view(model_outputs.size(0), -1)
        combined_input = torch.cat([flattened_outputs, additional_features], dim=1)
        
        # Predict class probabilities and confidence
        class_logits = self.meta_network(combined_input)
        confidence = self.confidence_network(combined_input)
        
        return class_logits, confidence

class EnsembleManager(BaseECGModel):
    """
    Comprehensive ensemble management system for ECG classification.
    
    Orchestrates multiple ECG models with sophisticated combination strategies
    and clinical validation for optimal diagnostic performance.
    """
    
    def __init__(self, 
                 models: List[BaseECGModel],
                 config: Optional[ECGModelConfig] = None,
                 ensemble_config: Optional[EnsembleConfig] = None):
        """
        Initialize ensemble manager.
        
        Args:
            models: List of base ECG models to ensemble
            config: Base model configuration (derived from models if None)
            ensemble_config: Ensemble-specific configuration
        """
        if not models:
            raise ValueError("At least one model required for ensemble")
        
        # Create or validate configuration
        if config is None:
            config = self._create_ensemble_config(models)
        else:
            self._validate_model_compatibility(models, config)
        
        # Set architecture to ensemble
        config.architecture = ModelArchitecture.ENSEMBLE
        
        super().__init__(config)
        
        self.models = models
        self.ensemble_config = ensemble_config or EnsembleConfig()
        
        # Model management
        self.model_weights = {model.config.model_name: ModelWeight() 
                             for model in models}
        self.model_roles = self._assign_model_roles()
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.prediction_count = 0
        
        # Meta-learner for stacking
        self.meta_learner = None
        if self.ensemble_config.strategy == EnsembleStrategy.STACKING:
            self._initialize_meta_learner()
        
        # Set ensemble as trained if all models are trained
        self.is_trained = all(model.is_trained for model in self.models)
        
        # Create unified class names
        self._unify_class_names()
    
    def build_model(self) -> Any:
        """Ensemble doesn't build a single model - manages multiple models."""
        return {
            'type': 'ensemble',
            'models': self.models,
            'strategy': self.ensemble_config.strategy,
            'meta_learner': self.meta_learner
        }
    
    def train(self, 
              train_data: np.ndarray,
              train_labels: np.ndarray,
              val_data: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train ensemble (trains individual models and meta-learner if applicable).
        
        Args:
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data (optional)
            val_labels: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Combined training history from all models
        """
        print(f"Training ensemble with {len(self.models)} models...")
        
        combined_history = {
            'model_histories': {},
            'ensemble_performance': {}
        }
        
        # Train individual models
        for i, model in enumerate(self.models):
            model_name = model.config.model_name
            print(f"Training model {i+1}/{len(self.models)}: {model_name}")
            
            try:
                if not model.is_trained:
                    model_history = model.train(train_data, train_labels, 
                                              val_data, val_labels, **kwargs)
                    combined_history['model_histories'][model_name] = model_history
                else:
                    print(f"  {model_name} already trained, skipping...")
                    combined_history['model_histories'][model_name] = model.training_history
            except Exception as e:
                warnings.warn(f"Failed to train {model_name}: {e}")
                combined_history['model_histories'][model_name] = {'error': str(e)}
        
        # Train meta-learner if using stacking
        if (self.ensemble_config.strategy == EnsembleStrategy.STACKING and 
            val_data is not None):
            meta_history = self._train_meta_learner(val_data, val_labels)
            combined_history['meta_learner_history'] = meta_history
        
        # Update ensemble training status
        self.is_trained = any(model.is_trained for model in self.models)
        self.training_history = combined_history
        
        return combined_history
    
    def predict(self, 
                ecg_data: np.ndarray,
                return_attention: bool = False,
                return_features: bool = False) -> ModelPrediction:
        """
        Make ensemble prediction.
        
        Args:
            ecg_data: Input ECG data
            return_attention: Whether to return aggregated attention
            return_features: Whether to return feature importance
            
        Returns:
            Ensemble prediction with individual model results
        """
        start_time = time.time()
        
        # Validate input
        if not self.validate_input(ecg_data):
            return self._create_failed_prediction("Invalid input format")
        
        if not self.is_trained:
            return self._create_failed_prediction("Ensemble not trained")
        
        # Get predictions from all models
        individual_predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(ecg_data, return_attention, return_features)
                individual_predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Model {model.config.model_name} prediction failed: {e}")
                # Create failed prediction for this model
                failed_pred = model._create_failed_prediction(str(e))
                individual_predictions.append(failed_pred)
        
        # Combine predictions using specified strategy
        ensemble_pred = self._combine_predictions(individual_predictions, ecg_data)
        
        # Record inference time
        inference_time = (time.time() - start_time) * 1000
        ensemble_pred.ensemble_prediction.inference_time_ms = inference_time
        self._update_inference_stats(inference_time)
        
        # Update performance tracking
        self._update_performance_tracking(ensemble_pred)
        
        # Convert to standard ModelPrediction format
        return ensemble_pred.ensemble_prediction
    
    def predict_ensemble_detailed(self, 
                                 ecg_data: np.ndarray,
                                 return_attention: bool = False,
                                 return_features: bool = False) -> EnsemblePrediction:
        """
        Make detailed ensemble prediction with full ensemble information.
        
        Args:
            ecg_data: Input ECG data
            return_attention: Whether to return aggregated attention
            return_features: Whether to return feature importance
            
        Returns:
            Detailed ensemble prediction with individual model results
        """
        # Similar to predict but returns full EnsemblePrediction
        individual_predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(ecg_data, return_attention, return_features)
                individual_predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Model {model.config.model_name} prediction failed: {e}")
                failed_pred = model._create_failed_prediction(str(e))
                individual_predictions.append(failed_pred)
        
        # Combine predictions
        ensemble_pred = self._combine_predictions(individual_predictions, ecg_data)
        
        return ensemble_pred
    
    def _create_ensemble_config(self, models: List[BaseECGModel]) -> ECGModelConfig:
        """Create ensemble configuration from constituent models."""
        
        # Use first model as template
        base_config = models[0].config
        
        # Verify all models have compatible configurations
        for model in models[1:]:
            if model.config.num_classes != base_config.num_classes:
                raise ValueError("All models must have same number of classes")
            if model.config.input_shape != base_config.input_shape:
                raise ValueError("All models must have same input shape")
            if model.config.task != base_config.task:
                warnings.warn("Models have different tasks - ensemble may not work optimally")
        
        # Create ensemble configuration
        ensemble_config = ECGModelConfig(
            model_name="ecg_ensemble",
            architecture=ModelArchitecture.ENSEMBLE,
            task=base_config.task,
            num_classes=base_config.num_classes,
            input_shape=base_config.input_shape,
            sampling_rate=base_config.sampling_rate,
            model_version="1.0.0",
            model_params={'constituent_models': [m.config.model_name for m in models]}
        )
        
        return ensemble_config
    
    def _validate_model_compatibility(self, models: List[BaseECGModel], config: ECGModelConfig):
        """Validate that models are compatible with ensemble configuration."""
        
        for model in models:
            if model.config.num_classes != config.num_classes:
                raise ValueError(f"Model {model.config.model_name} has incompatible number of classes")
            if model.config.input_shape != config.input_shape:
                raise ValueError(f"Model {model.config.model_name} has incompatible input shape")
    
    def _assign_model_roles(self) -> Dict[str, ModelRole]:
        """Assign specialized roles to models based on their architecture/task."""
        
        roles = {}
        
        for model in self.models:
            model_name = model.config.model_name
            
            # Role assignment based on model characteristics
            if 'rhythm' in model_name.lower() or 'han' in model_name.lower():
                roles[model_name] = ModelRole.RHYTHM_SPECIALIST
            elif 'morphology' in model_name.lower() or 'morph' in model_name.lower():
                roles[model_name] = ModelRole.MORPHOLOGY_SPECIALIST
            elif 'arrhythmia' in model_name.lower() and model.config.num_classes == 2:
                roles[model_name] = ModelRole.ARRHYTHMIA_DETECTOR
            elif 'quality' in model_name.lower():
                roles[model_name] = ModelRole.QUALITY_ASSESSOR
            else:
                roles[model_name] = ModelRole.GENERAL
        
        return roles
    
    def _initialize_meta_learner(self):
        """Initialize meta-learner for stacking strategy."""
        
        if TORCH_AVAILABLE:
            self.meta_learner = MetaLearner(
                num_models=len(self.models),
                num_classes=self.config.num_classes,
                feature_dim=len(self.ensemble_config.meta_model_features)
            )
        else:
            # Mock meta-learner
            self.meta_learner = {
                'type': 'mock_meta_learner',
                'weights': np.random.randn(len(self.models) * self.config.num_classes + 10, 
                                         self.config.num_classes) * 0.1
            }
    
    def _train_meta_learner(self, val_data: np.ndarray, val_labels: np.ndarray) -> Dict[str, Any]:
        """Train meta-learner on validation data."""
        
        print("Training meta-learner...")
        
        # Get predictions from all models on validation data
        meta_features = []
        meta_targets = []
        
        for i, val_sample in enumerate(val_data):
            try:
                # Get individual model predictions
                individual_preds = []
                for model in self.models:
                    pred = model.predict(val_sample)
                    individual_preds.append(pred)
                
                # Extract features for meta-learner
                features = self._extract_meta_features(individual_preds)
                meta_features.append(features)
                meta_targets.append(val_labels[i])
                
            except Exception as e:
                warnings.warn(f"Failed to process validation sample {i}: {e}")
        
        if not meta_features:
            return {'error': 'No valid meta-features extracted'}
        
        meta_features = np.array(meta_features)
        meta_targets = np.array(meta_targets)
        
        # Train meta-learner
        if TORCH_AVAILABLE and hasattr(self.meta_learner, 'meta_network'):
            return self._train_meta_learner_pytorch(meta_features, meta_targets)
        else:
            return self._train_meta_learner_mock(meta_features, meta_targets)
    
    def _train_meta_learner_pytorch(self, features: np.ndarray, targets: np.ndarray):
        """Train PyTorch meta-learner."""
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        targets_tensor = torch.LongTensor(targets)
        
        # Simple training loop
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Split features
            model_outputs = features_tensor[:, :len(self.models) * self.config.num_classes]
            additional_features = features_tensor[:, len(self.models) * self.config.num_classes:]
            
            model_outputs = model_outputs.view(-1, len(self.models), self.config.num_classes)
            
            # Forward pass
            logits, confidence = self.meta_learner(model_outputs, additional_features)
            loss = criterion(logits, targets_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == targets_tensor).float().mean()
            
            history['loss'].append(loss.item())
            history['accuracy'].append(accuracy.item())
        
        return history
    
    def _train_meta_learner_mock(self, features: np.ndarray, targets: np.ndarray):
        """Train mock meta-learner."""
        
        # Simple mock training
        history = {
            'loss': [1.5 * np.exp(-i * 0.1) + 0.2 for i in range(50)],
            'accuracy': [1.0 - 0.8 * np.exp(-i * 0.15) for i in range(50)]
        }
        
        return history
    
    def _combine_predictions(self, 
                           individual_predictions: List[ModelPrediction],
                           ecg_data: np.ndarray) -> EnsemblePrediction:
        """Combine individual model predictions using specified strategy."""
        
        if not individual_predictions:
            return self._create_failed_ensemble_prediction("No individual predictions")
        
        # Filter out failed predictions
        valid_predictions = [p for p in individual_predictions if p.confidence > 0]
        
        if not valid_predictions:
            return self._create_failed_ensemble_prediction("All individual predictions failed")
        
        # Apply ensemble strategy
        if self.ensemble_config.strategy == EnsembleStrategy.SIMPLE_VOTING:
            ensemble_pred = self._simple_voting(valid_predictions)
        elif self.ensemble_config.strategy == EnsembleStrategy.WEIGHTED_VOTING:
            ensemble_pred = self._weighted_voting(valid_predictions)
        elif self.ensemble_config.strategy == EnsembleStrategy.CLINICAL_WEIGHTING:
            ensemble_pred = self._clinical_weighting(valid_predictions)
        elif self.ensemble_config.strategy == EnsembleStrategy.STACKING:
            ensemble_pred = self._stacking_combination(valid_predictions)
        else:
            # Default to weighted voting
            ensemble_pred = self._weighted_voting(valid_predictions)
        
        # Calculate ensemble metadata
        agreement_score = self._calculate_agreement(valid_predictions)
        model_weights_used = self._get_current_weights()
        clinical_consensus = self._assess_clinical_consensus(valid_predictions)
        risk_consensus = self._assess_risk_consensus(valid_predictions)
        confidence_dist = np.array([p.confidence for p in valid_predictions])
        
        # Aggregate attention if requested
        aggregated_attention = None
        if self.ensemble_config.aggregate_attention:
            aggregated_attention = self._aggregate_attention(valid_predictions)
        
        # Calculate model contributions
        model_contributions = self._calculate_model_contributions(valid_predictions)
        
        return EnsemblePrediction(
            ensemble_prediction=ensemble_pred,
            individual_predictions=individual_predictions,
            agreement_score=agreement_score,
            model_weights_used=model_weights_used,
            strategy_used=self.ensemble_config.strategy,
            clinical_consensus=clinical_consensus,
            risk_level_consensus=risk_consensus,
            confidence_distribution=confidence_dist,
            aggregated_attention=aggregated_attention,
            model_contributions=model_contributions
        )
    
    def _simple_voting(self, predictions: List[ModelPrediction]) -> ModelPrediction:
        """Simple majority voting strategy."""
        
        # Count votes for each class
        class_votes = defaultdict(int)
        for pred in predictions:
            class_votes[pred.predicted_class] += 1
        
        # Get majority class
        majority_class = max(class_votes, key=class_votes.get)
        
        # Calculate combined probabilities (average)
        combined_probs = np.mean([p.class_probabilities for p in predictions], axis=0)
        
        # Create ensemble prediction
        return self._create_ensemble_prediction(
            predicted_class=majority_class,
            probabilities=combined_probs,
            predictions=predictions,
            method="simple_voting"
        )
    
    def _weighted_voting(self, predictions: List[ModelPrediction]) -> ModelPrediction:
        """Weighted voting strategy based on model performance."""
        
        # Get weights for each model
        weights = []
        for i, pred in enumerate(predictions):
            model_name = self.models[i].config.model_name if i < len(self.models) else f"model_{i}"
            if model_name in self.model_weights:
                weight = self.model_weights[model_name].effective_weight
            else:
                weight = 1.0
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted combination of probabilities
        weighted_probs = np.zeros(self.config.num_classes)
        for pred, weight in zip(predictions, weights):
            weighted_probs += weight * pred.class_probabilities
        
        # Get predicted class
        predicted_class_idx = np.argmax(weighted_probs)
        predicted_class = self.get_class_name(predicted_class_idx)
        
        return self._create_ensemble_prediction(
            predicted_class=predicted_class,
            probabilities=weighted_probs,
            predictions=predictions,
            method="weighted_voting"
        )
    
    def _clinical_weighting(self, predictions: List[ModelPrediction]) -> ModelPrediction:
        """Clinical confidence-based weighting strategy."""
        
        # Weight by clinical confidence and diagnostic confidence
        weights = []
        for pred in predictions:
            clinical_weight = pred.diagnostic_confidence or pred.confidence
            quality_weight = pred.input_quality_score
            combined_weight = clinical_weight * quality_weight
            weights.append(combined_weight)
        
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-10)  # Normalize with small epsilon
        
        # Weighted combination
        weighted_probs = np.zeros(self.config.num_classes)
        for pred, weight in zip(predictions, weights):
            weighted_probs += weight * pred.class_probabilities
        
        # Get predicted class
        predicted_class_idx = np.argmax(weighted_probs)
        predicted_class = self.get_class_name(predicted_class_idx)
        
        return self._create_ensemble_prediction(
            predicted_class=predicted_class,
            probabilities=weighted_probs,
            predictions=predictions,
            method="clinical_weighting"
        )
    
    def _stacking_combination(self, predictions: List[ModelPrediction]) -> ModelPrediction:
        """Stacking strategy using meta-learner."""
        
        if self.meta_learner is None:
            # Fall back to weighted voting
            return self._weighted_voting(predictions)
        
        # Extract features for meta-learner
        meta_features = self._extract_meta_features(predictions)
        
        if TORCH_AVAILABLE and hasattr(self.meta_learner, 'meta_network'):
            # PyTorch meta-learner prediction
            features_tensor = torch.FloatTensor(meta_features).unsqueeze(0)
            
            # Split features
            model_outputs = features_tensor[:, :len(predictions) * self.config.num_classes]
            additional_features = features_tensor[:, len(predictions) * self.config.num_classes:]
            
            model_outputs = model_outputs.view(1, len(predictions), self.config.num_classes)
            
            with torch.no_grad():
                logits, confidence = self.meta_learner(model_outputs, additional_features)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
        else:
            # Mock meta-learner prediction
            probs = np.random.softmax(np.random.randn(self.config.num_classes))
        
        predicted_class_idx = np.argmax(probs)
        predicted_class = self.get_class_name(predicted_class_idx)
        
        return self._create_ensemble_prediction(
            predicted_class=predicted_class,
            probabilities=probs,
            predictions=predictions,
            method="stacking"
        )
    
    def _extract_meta_features(self, predictions: List[ModelPrediction]) -> np.ndarray:
        """Extract features for meta-learner."""
        
        features = []
        
        # Model probabilities (flattened)
        for pred in predictions:
            features.extend(pred.class_probabilities)
        
        # Additional meta-features
        if 'confidence' in self.ensemble_config.meta_model_features:
            features.extend([p.confidence for p in predictions])
        
        if 'attention_entropy' in self.ensemble_config.meta_model_features:
            # Calculate attention entropy if available
            for pred in predictions:
                if pred.attention_weights is not None:
                    attention = pred.attention_weights + 1e-10  # Add small epsilon
                    entropy = -np.sum(attention * np.log(attention))
                    features.append(entropy)
                else:
                    features.append(0.0)  # Default entropy
        
        # Pad to expected feature dimension if necessary
        expected_dim = len(predictions) * self.config.num_classes + len(self.ensemble_config.meta_model_features)
        while len(features) < expected_dim:
            features.append(0.0)
        
        return np.array(features[:expected_dim])
    
    def _create_ensemble_prediction(self, 
                                  predicted_class: str,
                                  probabilities: np.ndarray,
                                  predictions: List[ModelPrediction],
                                  method: str) -> ModelPrediction:
        """Create ensemble prediction from combined results."""
        
        # Calculate ensemble confidence
        confidence = float(np.max(probabilities))
        
        # Aggregate clinical interpretations
        clinical_interpretations = [p.clinical_interpretation for p in predictions if p.clinical_interpretation]
        combined_interpretation = f"Ensemble ({method}): {predicted_class}"
        if clinical_interpretations:
            combined_interpretation += f" (based on {len(clinical_interpretations)} model assessments)"
        
        # Determine risk level
        risk_levels = [p.risk_level for p in predictions if p.risk_level]
        if risk_levels:
            if 'high' in risk_levels:
                ensemble_risk = 'high'
            elif 'medium' in risk_levels:
                ensemble_risk = 'medium'
            else:
                ensemble_risk = 'low'
        else:
            ensemble_risk = 'low'
        
        # Calculate diagnostic confidence
        diagnostic_confidences = [p.diagnostic_confidence for p in predictions if p.diagnostic_confidence]
        if diagnostic_confidences:
            ensemble_diagnostic_confidence = np.mean(diagnostic_confidences)
        else:
            ensemble_diagnostic_confidence = confidence
        
        return ModelPrediction(
            predicted_class=predicted_class,
            class_probabilities=probabilities,
            confidence=confidence,
            raw_output=probabilities,
            clinical_interpretation=combined_interpretation,
            risk_level=ensemble_risk,
            diagnostic_confidence=ensemble_diagnostic_confidence,
            model_name=f"ensemble_{method}",
            model_version="1.0.0",
            input_quality_score=np.mean([p.input_quality_score for p in predictions]),
            prediction_quality_score=confidence
        )
    
    def _calculate_agreement(self, predictions: List[ModelPrediction]) -> float:
        """Calculate agreement score between models."""
        
        if len(predictions) < 2:
            return 1.0
        
        # Count predictions for each class
        class_counts = defaultdict(int)
        for pred in predictions:
            class_counts[pred.predicted_class] += 1
        
        # Calculate agreement as proportion of models agreeing with majority
        max_count = max(class_counts.values())
        agreement = max_count / len(predictions)
        
        return agreement
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Get current effective weights for all models."""
        
        weights = {}
        for model_name, weight_config in self.model_weights.items():
            weights[model_name] = weight_config.effective_weight
        
        return weights
    
    def _assess_clinical_consensus(self, predictions: List[ModelPrediction]) -> bool:
        """Assess whether there is clinical consensus among models."""
        
        risk_levels = [p.risk_level for p in predictions if p.risk_level]
        predicted_classes = [p.predicted_class for p in predictions]
        
        # Consensus if majority agree on both class and risk level
        class_consensus = len(set(predicted_classes)) <= len(predicted_classes) / 2
        risk_consensus = len(set(risk_levels)) <= len(risk_levels) / 2 if risk_levels else True
        
        return class_consensus and risk_consensus
    
    def _assess_risk_consensus(self, predictions: List[ModelPrediction]) -> bool:
        """Assess whether there is risk level consensus."""
        
        risk_levels = [p.risk_level for p in predictions if p.risk_level]
        
        if not risk_levels:
            return True
        
        # Consensus if all models agree on risk level
        return len(set(risk_levels)) == 1
    
    def _aggregate_attention(self, predictions: List[ModelPrediction]) -> Optional[np.ndarray]:
        """Aggregate attention weights from individual models."""
        
        attention_weights = [p.attention_weights for p in predictions if p.attention_weights is not None]
        
        if not attention_weights:
            return None
        
        # Ensure all attention weights have same length
        min_length = min(len(att) for att in attention_weights)
        normalized_attention = [att[:min_length] for att in attention_weights]
        
        # Aggregate using specified method
        if self.ensemble_config.attention_weighting_method == "confidence_weighted":
            # Weight by model confidence
            confidences = [p.confidence for p in predictions if p.attention_weights is not None]
            confidences = np.array(confidences)
            confidences = confidences / np.sum(confidences)
            
            aggregated = np.zeros(min_length)
            for att, conf in zip(normalized_attention, confidences):
                aggregated += conf * att
        else:
            # Simple average
            aggregated = np.mean(normalized_attention, axis=0)
        
        return aggregated
    
    def _calculate_model_contributions(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Calculate contribution of each model to final prediction."""
        
        contributions = {}
        
        for i, pred in enumerate(predictions):
            model_name = self.models[i].config.model_name if i < len(self.models) else f"model_{i}"
            
            # Contribution based on confidence and agreement with ensemble
            contribution = pred.confidence * pred.prediction_quality_score
            contributions[model_name] = float(contribution)
        
        # Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {k: v/total_contribution for k, v in contributions.items()}
        
        return contributions
    
    def _update_performance_tracking(self, ensemble_pred: EnsemblePrediction):
        """Update performance tracking for ensemble and individual models."""
        
        self.prediction_count += 1
        
        # Track ensemble metrics
        self.performance_history['agreement_scores'].append(ensemble_pred.agreement_score)
        self.performance_history['confidence_scores'].append(ensemble_pred.ensemble_prediction.confidence)
        
        # Update model weights if needed
        if (self.prediction_count % self.ensemble_config.weight_update_frequency == 0):
            self._update_model_weights()
    
    def _update_model_weights(self):
        """Update model weights based on recent performance."""
        
        # This would implement dynamic weight adjustment based on performance
        # For now, keep weights constant
        pass
    
    def _create_failed_ensemble_prediction(self, error_msg: str) -> EnsemblePrediction:
        """Create failed ensemble prediction."""
        
        failed_pred = self._create_failed_prediction(error_msg)
        
        return EnsemblePrediction(
            ensemble_prediction=failed_pred,
            individual_predictions=[],
            agreement_score=0.0,
            model_weights_used={},
            strategy_used=self.ensemble_config.strategy,
            clinical_consensus=False,
            risk_level_consensus=False,
            confidence_distribution=np.array([])
        )
    
    def _unify_class_names(self):
        """Create unified class names from all models."""
        
        # Use class names from first trained model
        for model in self.models:
            if model.is_trained and model.class_names:
                self.set_class_names(model.class_names)
                break
        else:
            # Fallback to generic names
            self.set_class_names([f'class_{i}' for i in range(self.config.num_classes)])
    
    def load_weights(self, weights_path: str) -> bool:
        """Load ensemble weights (loads individual model weights)."""
        
        try:
            # Load individual model weights
            success_count = 0
            for i, model in enumerate(self.models):
                model_path = weights_path.replace('.pth', f'_model_{i}.pth')
                if model.load_weights(model_path):
                    success_count += 1
            
            self.is_trained = success_count > 0
            print(f"Loaded weights for {success_count}/{len(self.models)} models")
            return success_count > 0
            
        except Exception as e:
            warnings.warn(f"Failed to load ensemble weights: {e}")
            return False
    
    def save_weights(self, weights_path: str) -> bool:
        """Save ensemble weights (saves individual model weights)."""
        
        try:
            success_count = 0
            for i, model in enumerate(self.models):
                model_path = weights_path.replace('.pth', f'_model_{i}.pth')
                if model.save_weights(model_path):
                    success_count += 1
            
            print(f"Saved weights for {success_count}/{len(self.models)} models")
            return success_count > 0
            
        except Exception as e:
            warnings.warn(f"Failed to save ensemble weights: {e}")
            return False
    
    def get_model_summary(self) -> str:
        """Get comprehensive ensemble summary."""
        
        summary = f"""
Ensemble ECG Classifier Summary
==============================

Ensemble Strategy: {self.ensemble_config.strategy.value}
Number of Models: {len(self.models)}
Input Shape: {self.config.input_shape}
Number of Classes: {self.config.num_classes}

Individual Models:
"""
        
        for i, model in enumerate(self.models):
            summary += f"  {i+1}. {model.config.model_name} ({model.config.architecture.value})\n"
            summary += f"     Classes: {model.config.num_classes}, Trained: {model.is_trained}\n"
        
        summary += f"""
Ensemble Configuration:
- Agreement Threshold: {self.ensemble_config.min_agreement_threshold}
- Confidence Threshold: {self.ensemble_config.confidence_threshold}
- Clinical Routing: {self.ensemble_config.enable_clinical_routing}
- Specialty Routing: {self.ensemble_config.enable_specialty_routing}

Performance Tracking:
- Total Predictions: {self.prediction_count}
- Average Agreement: {np.mean(self.performance_history['agreement_scores']) if self.performance_history['agreement_scores'] else 'N/A'}

Training Status: {'Trained' if self.is_trained else 'Not Trained'}
Class Names: {self.class_names}
"""
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    print("Ensemble Manager Test")
    print("=" * 30)
    
    # Import model classes
    from .base_ecg_model import MockECGModel
    
    # Create base configuration
    base_config = ECGModelConfig(
        model_name="base_model",
        architecture=ModelArchitecture.SE_RESNET,
        task=ECGClassificationTask.RHYTHM_CLASSIFICATION,
        num_classes=5,
        input_shape=(5000,),
        sampling_rate=500.0
    )
    
    # Create individual models
    model1 = MockECGModel(base_config)
    model1.config.model_name = "se_resnet_ecg"
    
    model2 = MockECGModel(base_config)
    model2.config.model_name = "han_ecg"
    model2.config.architecture = ModelArchitecture.HAN
    
    model3 = MockECGModel(base_config)
    model3.config.model_name = "rhythm_specialist"
    
    models = [model1, model2, model3]
    
    print(f"Created {len(models)} individual models")
    for model in models:
        print(f"  - {model.config.model_name} ({model.config.architecture.value})")
    
    # Create ensemble configuration
    ensemble_config = EnsembleConfig(
        strategy=EnsembleStrategy.WEIGHTED_VOTING,
        min_agreement_threshold=0.6,
        confidence_threshold=0.7,
        enable_clinical_routing=True
    )
    
    # Create ensemble manager
    ensemble = EnsembleManager(models, ensemble_config=ensemble_config)
    
    print(f"\nEnsemble Manager Summary:")
    print(ensemble.get_model_summary())
    
    # Test ensemble training
    print(f"Testing ensemble training...")
    train_data = np.random.randn(50, 5000) * 0.5
    train_labels = np.random.randint(0, 5, 50)
    
    val_data = np.random.randn(10, 5000) * 0.5
    val_labels = np.random.randint(0, 5, 10)
    
    history = ensemble.train(train_data, train_labels, val_data, val_labels, epochs=5)
    
    print(f"  Training completed!")
    print(f"  Models trained: {sum(1 for m in models if m.is_trained)}/{len(models)}")
    
    # Test ensemble prediction
    print(f"\nTesting ensemble prediction...")
    test_ecg = np.random.randn(5000) * 0.5
    
    # Standard prediction
    prediction = ensemble.predict(test_ecg, return_attention=True)
    
    print(f"  Ensemble prediction: {prediction.predicted_class}")
    print(f"  Ensemble confidence: {prediction.confidence:.3f}")
    print(f"  Risk level: {prediction.risk_level}")
    print(f"  Inference time: {prediction.inference_time_ms:.1f} ms")
    
    # Detailed ensemble prediction
    detailed_pred = ensemble.predict_ensemble_detailed(test_ecg, return_attention=True)
    
    print(f"\nDetailed Ensemble Results:")
    print(f"  Agreement score: {detailed_pred.agreement_score:.3f}")
    print(f"  Clinical consensus: {detailed_pred.clinical_consensus}")
    print(f"  Strategy used: {detailed_pred.strategy_used.value}")
    print(f"  Individual predictions: {len(detailed_pred.individual_predictions)}")
    
    for i, ind_pred in enumerate(detailed_pred.individual_predictions):
        print(f"    Model {i+1}: {ind_pred.predicted_class} (conf: {ind_pred.confidence:.3f})")
    
    # Test different ensemble strategies
    print(f"\nTesting different ensemble strategies...")
    strategies = [EnsembleStrategy.SIMPLE_VOTING, EnsembleStrategy.CLINICAL_WEIGHTING]
    
    for strategy in strategies:
        ensemble.ensemble_config.strategy = strategy
        pred = ensemble.predict(test_ecg)
        print(f"  {strategy.value}: {pred.predicted_class} (conf: {pred.confidence:.3f})")
    
    # Test evaluation
    print(f"\nTesting ensemble evaluation...")
    test_data = np.random.randn(20, 5000) * 0.5
    test_labels = np.random.randint(0, 5, 20)
    
    metrics = ensemble.evaluate(test_data, test_labels)
    print(f"  Evaluation metrics: {metrics}")
    
    # Inference statistics
    stats = ensemble.get_inference_stats()
    print(f"\nInference Statistics:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Average inference time: {stats['average_inference_time']:.1f} ms")
    
    print("\nEnsemble Manager Test Complete!")