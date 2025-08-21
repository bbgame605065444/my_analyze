#!/usr/bin/env python3
"""
Base ECG Model Interface
=======================

Abstract base class and interfaces for ECG deep learning models in CoT-RAG Stage 4.
Provides standardized interface for model implementation, training, inference,
and integration with the CoT-RAG reasoning system.

Features:
- Standardized model interface for consistency
- Configuration management
- Model metadata and versioning
- Performance monitoring
- Clinical validation hooks
- Integration with CoT-RAG Stage 3 classifiers

Clinical Standards:
- Supports standard ECG classification tasks
- Compatible with medical device regulations
- Provides model interpretability
- Enables clinical validation workflows
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import warnings

class ECGClassificationTask(Enum):
    """Standard ECG classification tasks."""
    RHYTHM_CLASSIFICATION = "rhythm_classification"  # Normal, AFib, AFL, SVT, VT, etc.
    MORPHOLOGY_CLASSIFICATION = "morphology_classification"  # MI, ischemia, hypertrophy
    BEAT_CLASSIFICATION = "beat_classification"  # Normal, PVC, PAC, etc.
    ARRHYTHMIA_DETECTION = "arrhythmia_detection"  # Binary arrhythmia detection
    MULTI_LABEL_DIAGNOSIS = "multi_label_diagnosis"  # Multiple simultaneous conditions

class ModelArchitecture(Enum):
    """Supported model architectures."""
    SE_RESNET = "se_resnet"
    HAN = "han"  # Hierarchical Attention Network
    CNN_LSTM = "cnn_lstm"
    TRANSFORMER = "transformer"
    FAIRSEQ_TRANSFORMER = "fairseq_transformer"  # Fairseq-signals integration
    ENSEMBLE = "ensemble"

@dataclass
class ECGModelConfig:
    """Configuration for ECG models."""
    model_name: str
    architecture: ModelArchitecture
    task: ECGClassificationTask
    num_classes: int
    input_shape: Tuple[int, ...]  # (length,) for 1D or (length, channels) for multi-lead
    sampling_rate: float = 500.0
    model_version: str = "1.0.0"
    clinical_validation: bool = True
    interpretability_enabled: bool = True
    real_time_inference: bool = True
    
    # Model-specific parameters
    model_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}

@dataclass
class ModelPrediction:
    """Standardized prediction output from ECG models."""
    # Primary prediction
    predicted_class: Union[int, str]
    class_probabilities: np.ndarray
    confidence: float
    
    # Additional information
    raw_output: np.ndarray
    feature_importance: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    
    # Clinical information
    clinical_interpretation: Optional[str] = None
    risk_level: Optional[str] = None  # "low", "medium", "high"
    diagnostic_confidence: Optional[float] = None
    
    # Model metadata
    model_name: str = "unknown"
    model_version: str = "1.0.0"
    inference_time_ms: float = 0.0
    
    # Quality metrics
    input_quality_score: float = 1.0
    prediction_quality_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary format."""
        return {
            'predicted_class': self.predicted_class,
            'class_probabilities': self.class_probabilities.tolist() if isinstance(self.class_probabilities, np.ndarray) else self.class_probabilities,
            'confidence': self.confidence,
            'raw_output': self.raw_output.tolist() if isinstance(self.raw_output, np.ndarray) else self.raw_output,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'attention_weights': self.attention_weights.tolist() if self.attention_weights is not None else None,
            'clinical_interpretation': self.clinical_interpretation,
            'risk_level': self.risk_level,
            'diagnostic_confidence': self.diagnostic_confidence,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'inference_time_ms': self.inference_time_ms,
            'input_quality_score': self.input_quality_score,
            'prediction_quality_score': self.prediction_quality_score
        }

class BaseECGModel(ABC):
    """
    Abstract base class for ECG deep learning models.
    
    Provides standardized interface for ECG classification models
    used in CoT-RAG Stage 4 medical domain integration.
    """
    
    def __init__(self, config: ECGModelConfig):
        """
        Initialize ECG model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.is_trained = False
        self.model = None
        self.class_names = []
        self.training_history = {}
        self.performance_metrics = {}
        self.inference_stats = {
            'total_predictions': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0
        }
        
        # Validate configuration
        self._validate_config()
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build the model architecture.
        
        Returns:
            Built model (implementation-specific)
        """
        pass
    
    @abstractmethod
    def train(self, 
              train_data: np.ndarray,
              train_labels: np.ndarray,
              val_data: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training ECG data
            train_labels: Training labels
            val_data: Validation ECG data (optional)
            val_labels: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Training history and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, 
                ecg_data: np.ndarray,
                return_attention: bool = False,
                return_features: bool = False) -> ModelPrediction:
        """
        Make prediction on ECG data.
        
        Args:
            ecg_data: Input ECG data
            return_attention: Whether to return attention weights
            return_features: Whether to return feature importance
            
        Returns:
            ModelPrediction object with results
        """
        pass
    
    @abstractmethod
    def load_weights(self, weights_path: str) -> bool:
        """
        Load pre-trained model weights.
        
        Args:
            weights_path: Path to model weights file
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def save_weights(self, weights_path: str) -> bool:
        """
        Save model weights.
        
        Args:
            weights_path: Path to save weights
            
        Returns:
            Success status
        """
        pass
    
    def predict_batch(self, 
                     ecg_batch: np.ndarray,
                     batch_size: int = 32) -> List[ModelPrediction]:
        """
        Make predictions on batch of ECG data.
        
        Args:
            ecg_batch: Batch of ECG data
            batch_size: Processing batch size
            
        Returns:
            List of ModelPrediction objects
        """
        predictions = []
        
        for i in range(0, len(ecg_batch), batch_size):
            batch = ecg_batch[i:i + batch_size]
            
            for j, ecg in enumerate(batch):
                try:
                    prediction = self.predict(ecg)
                    predictions.append(prediction)
                except Exception as e:
                    warnings.warn(f"Failed to predict sample {i+j}: {e}")
                    # Create failed prediction
                    failed_prediction = self._create_failed_prediction(str(e))
                    predictions.append(failed_prediction)
        
        return predictions
    
    def evaluate(self, 
                test_data: np.ndarray,
                test_labels: np.ndarray,
                metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test ECG data
            test_labels: True labels
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Make predictions
        predictions = self.predict_batch(test_data)
        predicted_classes = [p.predicted_class for p in predictions]
        
        # Calculate metrics
        evaluation_results = self._calculate_metrics(
            test_labels, predicted_classes, metrics
        )
        
        # Store performance metrics
        self.performance_metrics.update(evaluation_results)
        
        return evaluation_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'config': {
                'model_name': self.config.model_name,
                'architecture': self.config.architecture.value,
                'task': self.config.task.value,
                'num_classes': self.config.num_classes,
                'input_shape': self.config.input_shape,
                'sampling_rate': self.config.sampling_rate,
                'model_version': self.config.model_version
            },
            'training_status': {
                'is_trained': self.is_trained,
                'class_names': self.class_names,
                'training_history': self.training_history
            },
            'performance': self.performance_metrics,
            'inference_stats': self.inference_stats
        }
    
    def set_class_names(self, class_names: List[str]):
        """Set human-readable class names."""
        if len(class_names) != self.config.num_classes:
            raise ValueError(f"Number of class names ({len(class_names)}) must match num_classes ({self.config.num_classes})")
        self.class_names = class_names
    
    def get_class_name(self, class_index: int) -> str:
        """Get class name for given index."""
        if self.class_names and 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        else:
            return f"class_{class_index}"
    
    def validate_input(self, ecg_data: np.ndarray) -> bool:
        """
        Validate input ECG data format.
        
        Args:
            ecg_data: Input ECG data
            
        Returns:
            Validation status
        """
        try:
            # Check data type
            if not isinstance(ecg_data, np.ndarray):
                return False
            
            # Check for invalid values
            if np.any(np.isnan(ecg_data)) or np.any(np.isinf(ecg_data)):
                return False
            
            # Check shape compatibility
            if len(ecg_data.shape) == 1:
                # Single channel ECG
                expected_shape = (self.config.input_shape[0],)
                return ecg_data.shape == expected_shape
            elif len(ecg_data.shape) == 2:
                # Multi-channel ECG
                return ecg_data.shape == self.config.input_shape
            else:
                return False
                
        except Exception:
            return False
    
    def preprocess_input(self, ecg_data: np.ndarray) -> np.ndarray:
        """
        Preprocess input ECG data for model.
        
        Args:
            ecg_data: Raw ECG data
            
        Returns:
            Preprocessed ECG data
        """
        # Basic preprocessing - subclasses can override
        
        # Ensure correct data type
        processed_data = ecg_data.astype(np.float32)
        
        # Normalize to [-1, 1] range if needed
        if np.max(np.abs(processed_data)) > 1.0:
            processed_data = processed_data / np.max(np.abs(processed_data))
        
        return processed_data
    
    def postprocess_output(self, 
                          raw_output: np.ndarray,
                          input_quality: float = 1.0) -> ModelPrediction:
        """
        Postprocess model output into standardized prediction format.
        
        Args:
            raw_output: Raw model output (logits or probabilities)
            input_quality: Quality score of input signal
            
        Returns:
            ModelPrediction object
        """
        import time
        
        # Convert to probabilities if needed
        if np.max(raw_output) > 1.0 or np.min(raw_output) < 0.0:
            # Assume logits, apply softmax
            exp_output = np.exp(raw_output - np.max(raw_output))
            probabilities = exp_output / np.sum(exp_output)
        else:
            probabilities = raw_output
        
        # Get predicted class
        predicted_class_idx = int(np.argmax(probabilities))
        predicted_class = self.get_class_name(predicted_class_idx)
        
        # Calculate confidence
        confidence = float(np.max(probabilities))
        
        # Generate clinical interpretation
        clinical_interpretation = self._generate_clinical_interpretation(
            predicted_class, confidence, probabilities
        )
        
        # Assess risk level
        risk_level = self._assess_risk_level(predicted_class, confidence)
        
        # Calculate prediction quality score
        prediction_quality = self._calculate_prediction_quality(probabilities, input_quality)
        
        return ModelPrediction(
            predicted_class=predicted_class,
            class_probabilities=probabilities,
            confidence=confidence,
            raw_output=raw_output,
            clinical_interpretation=clinical_interpretation,
            risk_level=risk_level,
            diagnostic_confidence=confidence * input_quality,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            input_quality_score=input_quality,
            prediction_quality_score=prediction_quality
        )
    
    def _validate_config(self):
        """Validate model configuration."""
        if self.config.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        
        if len(self.config.input_shape) == 0:
            raise ValueError("input_shape cannot be empty")
        
        if self.config.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
    
    def _calculate_metrics(self, 
                          true_labels: np.ndarray,
                          predicted_labels: np.ndarray,
                          metrics: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        results = {}
        
        try:
            # Convert to numpy arrays
            y_true = np.array(true_labels)
            y_pred = np.array(predicted_labels)
            
            # Basic accuracy
            if 'accuracy' in metrics:
                accuracy = np.mean(y_true == y_pred)
                results['accuracy'] = float(accuracy)
            
            # For multi-class metrics, we need to implement them
            # This is a simplified implementation
            if 'precision' in metrics or 'recall' in metrics or 'f1_score' in metrics:
                # Calculate per-class metrics and average
                unique_classes = np.unique(np.concatenate([y_true, y_pred]))
                
                precisions = []
                recalls = []
                f1_scores = []
                
                for cls in unique_classes:
                    tp = np.sum((y_true == cls) & (y_pred == cls))
                    fp = np.sum((y_true != cls) & (y_pred == cls))
                    fn = np.sum((y_true == cls) & (y_pred != cls))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
                
                if 'precision' in metrics:
                    results['precision'] = float(np.mean(precisions))
                if 'recall' in metrics:
                    results['recall'] = float(np.mean(recalls))
                if 'f1_score' in metrics:
                    results['f1_score'] = float(np.mean(f1_scores))
        
        except Exception as e:
            warnings.warn(f"Failed to calculate metrics: {e}")
            for metric in metrics:
                results[metric] = 0.0
        
        return results
    
    def _generate_clinical_interpretation(self, 
                                        predicted_class: str,
                                        confidence: float,
                                        probabilities: np.ndarray) -> str:
        """Generate clinical interpretation of prediction."""
        
        if not self.config.clinical_validation:
            return f"Predicted: {predicted_class} (confidence: {confidence:.2f})"
        
        # Generate more detailed clinical interpretation
        interpretation_parts = []
        
        # Primary finding
        if confidence > 0.9:
            interpretation_parts.append(f"High confidence prediction: {predicted_class}")
        elif confidence > 0.7:
            interpretation_parts.append(f"Confident prediction: {predicted_class}")
        elif confidence > 0.5:
            interpretation_parts.append(f"Moderate confidence prediction: {predicted_class}")
        else:
            interpretation_parts.append(f"Low confidence prediction: {predicted_class}")
        
        # Secondary findings
        sorted_indices = np.argsort(probabilities)[::-1]
        if len(sorted_indices) > 1 and probabilities[sorted_indices[1]] > 0.2:
            second_class = self.get_class_name(sorted_indices[1])
            second_prob = probabilities[sorted_indices[1]]
            interpretation_parts.append(f"Consider {second_class} (probability: {second_prob:.2f})")
        
        return ". ".join(interpretation_parts)
    
    def _assess_risk_level(self, predicted_class: str, confidence: float) -> str:
        """Assess clinical risk level based on prediction."""
        
        # Default risk assessment - subclasses can override for specific conditions
        high_risk_conditions = ['ventricular_tachycardia', 'ventricular_fibrillation', 
                               'third_degree_block', 'mi', 'stemi']
        medium_risk_conditions = ['atrial_fibrillation', 'supraventricular_tachycardia',
                                 'first_degree_block', 'bundle_branch_block']
        
        predicted_lower = predicted_class.lower()
        
        # Check for high-risk conditions
        for condition in high_risk_conditions:
            if condition in predicted_lower:
                return "high" if confidence > 0.7 else "medium"
        
        # Check for medium-risk conditions
        for condition in medium_risk_conditions:
            if condition in predicted_lower:
                return "medium" if confidence > 0.7 else "low"
        
        # Default to low risk for normal/unknown conditions
        return "low"
    
    def _calculate_prediction_quality(self, 
                                    probabilities: np.ndarray,
                                    input_quality: float) -> float:
        """Calculate prediction quality score."""
        
        # Entropy-based uncertainty measure
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(probabilities))
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Confidence-based quality
        confidence_quality = np.max(probabilities)
        
        # Combine with input quality
        prediction_quality = (confidence_quality * (1.0 - uncertainty) * input_quality)
        
        return float(np.clip(prediction_quality, 0.0, 1.0))
    
    def _create_failed_prediction(self, error_msg: str) -> ModelPrediction:
        """Create prediction object for failed inference."""
        
        # Return default/failed prediction
        default_probabilities = np.ones(self.config.num_classes) / self.config.num_classes
        
        return ModelPrediction(
            predicted_class="unknown",
            class_probabilities=default_probabilities,
            confidence=0.0,
            raw_output=default_probabilities,
            clinical_interpretation=f"Prediction failed: {error_msg}",
            risk_level="unknown",
            diagnostic_confidence=0.0,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            input_quality_score=0.0,
            prediction_quality_score=0.0
        )
    
    def _update_inference_stats(self, inference_time_ms: float):
        """Update inference statistics."""
        self.inference_stats['total_predictions'] += 1
        self.inference_stats['total_inference_time'] += inference_time_ms
        
        if self.inference_stats['total_predictions'] > 0:
            self.inference_stats['average_inference_time'] = (
                self.inference_stats['total_inference_time'] / 
                self.inference_stats['total_predictions']
            )
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return self.inference_stats.copy()
    
    def reset_inference_stats(self):
        """Reset inference statistics."""
        self.inference_stats = {
            'total_predictions': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0
        }


# Example implementation for testing
class MockECGModel(BaseECGModel):
    """Mock ECG model for testing and development."""
    
    def __init__(self, config: ECGModelConfig):
        super().__init__(config)
        # Build a simple mock model
        self.model = self._build_mock_model()
        self.is_trained = True  # Mock as pre-trained
        
        # Set default class names based on task
        if config.task == ECGClassificationTask.RHYTHM_CLASSIFICATION:
            self.set_class_names(['normal', 'atrial_fibrillation', 'atrial_flutter', 
                                'supraventricular_tachycardia', 'ventricular_tachycardia'])
        elif config.task == ECGClassificationTask.ARRHYTHMIA_DETECTION:
            self.set_class_names(['normal', 'arrhythmia'])
    
    def build_model(self):
        """Build mock model."""
        return self._build_mock_model()
    
    def _build_mock_model(self):
        """Create a simple mock model."""
        # Simple linear model for testing
        input_size = np.prod(self.config.input_shape)
        weights = np.random.randn(input_size, self.config.num_classes) * 0.1
        bias = np.zeros(self.config.num_classes)
        return {'weights': weights, 'bias': bias}
    
    def train(self, train_data, train_labels, val_data=None, val_labels=None, **kwargs):
        """Mock training."""
        # Simulate training
        history = {
            'loss': [1.0, 0.8, 0.6, 0.4],
            'accuracy': [0.6, 0.7, 0.8, 0.9]
        }
        self.training_history = history
        self.is_trained = True
        return history
    
    def predict(self, ecg_data, return_attention=False, return_features=False):
        """Mock prediction."""
        import time
        start_time = time.time()
        
        # Validate input
        if not self.validate_input(ecg_data):
            return self._create_failed_prediction("Invalid input format")
        
        # Preprocess
        processed_data = self.preprocess_input(ecg_data)
        
        # Mock inference - simple linear transformation
        flattened = processed_data.flatten()
        raw_output = np.dot(flattened, self.model['weights']) + self.model['bias']
        
        # Add some randomness for realistic behavior
        raw_output += np.random.randn(self.config.num_classes) * 0.1
        
        # Record inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        self._update_inference_stats(inference_time)
        
        # Postprocess
        prediction = self.postprocess_output(raw_output, input_quality=0.8)
        prediction.inference_time_ms = inference_time
        
        # Add mock attention/features if requested
        if return_attention:
            prediction.attention_weights = np.random.rand(len(processed_data))
        if return_features:
            prediction.feature_importance = np.random.rand(len(processed_data))
        
        return prediction
    
    def load_weights(self, weights_path):
        """Mock weight loading."""
        print(f"Mock loading weights from {weights_path}")
        return True
    
    def save_weights(self, weights_path):
        """Mock weight saving."""
        print(f"Mock saving weights to {weights_path}")
        return True


# Example usage and testing
if __name__ == "__main__":
    print("Base ECG Model Interface Test")
    print("=" * 40)
    
    # Create mock configuration
    config = ECGModelConfig(
        model_name="test_ecg_model",
        architecture=ModelArchitecture.SE_RESNET,
        task=ECGClassificationTask.RHYTHM_CLASSIFICATION,
        num_classes=5,
        input_shape=(5000,),  # 10 seconds at 500 Hz
        sampling_rate=500.0
    )
    
    print(f"Configuration: {config.model_name}")
    print(f"  Architecture: {config.architecture.value}")
    print(f"  Task: {config.task.value}")
    print(f"  Classes: {config.num_classes}")
    print(f"  Input shape: {config.input_shape}")
    
    # Create mock model
    model = MockECGModel(config)
    
    print(f"\nModel Info:")
    info = model.get_model_info()
    print(f"  Trained: {info['training_status']['is_trained']}")
    print(f"  Class names: {info['training_status']['class_names']}")
    
    # Test prediction
    print(f"\nTesting prediction...")
    test_ecg = np.random.randn(5000) * 0.5  # Mock ECG signal
    
    prediction = model.predict(test_ecg, return_attention=True, return_features=True)
    
    print(f"  Predicted class: {prediction.predicted_class}")
    print(f"  Confidence: {prediction.confidence:.3f}")
    print(f"  Risk level: {prediction.risk_level}")
    print(f"  Inference time: {prediction.inference_time_ms:.1f} ms")
    print(f"  Clinical interpretation: {prediction.clinical_interpretation}")
    
    # Test batch prediction
    print(f"\nTesting batch prediction...")
    batch_ecg = np.random.randn(10, 5000) * 0.5
    batch_predictions = model.predict_batch(batch_ecg)
    
    print(f"  Batch size: {len(batch_predictions)}")
    print(f"  Predictions: {[p.predicted_class for p in batch_predictions]}")
    
    # Test evaluation
    print(f"\nTesting evaluation...")
    test_data = np.random.randn(20, 5000) * 0.5
    test_labels = np.random.randint(0, 5, 20)
    
    metrics = model.evaluate(test_data, test_labels)
    print(f"  Evaluation metrics: {metrics}")
    
    # Inference statistics
    stats = model.get_inference_stats()
    print(f"\nInference Statistics:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Average inference time: {stats['average_inference_time']:.1f} ms")
    
    print("\nBase ECG Model Interface Test Complete!")