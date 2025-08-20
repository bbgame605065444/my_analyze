#!/usr/bin/env python3
"""
Model Registry and Management System
===================================

Centralized model management system for ECG classification models in CoT-RAG Stage 4.
Provides model registration, versioning, loading, and deployment management
for clinical ECG analysis systems.

Features:
- Model registration and metadata management
- Version control and model lineage tracking
- Performance monitoring and comparison
- A/B testing and deployment management
- Model validation and clinical approval workflows
- Automated model discovery and loading

Clinical Applications:
- Model validation pipelines
- Performance benchmarking
- Clinical approval workflows
- Production deployment management
- Model rollback and safety mechanisms

Usage:
    registry = ModelRegistry()
    registry.register_model(model, metadata)
    model = registry.load_model("se_resnet_v1")
"""

import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import warnings
from datetime import datetime
import pickle

from .base_ecg_model import BaseECGModel, ECGModelConfig, ModelArchitecture, ECGClassificationTask

class ModelStatus(Enum):
    """Model validation and deployment status."""
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    CLINICAL_REVIEW = "clinical_review"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class ModelType(Enum):
    """Type of model for categorization."""
    CLASSIFIER = "classifier"
    ENSEMBLE = "ensemble"
    META_LEARNER = "meta_learner"
    PREPROCESSOR = "preprocessor"
    QUALITY_ASSESSOR = "quality_assessor"

@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    
    # Clinical metrics
    clinical_accuracy: float = 0.0
    expert_agreement: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    
    # Efficiency metrics
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_samples_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)

@dataclass
class ModelInfo:
    """Comprehensive model information and metadata."""
    # Basic information
    model_id: str
    model_name: str
    version: str
    architecture: ModelArchitecture
    task: ECGClassificationTask
    model_type: ModelType = ModelType.CLASSIFIER
    
    # Configuration
    config: ECGModelConfig = None
    
    # File paths
    weights_path: Optional[str] = None
    config_path: Optional[str] = None
    metadata_path: Optional[str] = None
    
    # Status and validation
    status: ModelStatus = ModelStatus.DEVELOPMENT
    
    # Performance
    performance_metrics: PerformanceMetrics = None
    validation_dataset: Optional[str] = None
    
    # Metadata
    description: str = ""
    author: str = ""
    created_date: str = ""
    last_updated: str = ""
    tags: List[str] = None
    
    # Dependencies
    framework_version: str = ""
    dependencies: Dict[str, str] = None
    
    # Clinical information
    clinical_validation_status: str = ""
    regulatory_approval: str = ""
    contraindications: List[str] = None
    clinical_notes: str = ""
    
    # Usage statistics
    total_predictions: int = 0
    deployment_count: int = 0
    last_used: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = {}
        if self.contraindications is None:
            self.contraindications = []
        if self.performance_metrics is None:
            self.performance_metrics = PerformanceMetrics()
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        if not self.last_updated:
            self.last_updated = self.created_date
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model info to dictionary."""
        result = asdict(self)
        # Convert enums to strings
        result['architecture'] = self.architecture.value if self.architecture else None
        result['task'] = self.task.value if self.task else None
        result['model_type'] = self.model_type.value
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create ModelInfo from dictionary."""
        # Convert string enums back to enum objects
        if data.get('architecture'):
            data['architecture'] = ModelArchitecture(data['architecture'])
        if data.get('task'):
            data['task'] = ECGClassificationTask(data['task'])
        if data.get('model_type'):
            data['model_type'] = ModelType(data['model_type'])
        if data.get('status'):
            data['status'] = ModelStatus(data['status'])
        
        # Handle nested objects
        if data.get('performance_metrics'):
            data['performance_metrics'] = PerformanceMetrics(**data['performance_metrics'])
        if data.get('config'):
            data['config'] = ECGModelConfig(**data['config'])
        
        return cls(**data)

class ModelRegistry:
    """
    Centralized registry for ECG classification models.
    
    Manages model registration, versioning, validation, and deployment
    for clinical ECG analysis systems.
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry storage directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        # Registry files
        self.models_file = self.registry_path / "models.json"
        self.metadata_dir = self.registry_path / "metadata"
        self.weights_dir = self.registry_path / "weights"
        
        # Create subdirectories
        self.metadata_dir.mkdir(exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)
        
        # Load existing registry
        self.models: Dict[str, ModelInfo] = self._load_registry()
    
    def register_model(self, 
                      model: BaseECGModel,
                      model_info: Optional[ModelInfo] = None,
                      save_weights: bool = True,
                      performance_metrics: Optional[PerformanceMetrics] = None) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model: ECG model to register
            model_info: Optional model information (auto-generated if None)
            save_weights: Whether to save model weights
            performance_metrics: Optional performance metrics
            
        Returns:
            Model ID assigned to the registered model
        """
        # Generate model info if not provided
        if model_info is None:
            model_info = self._create_model_info_from_model(model, performance_metrics)
        
        # Ensure unique model ID
        model_id = self._generate_unique_id(model_info.model_name, model_info.version)
        model_info.model_id = model_id
        
        # Save model weights if requested
        if save_weights and model.is_trained:
            weights_path = self.weights_dir / f"{model_id}.pth"
            if model.save_weights(str(weights_path)):
                model_info.weights_path = str(weights_path)
            else:
                warnings.warn(f"Failed to save weights for model {model_id}")
        
        # Save model configuration
        config_path = self.metadata_dir / f"{model_id}_config.json"
        self._save_model_config(model.config, config_path)
        model_info.config_path = str(config_path)
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{model_id}_metadata.json"
        self._save_model_metadata(model_info, metadata_path)
        model_info.metadata_path = str(metadata_path)
        
        # Add to registry
        self.models[model_id] = model_info
        
        # Save updated registry
        self._save_registry()
        
        print(f"Registered model: {model_id}")
        return model_id
    
    def load_model(self, 
                   model_id: str,
                   load_weights: bool = True) -> Optional[BaseECGModel]:
        """
        Load a model from the registry.
        
        Args:
            model_id: ID of model to load
            load_weights: Whether to load trained weights
            
        Returns:
            Loaded ECG model or None if not found
        """
        if model_id not in self.models:
            warnings.warn(f"Model {model_id} not found in registry")
            return None
        
        model_info = self.models[model_id]
        
        try:
            # Load model configuration
            if model_info.config_path and os.path.exists(model_info.config_path):
                config = self._load_model_config(model_info.config_path)
            else:
                config = model_info.config
            
            if not config:
                warnings.warn(f"No configuration found for model {model_id}")
                return None
            
            # Create model instance based on architecture
            model = self._create_model_instance(config)
            
            # Load weights if requested and available
            if (load_weights and 
                model_info.weights_path and 
                os.path.exists(model_info.weights_path)):
                
                success = model.load_weights(model_info.weights_path)
                if not success:
                    warnings.warn(f"Failed to load weights for model {model_id}")
            
            # Update usage statistics
            self._update_usage_stats(model_id)
            
            print(f"Loaded model: {model_id}")
            return model
            
        except Exception as e:
            warnings.warn(f"Failed to load model {model_id}: {e}")
            return None
    
    def list_models(self, 
                   status: Optional[ModelStatus] = None,
                   architecture: Optional[ModelArchitecture] = None,
                   task: Optional[ECGClassificationTask] = None) -> List[ModelInfo]:
        """
        List models in registry with optional filtering.
        
        Args:
            status: Filter by model status
            architecture: Filter by model architecture
            task: Filter by classification task
            
        Returns:
            List of matching model information
        """
        models = list(self.models.values())
        
        # Apply filters
        if status:
            models = [m for m in models if m.status == status]
        
        if architecture:
            models = [m for m in models if m.architecture == architecture]
        
        if task:
            models = [m for m in models if m.task == task]
        
        # Sort by last updated (most recent first)
        models.sort(key=lambda m: m.last_updated, reverse=True)
        
        return models
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get detailed information about a model."""
        return self.models.get(model_id)
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """
        Update model status (e.g., for clinical approval workflow).
        
        Args:
            model_id: ID of model to update
            status: New status
            
        Returns:
            Success status
        """
        if model_id not in self.models:
            return False
        
        self.models[model_id].status = status
        self.models[model_id].last_updated = datetime.now().isoformat()
        
        self._save_registry()
        print(f"Updated {model_id} status to {status.value}")
        return True
    
    def update_performance_metrics(self, 
                                  model_id: str, 
                                  metrics: PerformanceMetrics) -> bool:
        """
        Update performance metrics for a model.
        
        Args:
            model_id: ID of model to update
            metrics: New performance metrics
            
        Returns:
            Success status
        """
        if model_id not in self.models:
            return False
        
        self.models[model_id].performance_metrics = metrics
        self.models[model_id].last_updated = datetime.now().isoformat()
        
        self._save_registry()
        print(f"Updated performance metrics for {model_id}")
        return True
    
    def compare_models(self, 
                      model_ids: List[str],
                      metric: str = "accuracy") -> Dict[str, float]:
        """
        Compare performance metrics across models.
        
        Args:
            model_ids: List of model IDs to compare
            metric: Metric to compare (from PerformanceMetrics)
            
        Returns:
            Dictionary mapping model_id to metric value
        """
        comparison = {}
        
        for model_id in model_ids:
            if model_id in self.models:
                model_info = self.models[model_id]
                if hasattr(model_info.performance_metrics, metric):
                    value = getattr(model_info.performance_metrics, metric)
                    comparison[model_id] = value
                else:
                    comparison[model_id] = None
            else:
                comparison[model_id] = None
        
        return comparison
    
    def get_best_model(self, 
                      metric: str = "accuracy",
                      status: Optional[ModelStatus] = None,
                      architecture: Optional[ModelArchitecture] = None) -> Optional[str]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to optimize for
            status: Filter by status
            architecture: Filter by architecture
            
        Returns:
            Model ID of best performing model
        """
        models = self.list_models(status=status, architecture=architecture)
        
        if not models:
            return None
        
        # Find model with best metric
        best_model = None
        best_value = -float('inf')
        
        for model in models:
            if hasattr(model.performance_metrics, metric):
                value = getattr(model.performance_metrics, metric)
                if value is not None and value > best_value:
                    best_value = value
                    best_model = model.model_id
        
        return best_model
    
    def deploy_model(self, model_id: str) -> bool:
        """
        Mark model as deployed and update deployment count.
        
        Args:
            model_id: ID of model to deploy
            
        Returns:
            Success status
        """
        if model_id not in self.models:
            return False
        
        model_info = self.models[model_id]
        
        # Check if model is approved for deployment
        if model_info.status not in [ModelStatus.APPROVED, ModelStatus.DEPLOYED]:
            warnings.warn(f"Model {model_id} not approved for deployment (status: {model_info.status.value})")
            return False
        
        # Update status and counters
        model_info.status = ModelStatus.DEPLOYED
        model_info.deployment_count += 1
        model_info.last_updated = datetime.now().isoformat()
        
        self._save_registry()
        print(f"Deployed model: {model_id}")
        return True
    
    def archive_model(self, model_id: str) -> bool:
        """
        Archive a model (change status to archived).
        
        Args:
            model_id: ID of model to archive
            
        Returns:
            Success status
        """
        return self.update_model_status(model_id, ModelStatus.ARCHIVED)
    
    def delete_model(self, model_id: str, confirm: bool = False) -> bool:
        """
        Delete a model from registry (irreversible).
        
        Args:
            model_id: ID of model to delete
            confirm: Confirmation flag to prevent accidental deletion
            
        Returns:
            Success status
        """
        if not confirm:
            warnings.warn("Model deletion requires confirmation flag")
            return False
        
        if model_id not in self.models:
            return False
        
        model_info = self.models[model_id]
        
        # Delete associated files
        files_to_delete = [
            model_info.weights_path,
            model_info.config_path,
            model_info.metadata_path
        ]
        
        for file_path in files_to_delete:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    warnings.warn(f"Failed to delete {file_path}: {e}")
        
        # Remove from registry
        del self.models[model_id]
        self._save_registry()
        
        print(f"Deleted model: {model_id}")
        return True
    
    def export_model_info(self, model_id: str, export_path: str) -> bool:
        """
        Export model information to file.
        
        Args:
            model_id: ID of model to export
            export_path: Path to export file
            
        Returns:
            Success status
        """
        if model_id not in self.models:
            return False
        
        try:
            model_info = self.models[model_id]
            with open(export_path, 'w') as f:
                json.dump(model_info.to_dict(), f, indent=2)
            
            print(f"Exported model info for {model_id} to {export_path}")
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to export model info: {e}")
            return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        
        if not self.models:
            return {'total_models': 0}
        
        stats = {
            'total_models': len(self.models),
            'by_status': {},
            'by_architecture': {},
            'by_task': {},
            'by_type': {},
            'total_predictions': sum(m.total_predictions for m in self.models.values()),
            'total_deployments': sum(m.deployment_count for m in self.models.values())
        }
        
        # Count by categories
        for model in self.models.values():
            # By status
            status = model.status.value
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # By architecture
            if model.architecture:
                arch = model.architecture.value
                stats['by_architecture'][arch] = stats['by_architecture'].get(arch, 0) + 1
            
            # By task
            if model.task:
                task = model.task.value
                stats['by_task'][task] = stats['by_task'].get(task, 0) + 1
            
            # By type
            model_type = model.model_type.value
            stats['by_type'][model_type] = stats['by_type'].get(model_type, 0) + 1
        
        return stats
    
    def _load_registry(self) -> Dict[str, ModelInfo]:
        """Load registry from file."""
        
        if not self.models_file.exists():
            return {}
        
        try:
            with open(self.models_file, 'r') as f:
                data = json.load(f)
            
            models = {}
            for model_id, model_data in data.items():
                models[model_id] = ModelInfo.from_dict(model_data)
            
            return models
            
        except Exception as e:
            warnings.warn(f"Failed to load registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save registry to file."""
        
        try:
            data = {}
            for model_id, model_info in self.models.items():
                data[model_id] = model_info.to_dict()
            
            with open(self.models_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            warnings.warn(f"Failed to save registry: {e}")
    
    def _create_model_info_from_model(self, 
                                     model: BaseECGModel,
                                     performance_metrics: Optional[PerformanceMetrics]) -> ModelInfo:
        """Create ModelInfo from model instance."""
        
        # Extract basic information
        model_name = model.config.model_name or "unnamed_model"
        version = model.config.model_version or "1.0.0"
        
        # Determine model type
        if model.config.architecture == ModelArchitecture.ENSEMBLE:
            model_type = ModelType.ENSEMBLE
        elif "quality" in model_name.lower():
            model_type = ModelType.QUALITY_ASSESSOR
        else:
            model_type = ModelType.CLASSIFIER
        
        # Create model info
        model_info = ModelInfo(
            model_id="",  # Will be set during registration
            model_name=model_name,
            version=version,
            architecture=model.config.architecture,
            task=model.config.task,
            model_type=model_type,
            config=model.config,
            performance_metrics=performance_metrics or PerformanceMetrics(),
            status=ModelStatus.DEVELOPMENT,
            description=f"Auto-registered {model.config.architecture.value} model",
            tags=[model.config.architecture.value, model.config.task.value]
        )
        
        return model_info
    
    def _generate_unique_id(self, model_name: str, version: str) -> str:
        """Generate unique model ID."""
        
        base_id = f"{model_name}_{version}"
        
        # Ensure uniqueness
        counter = 0
        unique_id = base_id
        
        while unique_id in self.models:
            counter += 1
            unique_id = f"{base_id}_{counter}"
        
        return unique_id
    
    def _save_model_config(self, config: ECGModelConfig, config_path: Path):
        """Save model configuration to file."""
        
        try:
            config_dict = {
                'model_name': config.model_name,
                'architecture': config.architecture.value,
                'task': config.task.value,
                'num_classes': config.num_classes,
                'input_shape': config.input_shape,
                'sampling_rate': config.sampling_rate,
                'model_version': config.model_version,
                'clinical_validation': config.clinical_validation,
                'interpretability_enabled': config.interpretability_enabled,
                'real_time_inference': config.real_time_inference,
                'model_params': config.model_params
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
        except Exception as e:
            warnings.warn(f"Failed to save model config: {e}")
    
    def _load_model_config(self, config_path: str) -> Optional[ECGModelConfig]:
        """Load model configuration from file."""
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Convert enum strings back to enums
            data['architecture'] = ModelArchitecture(data['architecture'])
            data['task'] = ECGClassificationTask(data['task'])
            
            return ECGModelConfig(**data)
            
        except Exception as e:
            warnings.warn(f"Failed to load model config: {e}")
            return None
    
    def _save_model_metadata(self, model_info: ModelInfo, metadata_path: Path):
        """Save model metadata to file."""
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(model_info.to_dict(), f, indent=2)
                
        except Exception as e:
            warnings.warn(f"Failed to save model metadata: {e}")
    
    def _create_model_instance(self, config: ECGModelConfig) -> BaseECGModel:
        """Create model instance based on configuration."""
        
        # Import model classes (would normally be done at module level)
        from .base_ecg_model import MockECGModel
        from .se_resnet_classifier import SEResNetClassifier
        from .han_classifier import HANClassifier
        from .ensemble_manager import EnsembleManager
        
        # Create model based on architecture
        if config.architecture == ModelArchitecture.SE_RESNET:
            return SEResNetClassifier(config)
        elif config.architecture == ModelArchitecture.HAN:
            return HANClassifier(config)
        elif config.architecture == ModelArchitecture.ENSEMBLE:
            # For ensemble, we would need to load constituent models
            # For now, create a mock ensemble
            base_models = [MockECGModel(config) for _ in range(2)]
            return EnsembleManager(base_models, config)
        else:
            # Default to mock model
            return MockECGModel(config)
    
    def _update_usage_stats(self, model_id: str):
        """Update model usage statistics."""
        
        if model_id in self.models:
            self.models[model_id].total_predictions += 1
            self.models[model_id].last_used = datetime.now().isoformat()
            
            # Save periodically (every 10 uses)
            if self.models[model_id].total_predictions % 10 == 0:
                self._save_registry()


# Example usage and testing
if __name__ == "__main__":
    print("Model Registry Test")
    print("=" * 25)
    
    # Import model classes
    from .base_ecg_model import MockECGModel, ECGModelConfig
    
    # Create test registry
    registry = ModelRegistry("test_registry")
    
    # Create test models
    config1 = ECGModelConfig(
        model_name="test_se_resnet",
        architecture=ModelArchitecture.SE_RESNET,
        task=ECGClassificationTask.RHYTHM_CLASSIFICATION,
        num_classes=5,
        input_shape=(5000,),
        model_version="1.0.0"
    )
    
    config2 = ECGModelConfig(
        model_name="test_han",
        architecture=ModelArchitecture.HAN,
        task=ECGClassificationTask.ARRHYTHMIA_DETECTION,
        num_classes=2,
        input_shape=(5000,),
        model_version="1.0.0"
    )
    
    model1 = MockECGModel(config1)
    model2 = MockECGModel(config2)
    
    print(f"Created test models:")
    print(f"  - {model1.config.model_name}")
    print(f"  - {model2.config.model_name}")
    
    # Register models
    print(f"\nRegistering models...")
    
    # Create performance metrics
    metrics1 = PerformanceMetrics(
        accuracy=0.92,
        precision=0.89,
        recall=0.94,
        f1_score=0.91,
        auc_roc=0.95,
        inference_time_ms=12.5
    )
    
    metrics2 = PerformanceMetrics(
        accuracy=0.88,
        precision=0.85,
        recall=0.91,
        f1_score=0.88,
        auc_roc=0.93,
        inference_time_ms=18.3
    )
    
    model_id1 = registry.register_model(model1, performance_metrics=metrics1)
    model_id2 = registry.register_model(model2, performance_metrics=metrics2)
    
    print(f"  Registered: {model_id1}")
    print(f"  Registered: {model_id2}")
    
    # List models
    print(f"\nListing all models:")
    models = registry.list_models()
    for model in models:
        print(f"  - {model.model_id}: {model.model_name} v{model.version} "
              f"({model.status.value})")
    
    # Get model info
    print(f"\nModel info for {model_id1}:")
    model_info = registry.get_model_info(model_id1)
    if model_info:
        print(f"  Architecture: {model_info.architecture.value}")
        print(f"  Task: {model_info.task.value}")
        print(f"  Status: {model_info.status.value}")
        print(f"  Accuracy: {model_info.performance_metrics.accuracy:.3f}")
    
    # Update model status
    print(f"\nUpdating model status...")
    registry.update_model_status(model_id1, ModelStatus.VALIDATION)
    registry.update_model_status(model_id2, ModelStatus.APPROVED)
    
    # Compare models
    print(f"\nComparing models by accuracy:")
    comparison = registry.compare_models([model_id1, model_id2], "accuracy")
    for mid, acc in comparison.items():
        print(f"  {mid}: {acc:.3f}")
    
    # Get best model
    best_model_id = registry.get_best_model("accuracy", status=ModelStatus.APPROVED)
    print(f"Best approved model: {best_model_id}")
    
    # Load model
    print(f"\nLoading model {model_id1}...")
    loaded_model = registry.load_model(model_id1)
    if loaded_model:
        print(f"  Successfully loaded: {loaded_model.config.model_name}")
        print(f"  Is trained: {loaded_model.is_trained}")
    
    # Registry statistics
    print(f"\nRegistry Statistics:")
    stats = registry.get_registry_stats()
    print(f"  Total models: {stats['total_models']}")
    print(f"  By status: {stats['by_status']}")
    print(f"  By architecture: {stats['by_architecture']}")
    print(f"  Total predictions: {stats['total_predictions']}")
    
    # Deploy model
    print(f"\nDeploying model {model_id2}...")
    success = registry.deploy_model(model_id2)
    print(f"  Deployment success: {success}")
    
    # Export model info
    export_path = "test_model_export.json"
    print(f"\nExporting model info to {export_path}...")
    success = registry.export_model_info(model_id1, export_path)
    print(f"  Export success: {success}")
    
    print("\nModel Registry Test Complete!")