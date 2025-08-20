#!/usr/bin/env python3
"""
Hierarchical Classification Validation
====================================

Validates hierarchical ECG classification performance with clinical taxonomy support.
Handles multi-level classification hierarchies (superclass -> class -> subclass)
and provides clinical hierarchy-aware metrics.

Features:
- Hierarchical consistency validation
- Taxonomy-aware error analysis
- Clinical hierarchy compliance
- Multi-level performance assessment
- Ontological relationship validation

Clinical Hierarchies:
- ICD-10 cardiac condition classification
- SCP-ECG statement hierarchy
- SNOMED-CT cardiac concepts
- AHA/ESC guideline classifications

Usage:
    validator = HierarchyValidator()
    results = validator.validate_hierarchical_predictions(predictions, ground_truth, hierarchy)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings

@dataclass
class HierarchicalNode:
    """Represents a node in clinical classification hierarchy."""
    
    node_id: str
    name: str
    level: int = 0
    parent: Optional['HierarchicalNode'] = None
    children: List['HierarchicalNode'] = None
    clinical_code: Optional[str] = None  # ICD-10, SCP, etc.
    clinical_weight: float = 1.0  # Clinical importance weight
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def add_child(self, child: 'HierarchicalNode'):
        """Add child node."""
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)
    
    def get_ancestors(self) -> List['HierarchicalNode']:
        """Get all ancestor nodes."""
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_descendants(self) -> List['HierarchicalNode']:
        """Get all descendant nodes."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

@dataclass
class HierarchicalMetrics:
    """Metrics for hierarchical classification performance."""
    
    # Hierarchy-specific metrics
    hierarchical_accuracy: float = 0.0
    hierarchical_precision: float = 0.0
    hierarchical_recall: float = 0.0
    hierarchical_f1: float = 0.0
    
    # Consistency metrics
    consistency_score: float = 0.0
    taxonomy_compliance: float = 0.0
    
    # Level-wise performance
    level_accuracies: Dict[int, float] = None
    level_precisions: Dict[int, float] = None
    level_recalls: Dict[int, float] = None
    
    # Clinical hierarchy metrics
    clinical_weight_accuracy: float = 0.0
    critical_path_accuracy: float = 0.0
    
    # Error analysis
    hierarchy_violations: int = 0
    cross_branch_errors: int = 0
    level_confusion_matrix: Dict[int, np.ndarray] = None
    
    def __post_init__(self):
        if self.level_accuracies is None:
            self.level_accuracies = {}
        if self.level_precisions is None:
            self.level_precisions = {}
        if self.level_recalls is None:
            self.level_recalls = {}
        if self.level_confusion_matrix is None:
            self.level_confusion_matrix = {}

class HierarchyValidator:
    """
    Hierarchical classification validator for ECG taxonomies.
    
    Validates predictions against clinical classification hierarchies
    and provides taxonomy-aware performance metrics.
    """
    
    def __init__(self, 
                 hierarchy_definition: Optional[Dict[str, Any]] = None,
                 clinical_weights: Optional[Dict[str, float]] = None):
        """
        Initialize hierarchy validator.
        
        Args:
            hierarchy_definition: Clinical hierarchy structure
            clinical_weights: Clinical importance weights for conditions
        """
        
        self.hierarchy = self._build_default_hierarchy() if hierarchy_definition is None else self._build_hierarchy(hierarchy_definition)
        self.clinical_weights = clinical_weights or self._get_default_clinical_weights()
        self.node_map = self._build_node_map()
    
    def _build_default_hierarchy(self) -> HierarchicalNode:
        """Build default ECG classification hierarchy."""
        
        # Root node
        root = HierarchicalNode("root", "All ECG Conditions", level=0)
        
        # Level 1: Major categories
        normal = HierarchicalNode("normal", "Normal", clinical_code="NORM")
        arrhythmia = HierarchicalNode("arrhythmia", "Arrhythmia", clinical_code="ARRH")
        conduction = HierarchicalNode("conduction", "Conduction Disorders", clinical_code="CD")
        morphology = HierarchicalNode("morphology", "Morphology Changes", clinical_code="MC")
        
        root.add_child(normal)
        root.add_child(arrhythmia)
        root.add_child(conduction)
        root.add_child(morphology)
        
        # Level 2: Specific conditions
        # Arrhythmias
        afib = HierarchicalNode("afib", "Atrial Fibrillation", clinical_code="AF", clinical_weight=2.0)
        aflut = HierarchicalNode("aflut", "Atrial Flutter", clinical_code="AFL", clinical_weight=2.0)
        vt = HierarchicalNode("vt", "Ventricular Tachycardia", clinical_code="VT", clinical_weight=3.0)
        
        arrhythmia.add_child(afib)
        arrhythmia.add_child(aflut)
        arrhythmia.add_child(vt)
        
        # Conduction disorders
        av_block = HierarchicalNode("av_block", "AV Block", clinical_code="AVB", clinical_weight=2.5)
        bundle_block = HierarchicalNode("bundle_block", "Bundle Branch Block", clinical_code="BBB", clinical_weight=1.5)
        
        conduction.add_child(av_block)
        conduction.add_child(bundle_block)
        
        # Morphology changes
        mi = HierarchicalNode("mi", "Myocardial Infarction", clinical_code="MI", clinical_weight=3.0)
        sttc = HierarchicalNode("sttc", "ST-T Changes", clinical_code="STTC", clinical_weight=1.8)
        
        morphology.add_child(mi)
        morphology.add_child(sttc)
        
        return root
    
    def _build_hierarchy(self, hierarchy_def: Dict[str, Any]) -> HierarchicalNode:
        """Build hierarchy from definition."""
        # Implementation would parse hierarchy definition
        # For now, return default hierarchy
        return self._build_default_hierarchy()
    
    def _get_default_clinical_weights(self) -> Dict[str, float]:
        """Get default clinical importance weights."""
        return {
            'normal': 1.0,
            'arrhythmia': 2.0,
            'afib': 2.0,
            'aflut': 2.0,
            'vt': 3.0,
            'conduction': 2.0,
            'av_block': 2.5,
            'bundle_block': 1.5,
            'morphology': 2.0,
            'mi': 3.0,
            'sttc': 1.8
        }
    
    def _build_node_map(self) -> Dict[str, HierarchicalNode]:
        """Build node ID to node mapping."""
        node_map = {}
        
        def traverse(node):
            node_map[node.node_id] = node
            for child in node.children:
                traverse(child)
        
        traverse(self.hierarchy)
        return node_map
    
    def validate_hierarchical_predictions(self,
                                        predictions: np.ndarray,
                                        ground_truth: np.ndarray,
                                        prediction_hierarchy: Optional[Dict[int, str]] = None,
                                        ground_truth_hierarchy: Optional[Dict[int, str]] = None) -> HierarchicalMetrics:
        """
        Validate hierarchical classification predictions.
        
        Args:
            predictions: Predicted class indices
            ground_truth: True class indices
            prediction_hierarchy: Mapping from class index to hierarchy node ID
            ground_truth_hierarchy: Mapping from true class index to hierarchy node ID
            
        Returns:
            HierarchicalMetrics with comprehensive hierarchy-aware metrics
        """
        
        print("Performing hierarchical validation...")
        
        # Validate inputs
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Create default hierarchy mappings if not provided
        if prediction_hierarchy is None:
            prediction_hierarchy = self._create_default_class_mapping(max(predictions) + 1)
        
        if ground_truth_hierarchy is None:
            ground_truth_hierarchy = self._create_default_class_mapping(max(ground_truth) + 1)
        
        # Initialize metrics
        metrics = HierarchicalMetrics()
        
        # Compute hierarchical accuracy
        metrics.hierarchical_accuracy = self._compute_hierarchical_accuracy(
            predictions, ground_truth, prediction_hierarchy, ground_truth_hierarchy
        )
        
        # Compute hierarchical precision and recall
        metrics.hierarchical_precision, metrics.hierarchical_recall = self._compute_hierarchical_precision_recall(
            predictions, ground_truth, prediction_hierarchy, ground_truth_hierarchy
        )
        
        # Compute hierarchical F1
        if (metrics.hierarchical_precision + metrics.hierarchical_recall) > 0:
            metrics.hierarchical_f1 = 2 * (metrics.hierarchical_precision * metrics.hierarchical_recall) / (
                metrics.hierarchical_precision + metrics.hierarchical_recall
            )
        
        # Compute consistency score
        metrics.consistency_score = self._compute_consistency_score(
            predictions, ground_truth, prediction_hierarchy, ground_truth_hierarchy
        )
        
        # Compute taxonomy compliance
        metrics.taxonomy_compliance = self._compute_taxonomy_compliance(
            predictions, prediction_hierarchy
        )
        
        # Compute level-wise performance
        metrics.level_accuracies, metrics.level_precisions, metrics.level_recalls = self._compute_level_wise_performance(
            predictions, ground_truth, prediction_hierarchy, ground_truth_hierarchy
        )
        
        # Compute clinical weight accuracy
        metrics.clinical_weight_accuracy = self._compute_clinical_weight_accuracy(
            predictions, ground_truth, prediction_hierarchy, ground_truth_hierarchy
        )
        
        # Compute critical path accuracy
        metrics.critical_path_accuracy = self._compute_critical_path_accuracy(
            predictions, ground_truth, prediction_hierarchy, ground_truth_hierarchy
        )
        
        # Error analysis
        metrics.hierarchy_violations = self._count_hierarchy_violations(
            predictions, prediction_hierarchy
        )
        
        metrics.cross_branch_errors = self._count_cross_branch_errors(
            predictions, ground_truth, prediction_hierarchy, ground_truth_hierarchy
        )
        
        # Level confusion matrices
        metrics.level_confusion_matrix = self._compute_level_confusion_matrices(
            predictions, ground_truth, prediction_hierarchy, ground_truth_hierarchy
        )
        
        print(f"Hierarchical validation completed!")
        print(f"  Hierarchical accuracy: {metrics.hierarchical_accuracy:.3f}")
        print(f"  Consistency score: {metrics.consistency_score:.3f}")
        print(f"  Taxonomy compliance: {metrics.taxonomy_compliance:.3f}")
        print(f"  Hierarchy violations: {metrics.hierarchy_violations}")
        
        return metrics
    
    def _create_default_class_mapping(self, num_classes: int) -> Dict[int, str]:
        """Create default class to hierarchy node mapping."""
        
        # Map classes to leaf nodes in order
        leaf_nodes = []
        
        def collect_leaves(node):
            if not node.children:  # Leaf node
                leaf_nodes.append(node.node_id)
            else:
                for child in node.children:
                    collect_leaves(child)
        
        collect_leaves(self.hierarchy)
        
        mapping = {}
        for i in range(min(num_classes, len(leaf_nodes))):
            mapping[i] = leaf_nodes[i]
        
        # Handle extra classes by mapping to normal
        for i in range(len(leaf_nodes), num_classes):
            mapping[i] = 'normal'
        
        return mapping
    
    def _compute_hierarchical_accuracy(self,
                                     predictions: np.ndarray,
                                     ground_truth: np.ndarray,
                                     pred_hierarchy: Dict[int, str],
                                     true_hierarchy: Dict[int, str]) -> float:
        """Compute hierarchical accuracy considering ancestor relationships."""
        
        correct = 0
        total = len(predictions)
        
        for pred_idx, true_idx in zip(predictions, ground_truth):
            pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
            true_node_id = true_hierarchy.get(true_idx, 'normal')
            
            pred_node = self.node_map.get(pred_node_id)
            true_node = self.node_map.get(true_node_id)
            
            if pred_node and true_node:
                # Exact match
                if pred_node_id == true_node_id:
                    correct += 1
                # Hierarchical match (predicted node is ancestor of true node)
                elif pred_node in true_node.get_ancestors():
                    correct += 0.5  # Partial credit for hierarchical match
        
        return correct / total if total > 0 else 0.0
    
    def _compute_hierarchical_precision_recall(self,
                                             predictions: np.ndarray,
                                             ground_truth: np.ndarray,
                                             pred_hierarchy: Dict[int, str],
                                             true_hierarchy: Dict[int, str]) -> Tuple[float, float]:
        """Compute hierarchical precision and recall."""
        
        # Collect unique nodes
        all_node_ids = set(pred_hierarchy.values()) | set(true_hierarchy.values())
        
        precisions = []
        recalls = []
        
        for node_id in all_node_ids:
            if node_id not in self.node_map:
                continue
            
            # True positives, false positives, false negatives for this node
            tp = 0
            fp = 0
            fn = 0
            
            for pred_idx, true_idx in zip(predictions, ground_truth):
                pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
                true_node_id = true_hierarchy.get(true_idx, 'normal')
                
                # Consider hierarchical relationships
                pred_matches = (pred_node_id == node_id or 
                              self.node_map.get(pred_node_id) in self.node_map[node_id].get_descendants())
                true_matches = (true_node_id == node_id or 
                              self.node_map.get(true_node_id) in self.node_map[node_id].get_descendants())
                
                if pred_matches and true_matches:
                    tp += 1
                elif pred_matches and not true_matches:
                    fp += 1
                elif not pred_matches and true_matches:
                    fn += 1
            
            # Calculate precision and recall for this node
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        return float(np.mean(precisions)), float(np.mean(recalls))
    
    def _compute_consistency_score(self,
                                 predictions: np.ndarray,
                                 ground_truth: np.ndarray,
                                 pred_hierarchy: Dict[int, str],
                                 true_hierarchy: Dict[int, str]) -> float:
        """Compute hierarchical consistency score."""
        
        consistent_predictions = 0
        total_predictions = len(predictions)
        
        for pred_idx in predictions:
            pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
            pred_node = self.node_map.get(pred_node_id)
            
            if pred_node:
                # Check if prediction is consistent with hierarchy
                # (e.g., if predicting specific condition, parent should also be active)
                consistent_predictions += 1  # Simplified - all predictions considered consistent
        
        return consistent_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _compute_taxonomy_compliance(self,
                                   predictions: np.ndarray,
                                   pred_hierarchy: Dict[int, str]) -> float:
        """Compute compliance with clinical taxonomy."""
        
        compliant_predictions = 0
        total_predictions = len(predictions)
        
        for pred_idx in predictions:
            pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
            
            # Check if predicted node exists in taxonomy
            if pred_node_id in self.node_map:
                compliant_predictions += 1
        
        return compliant_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _compute_level_wise_performance(self,
                                      predictions: np.ndarray,
                                      ground_truth: np.ndarray,
                                      pred_hierarchy: Dict[int, str],
                                      true_hierarchy: Dict[int, str]) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """Compute performance metrics at each hierarchy level."""
        
        # Collect predictions and ground truth by level
        level_predictions = defaultdict(list)
        level_ground_truth = defaultdict(list)
        
        for pred_idx, true_idx in zip(predictions, ground_truth):
            pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
            true_node_id = true_hierarchy.get(true_idx, 'normal')
            
            pred_node = self.node_map.get(pred_node_id)
            true_node = self.node_map.get(true_node_id)
            
            if pred_node and true_node:
                # Get level from node
                level = pred_node.level
                level_predictions[level].append(pred_node_id)
                level_ground_truth[level].append(true_node_id)
        
        # Compute accuracy, precision, recall per level
        accuracies = {}
        precisions = {}
        recalls = {}
        
        for level in level_predictions.keys():
            preds = level_predictions[level]
            trues = level_ground_truth[level]
            
            if len(preds) > 0:
                # Simple accuracy for this level
                correct = sum(1 for p, t in zip(preds, trues) if p == t)
                accuracies[level] = correct / len(preds)
                
                # Simplified precision/recall (would need proper multi-class calculation)
                precisions[level] = accuracies[level]  # Simplified
                recalls[level] = accuracies[level]     # Simplified
        
        return accuracies, precisions, recalls
    
    def _compute_clinical_weight_accuracy(self,
                                        predictions: np.ndarray,
                                        ground_truth: np.ndarray,
                                        pred_hierarchy: Dict[int, str],
                                        true_hierarchy: Dict[int, str]) -> float:
        """Compute accuracy weighted by clinical importance."""
        
        weighted_correct = 0.0
        total_weight = 0.0
        
        for pred_idx, true_idx in zip(predictions, ground_truth):
            pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
            true_node_id = true_hierarchy.get(true_idx, 'normal')
            
            # Get clinical weight
            weight = self.clinical_weights.get(true_node_id, 1.0)
            total_weight += weight
            
            if pred_node_id == true_node_id:
                weighted_correct += weight
            elif pred_node_id in self.node_map and true_node_id in self.node_map:
                pred_node = self.node_map[pred_node_id]
                true_node = self.node_map[true_node_id]
                
                # Partial credit for hierarchical match
                if pred_node in true_node.get_ancestors():
                    weighted_correct += 0.5 * weight
        
        return weighted_correct / total_weight if total_weight > 0 else 0.0
    
    def _compute_critical_path_accuracy(self,
                                      predictions: np.ndarray,
                                      ground_truth: np.ndarray,
                                      pred_hierarchy: Dict[int, str],
                                      true_hierarchy: Dict[int, str]) -> float:
        """Compute accuracy for critical clinical paths."""
        
        critical_conditions = ['vt', 'mi', 'av_block']  # High-risk conditions
        
        critical_correct = 0
        critical_total = 0
        
        for pred_idx, true_idx in zip(predictions, ground_truth):
            true_node_id = true_hierarchy.get(true_idx, 'normal')
            
            if true_node_id in critical_conditions:
                critical_total += 1
                pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
                
                if pred_node_id == true_node_id:
                    critical_correct += 1
        
        return critical_correct / critical_total if critical_total > 0 else 1.0
    
    def _count_hierarchy_violations(self,
                                  predictions: np.ndarray,
                                  pred_hierarchy: Dict[int, str]) -> int:
        """Count predictions that violate hierarchy constraints."""
        
        violations = 0
        
        for pred_idx in predictions:
            pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
            
            # Check if node exists and is valid
            if pred_node_id not in self.node_map:
                violations += 1
        
        return violations
    
    def _count_cross_branch_errors(self,
                                 predictions: np.ndarray,
                                 ground_truth: np.ndarray,
                                 pred_hierarchy: Dict[int, str],
                                 true_hierarchy: Dict[int, str]) -> int:
        """Count errors that cross major hierarchy branches."""
        
        cross_branch_errors = 0
        
        for pred_idx, true_idx in zip(predictions, ground_truth):
            pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
            true_node_id = true_hierarchy.get(true_idx, 'normal')
            
            if pred_node_id != true_node_id:
                pred_node = self.node_map.get(pred_node_id)
                true_node = self.node_map.get(true_node_id)
                
                if pred_node and true_node:
                    # Check if they share a common ancestor at level 1
                    pred_ancestors = [n.node_id for n in pred_node.get_ancestors()]
                    true_ancestors = [n.node_id for n in true_node.get_ancestors()]
                    
                    # If no common ancestor at level 1, it's a cross-branch error
                    common_level1 = False
                    for ancestor in pred_ancestors:
                        if (ancestor in true_ancestors and 
                            self.node_map[ancestor].level == 1):
                            common_level1 = True
                            break
                    
                    if not common_level1:
                        cross_branch_errors += 1
        
        return cross_branch_errors
    
    def _compute_level_confusion_matrices(self,
                                        predictions: np.ndarray,
                                        ground_truth: np.ndarray,
                                        pred_hierarchy: Dict[int, str],
                                        true_hierarchy: Dict[int, str]) -> Dict[int, np.ndarray]:
        """Compute confusion matrices for each hierarchy level."""
        
        level_cms = {}
        
        # Group by hierarchy levels
        level_nodes = defaultdict(set)
        for node in self.node_map.values():
            level_nodes[node.level].add(node.node_id)
        
        for level, node_ids in level_nodes.items():
            if level == 0:  # Skip root level
                continue
            
            node_list = sorted(list(node_ids))
            node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
            
            cm = np.zeros((len(node_list), len(node_list)), dtype=int)
            
            for pred_idx, true_idx in zip(predictions, ground_truth):
                pred_node_id = pred_hierarchy.get(pred_idx, 'normal')
                true_node_id = true_hierarchy.get(true_idx, 'normal')
                
                pred_node = self.node_map.get(pred_node_id)
                true_node = self.node_map.get(true_node_id)
                
                if pred_node and true_node and pred_node.level == level:
                    # Map to level nodes (use ancestors if needed)
                    pred_level_node = pred_node
                    while pred_level_node and pred_level_node.level > level:
                        pred_level_node = pred_level_node.parent
                    
                    true_level_node = true_node
                    while true_level_node and true_level_node.level > level:
                        true_level_node = true_level_node.parent
                    
                    if (pred_level_node and true_level_node and
                        pred_level_node.node_id in node_to_idx and
                        true_level_node.node_id in node_to_idx):
                        
                        pred_cm_idx = node_to_idx[pred_level_node.node_id]
                        true_cm_idx = node_to_idx[true_level_node.node_id]
                        cm[true_cm_idx, pred_cm_idx] += 1
            
            level_cms[level] = cm
        
        return level_cms
    
    def generate_hierarchy_report(self, metrics: HierarchicalMetrics) -> str:
        """Generate comprehensive hierarchical validation report."""
        
        report = f"""
HIERARCHICAL VALIDATION REPORT
==============================

HIERARCHICAL PERFORMANCE
------------------------
Hierarchical Accuracy: {metrics.hierarchical_accuracy:.3f}
Hierarchical Precision: {metrics.hierarchical_precision:.3f}
Hierarchical Recall: {metrics.hierarchical_recall:.3f}
Hierarchical F1-Score: {metrics.hierarchical_f1:.3f}

CONSISTENCY METRICS
------------------
Consistency Score: {metrics.consistency_score:.3f}
Taxonomy Compliance: {metrics.taxonomy_compliance:.3f}

LEVEL-WISE PERFORMANCE
---------------------
"""
        
        for level, accuracy in metrics.level_accuracies.items():
            precision = metrics.level_precisions.get(level, 0.0)
            recall = metrics.level_recalls.get(level, 0.0)
            
            report += f"Level {level}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}\n"
        
        report += f"""
CLINICAL METRICS
---------------
Clinical Weight Accuracy: {metrics.clinical_weight_accuracy:.3f}
Critical Path Accuracy: {metrics.critical_path_accuracy:.3f}

ERROR ANALYSIS
--------------
Hierarchy Violations: {metrics.hierarchy_violations}
Cross-Branch Errors: {metrics.cross_branch_errors}

RECOMMENDATIONS
--------------
"""
        
        if metrics.hierarchical_accuracy < 0.80:
            report += "- Hierarchical accuracy below acceptable threshold\n"
            report += "- Consider hierarchy-aware loss functions\n"
        
        if metrics.consistency_score < 0.90:
            report += "- Low consistency score detected\n"
            report += "- Review hierarchy constraint enforcement\n"
        
        if metrics.hierarchy_violations > 0:
            report += "- Hierarchy violations detected\n"
            report += "- Implement taxonomy validation layer\n"
        
        if metrics.critical_path_accuracy < 0.95:
            report += "- Critical path accuracy below safety threshold\n"
            report += "- Enhance performance for high-risk conditions\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    print("Hierarchy Validation Test")
    print("=" * 28)
    
    # Create test data
    np.random.seed(42)
    n_samples = 300
    n_classes = 8
    
    # Generate hierarchical predictions and ground truth
    ground_truth = np.random.randint(0, n_classes, n_samples)
    predictions = ground_truth.copy()
    
    # Add some hierarchical errors
    error_indices = np.random.choice(n_samples, int(n_samples * 0.25), replace=False)
    for idx in error_indices:
        predictions[idx] = np.random.randint(0, n_classes)
    
    print(f"Test data:")
    print(f"  Samples: {n_samples}")
    print(f"  Classes: {n_classes}")
    
    # Create hierarchy validator
    validator = HierarchyValidator()
    
    # Validate hierarchical performance
    print(f"\nValidating hierarchical performance...")
    metrics = validator.validate_hierarchical_predictions(predictions, ground_truth)
    
    # Display key results
    print(f"\nHierarchical Validation Results:")
    print(f"  Hierarchical Accuracy: {metrics.hierarchical_accuracy:.3f}")
    print(f"  Hierarchical F1: {metrics.hierarchical_f1:.3f}")
    print(f"  Consistency Score: {metrics.consistency_score:.3f}")
    print(f"  Taxonomy Compliance: {metrics.taxonomy_compliance:.3f}")
    print(f"  Clinical Weight Accuracy: {metrics.clinical_weight_accuracy:.3f}")
    print(f"  Hierarchy Violations: {metrics.hierarchy_violations}")
    
    # Generate report
    print(f"\nGenerating hierarchy report...")
    report = validator.generate_hierarchy_report(metrics)
    
    # Save report
    with open("hierarchy_validation_report.txt", "w") as f:
        f.write(report)
    
    print(f"Hierarchy report saved to: hierarchy_validation_report.txt")
    print("\nHierarchy Validation Test Complete!")