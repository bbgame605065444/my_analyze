"""
Hierarchical Evaluation Metrics for CoT-RAG
Specialized evaluation metrics for medical hierarchical classification and reasoning chains.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json

class MetricType(Enum):
    """Types of evaluation metrics."""
    HIERARCHICAL_ACCURACY = "hierarchical_accuracy"
    HIERARCHICAL_PRECISION = "hierarchical_precision" 
    HIERARCHICAL_RECALL = "hierarchical_recall"
    HIERARCHICAL_F1 = "hierarchical_f1"
    PATH_ACCURACY = "path_accuracy"
    REASONING_COHERENCE = "reasoning_coherence"
    CLINICAL_VALIDITY = "clinical_validity"

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_type: MetricType
    value: float
    details: Dict[str, Any]
    confidence_interval: Optional[Tuple[float, float]] = None

class HierarchicalEvaluator:
    """
    Evaluator for hierarchical medical classification and reasoning.
    
    Implements specialized metrics that account for the hierarchical structure
    of medical diagnoses and the reasoning chains in CoT-RAG.
    """
    
    def __init__(self, hierarchy: Dict[str, List[str]] = None):
        """
        Initialize hierarchical evaluator.
        
        Args:
            hierarchy: Dictionary mapping parent codes to child codes
        """
        self.hierarchy = hierarchy or {}
        self.parent_map = self._build_parent_map()
    
    def _build_parent_map(self) -> Dict[str, str]:
        """Build reverse mapping from child to parent codes."""
        parent_map = {}
        for parent, children in self.hierarchy.items():
            for child in children:
                parent_map[child] = parent
        return parent_map
    
    def _augment_labels(self, labels: List[str]) -> Set[str]:
        """
        Augment label set with all ancestor labels.
        
        Args:
            labels: List of diagnostic labels
            
        Returns:
            Set of labels including all ancestors
        """
        augmented = set(labels)
        
        for label in labels:
            current = label
            while current in self.parent_map:
                parent = self.parent_map[current]
                augmented.add(parent)
                current = parent
        
        return augmented
    
    def hierarchical_precision(self, y_true: List[List[str]], 
                             y_pred: List[List[str]]) -> EvaluationResult:
        """
        Calculate hierarchical precision.
        
        Args:
            y_true: List of true label sets for each sample
            y_pred: List of predicted label sets for each sample
            
        Returns:
            EvaluationResult with hierarchical precision
        """
        precisions = []
        
        for true_labels, pred_labels in zip(y_true, y_pred):
            true_aug = self._augment_labels(true_labels)
            pred_aug = self._augment_labels(pred_labels)
            
            if len(pred_aug) == 0:
                precision = 0.0
            else:
                intersection = len(true_aug.intersection(pred_aug))
                precision = intersection / len(pred_aug)
            
            precisions.append(precision)
        
        mean_precision = np.mean(precisions)
        
        return EvaluationResult(
            metric_type=MetricType.HIERARCHICAL_PRECISION,
            value=mean_precision,
            details={
                'per_sample_precisions': precisions,
                'std': np.std(precisions),
                'samples_evaluated': len(precisions)
            }
        )
    
    def hierarchical_recall(self, y_true: List[List[str]], 
                           y_pred: List[List[str]]) -> EvaluationResult:
        """
        Calculate hierarchical recall.
        
        Args:
            y_true: List of true label sets for each sample
            y_pred: List of predicted label sets for each sample
            
        Returns:
            EvaluationResult with hierarchical recall
        """
        recalls = []
        
        for true_labels, pred_labels in zip(y_true, y_pred):
            true_aug = self._augment_labels(true_labels)
            pred_aug = self._augment_labels(pred_labels)
            
            if len(true_aug) == 0:
                recall = 1.0 if len(pred_aug) == 0 else 0.0
            else:
                intersection = len(true_aug.intersection(pred_aug))
                recall = intersection / len(true_aug)
            
            recalls.append(recall)
        
        mean_recall = np.mean(recalls)
        
        return EvaluationResult(
            metric_type=MetricType.HIERARCHICAL_RECALL,
            value=mean_recall,
            details={
                'per_sample_recalls': recalls,
                'std': np.std(recalls),
                'samples_evaluated': len(recalls)
            }
        )
    
    def hierarchical_f1(self, y_true: List[List[str]], 
                       y_pred: List[List[str]]) -> EvaluationResult:
        """
        Calculate hierarchical F1-score.
        
        Args:
            y_true: List of true label sets for each sample
            y_pred: List of predicted label sets for each sample
            
        Returns:
            EvaluationResult with hierarchical F1
        """
        precision_result = self.hierarchical_precision(y_true, y_pred)
        recall_result = self.hierarchical_recall(y_true, y_pred)
        
        precision = precision_result.value
        recall = recall_result.value
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return EvaluationResult(
            metric_type=MetricType.HIERARCHICAL_F1,
            value=f1,
            details={
                'precision': precision,
                'recall': recall,
                'precision_details': precision_result.details,
                'recall_details': recall_result.details
            }
        )
    
    def path_accuracy(self, true_paths: List[List[str]], 
                     pred_paths: List[List[str]]) -> EvaluationResult:
        """
        Calculate path accuracy for reasoning chains.
        
        Args:
            true_paths: List of true reasoning paths (node sequences)
            pred_paths: List of predicted reasoning paths
            
        Returns:
            EvaluationResult with path accuracy
        """
        exact_matches = 0
        partial_matches = []
        
        for true_path, pred_path in zip(true_paths, pred_paths):
            if true_path == pred_path:
                exact_matches += 1
                partial_matches.append(1.0)
            else:
                # Calculate partial match as longest common subsequence ratio
                lcs_length = self._longest_common_subsequence_length(true_path, pred_path)
                partial_match = lcs_length / max(len(true_path), 1)
                partial_matches.append(partial_match)
        
        exact_accuracy = exact_matches / len(true_paths)
        partial_accuracy = np.mean(partial_matches)
        
        return EvaluationResult(
            metric_type=MetricType.PATH_ACCURACY,
            value=exact_accuracy,
            details={
                'exact_accuracy': exact_accuracy,
                'partial_accuracy': partial_accuracy,
                'exact_matches': exact_matches,
                'total_samples': len(true_paths),
                'per_sample_partial_accuracy': partial_matches
            }
        )
    
    def _longest_common_subsequence_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length between two sequences."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def reasoning_coherence(self, reasoning_chains: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate coherence of reasoning chains.
        
        Args:
            reasoning_chains: List of reasoning chain dictionaries with evidence, decisions, etc.
            
        Returns:
            EvaluationResult with coherence score
        """
        coherence_scores = []
        
        for chain in reasoning_chains:
            score = self._evaluate_chain_coherence(chain)
            coherence_scores.append(score)
        
        mean_coherence = np.mean(coherence_scores)
        
        return EvaluationResult(
            metric_type=MetricType.REASONING_COHERENCE,
            value=mean_coherence,
            details={
                'per_chain_scores': coherence_scores,
                'std': np.std(coherence_scores),
                'chains_evaluated': len(coherence_scores)
            }
        )
    
    def _evaluate_chain_coherence(self, chain: Dict[str, Any]) -> float:
        """
        Evaluate coherence of a single reasoning chain.
        
        Criteria:
        - Evidence relevance to questions
        - Decision consistency with evidence
        - Logical flow between steps
        - Confidence consistency
        """
        evidence_steps = chain.get('evidence_chain', [])
        
        if not evidence_steps:
            return 0.0
        
        coherence_factors = []
        
        # Check evidence-question relevance
        for step in evidence_steps:
            question = step.get('question', '').lower()
            evidence = step.get('evidence', '').lower()
            
            # Simple keyword overlap check
            question_words = set(question.split())
            evidence_words = set(evidence.split())
            
            if len(question_words) > 0:
                overlap = len(question_words.intersection(evidence_words))
                relevance = overlap / len(question_words)
                coherence_factors.append(relevance)
        
        # Check confidence consistency
        confidences = [step.get('confidence', 0.0) for step in evidence_steps]
        if confidences:
            conf_variance = np.var(confidences)
            consistency = 1.0 / (1.0 + conf_variance)  # Lower variance = higher consistency
            coherence_factors.append(consistency)
        
        # Check decision logic (simplified)
        decisions = [step.get('decision', '') for step in evidence_steps]
        non_empty_decisions = [d for d in decisions if d.strip()]
        if non_empty_decisions:
            decision_consistency = 1.0  # Simplified - could implement more sophisticated logic
            coherence_factors.append(decision_consistency)
        
        return np.mean(coherence_factors) if coherence_factors else 0.0

class ClinicalValidationEvaluator:
    """
    Evaluator for clinical validity of diagnostic reasoning.
    """
    
    def __init__(self):
        """Initialize clinical validation evaluator."""
        self.clinical_rules = self._load_clinical_rules()
    
    def _load_clinical_rules(self) -> Dict[str, Any]:
        """Load clinical validation rules."""
        return {
            # Temporal consistency rules
            'temporal_rules': {
                'acute_before_chronic': ['AMI', 'IMI'],  # Acute MI should precede chronic changes
                'progression_patterns': ['1AVB', '2AVB', '3AVB']  # AV block progression
            },
            
            # Anatomical consistency rules
            'anatomical_rules': {
                'anterior_leads': ['AMI'],  # Anterior MI affects specific leads
                'inferior_leads': ['IMI'],  # Inferior MI affects different leads
                'reciprocal_changes': True  # ST elevation should have reciprocal changes
            },
            
            # Clinical severity rules
            'severity_rules': {
                'life_threatening': ['3AVB', 'VT', 'VF'],  # Require immediate intervention
                'urgent': ['2AVB', 'STEMI'],  # Require prompt intervention
                'routine': ['1AVB', 'LVH']  # Can be managed electively
            }
        }
    
    def validate_clinical_logic(self, diagnosis_chain: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Validate clinical logic of diagnostic reasoning.
        
        Args:
            diagnosis_chain: Chain of diagnostic steps with evidence
            
        Returns:
            EvaluationResult with clinical validity score
        """
        validation_scores = []
        violations = []
        
        # Check temporal consistency
        temporal_score, temporal_violations = self._check_temporal_consistency(diagnosis_chain)
        validation_scores.append(temporal_score)
        violations.extend(temporal_violations)
        
        # Check anatomical consistency
        anatomical_score, anatomical_violations = self._check_anatomical_consistency(diagnosis_chain)
        validation_scores.append(anatomical_score)
        violations.extend(anatomical_violations)
        
        # Check severity appropriateness
        severity_score, severity_violations = self._check_severity_appropriateness(diagnosis_chain)
        validation_scores.append(severity_score)
        violations.extend(severity_violations)
        
        overall_validity = np.mean(validation_scores)
        
        return EvaluationResult(
            metric_type=MetricType.CLINICAL_VALIDITY,
            value=overall_validity,
            details={
                'temporal_score': temporal_score,
                'anatomical_score': anatomical_score,
                'severity_score': severity_score,
                'violations': violations,
                'total_checks': len(validation_scores)
            }
        )
    
    def _check_temporal_consistency(self, chain: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        """Check temporal consistency of diagnostic reasoning."""
        # Simplified temporal check
        violations = []
        
        # Check for illogical temporal sequences
        diagnoses = [step.get('diagnosis', '') for step in chain]
        
        # Example: Chronic changes shouldn't appear before acute in same episode
        if 'chronic_MI' in diagnoses and 'acute_MI' in diagnoses:
            chronic_idx = diagnoses.index('chronic_MI')
            acute_idx = diagnoses.index('acute_MI')
            if chronic_idx < acute_idx:
                violations.append("Chronic MI diagnosed before acute MI")
        
        score = 1.0 - (len(violations) * 0.2)  # Penalize violations
        return max(0.0, score), violations
    
    def _check_anatomical_consistency(self, chain: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        """Check anatomical consistency of findings."""
        violations = []
        
        # Check for anatomically inconsistent findings
        findings = [step.get('evidence', '').lower() for step in chain]
        
        # Example: Anterior MI shouldn't have inferior lead changes as primary finding
        if any('anterior' in finding for finding in findings):
            if any('inferior lead' in finding for finding in findings):
                violations.append("Anterior MI with primary inferior lead changes")
        
        score = 1.0 - (len(violations) * 0.15)
        return max(0.0, score), violations
    
    def _check_severity_appropriateness(self, chain: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        """Check appropriateness of urgency/severity assessment."""
        violations = []
        
        # Check if high-severity conditions are appropriately flagged
        for step in chain:
            diagnosis = step.get('diagnosis', '')
            confidence = step.get('confidence', 0.0)
            
            if diagnosis in self.clinical_rules['severity_rules']['life_threatening']:
                if confidence < 0.8:  # Life-threatening should have high confidence
                    violations.append(f"Low confidence ({confidence}) for life-threatening condition: {diagnosis}")
        
        score = 1.0 - (len(violations) * 0.25)  # Heavy penalty for severity misassessment
        return max(0.0, score), violations

class ComprehensiveEvaluationSuite:
    """
    Comprehensive evaluation suite for CoT-RAG systems.
    """
    
    def __init__(self, hierarchy: Dict[str, List[str]] = None):
        """Initialize comprehensive evaluation suite."""
        self.hierarchical_evaluator = HierarchicalEvaluator(hierarchy)
        self.clinical_evaluator = ClinicalValidationEvaluator()
    
    def evaluate_complete_system(self, predictions: Dict[str, Any], 
                                ground_truth: Dict[str, Any]) -> Dict[str, EvaluationResult]:
        """
        Evaluate complete CoT-RAG system performance.
        
        Args:
            predictions: System predictions with diagnoses and reasoning chains
            ground_truth: Ground truth data
            
        Returns:
            Dictionary of evaluation results
        """
        results = {}
        
        # Extract data for evaluation
        y_true = ground_truth.get('diagnoses', [])
        y_pred = predictions.get('diagnoses', [])
        true_paths = ground_truth.get('reasoning_paths', [])
        pred_paths = predictions.get('reasoning_paths', [])
        reasoning_chains = predictions.get('reasoning_chains', [])
        
        # Hierarchical classification metrics
        if y_true and y_pred:
            results['hierarchical_precision'] = self.hierarchical_evaluator.hierarchical_precision(y_true, y_pred)
            results['hierarchical_recall'] = self.hierarchical_evaluator.hierarchical_recall(y_true, y_pred)
            results['hierarchical_f1'] = self.hierarchical_evaluator.hierarchical_f1(y_true, y_pred)
        
        # Reasoning path evaluation
        if true_paths and pred_paths:
            results['path_accuracy'] = self.hierarchical_evaluator.path_accuracy(true_paths, pred_paths)
        
        # Reasoning coherence
        if reasoning_chains:
            results['reasoning_coherence'] = self.hierarchical_evaluator.reasoning_coherence(reasoning_chains)
        
        # Clinical validation
        if reasoning_chains:
            clinical_results = []
            for chain in reasoning_chains:
                clinical_result = self.clinical_evaluator.validate_clinical_logic([chain])
                clinical_results.append(clinical_result.value)
            
            # Average clinical validity across all chains
            avg_clinical_validity = np.mean(clinical_results) if clinical_results else 0.0
            results['clinical_validity'] = EvaluationResult(
                metric_type=MetricType.CLINICAL_VALIDITY,
                value=avg_clinical_validity,
                details={'per_chain_validity': clinical_results}
            )
        
        return results
    
    def generate_evaluation_report(self, results: Dict[str, EvaluationResult]) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            Formatted evaluation report
        """
        report_lines = [
            "=" * 70,
            "CoT-RAG System Evaluation Report",
            "=" * 70,
            ""
        ]
        
        # Summary metrics
        report_lines.append("SUMMARY METRICS:")
        report_lines.append("-" * 30)
        
        for metric_name, result in results.items():
            report_lines.append(f"{metric_name.replace('_', ' ').title()}: {result.value:.4f}")
        
        report_lines.append("")
        
        # Detailed analysis
        report_lines.append("DETAILED ANALYSIS:")
        report_lines.append("-" * 30)
        
        for metric_name, result in results.items():
            report_lines.append(f"\n{metric_name.replace('_', ' ').title()}:")
            report_lines.append(f"  Value: {result.value:.4f}")
            
            # Add relevant details
            if 'std' in result.details:
                report_lines.append(f"  Standard Deviation: {result.details['std']:.4f}")
            
            if 'violations' in result.details:
                violations = result.details['violations']
                if violations:
                    report_lines.append(f"  Clinical Violations ({len(violations)}):")
                    for violation in violations[:5]:  # Show first 5
                        report_lines.append(f"    - {violation}")
        
        return "\n".join(report_lines)

# Convenience functions
def evaluate_hierarchical_predictions(y_true: List[List[str]], y_pred: List[List[str]], 
                                    hierarchy: Dict[str, List[str]] = None) -> Dict[str, float]:
    """
    Quick evaluation of hierarchical predictions.
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    evaluator = HierarchicalEvaluator(hierarchy)
    
    precision = evaluator.hierarchical_precision(y_true, y_pred)
    recall = evaluator.hierarchical_recall(y_true, y_pred)
    f1 = evaluator.hierarchical_f1(y_true, y_pred)
    
    return {
        'precision': precision.value,
        'recall': recall.value,
        'f1': f1.value
    }