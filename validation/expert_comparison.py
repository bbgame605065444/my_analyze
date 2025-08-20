#!/usr/bin/env python3
"""
Expert Comparison and Agreement Analysis
=======================================

Compares model predictions against expert cardiologist annotations
and evaluates inter-expert agreement for clinical validation.

Features:
- Multi-expert annotation support
- Inter-expert agreement analysis (Kappa, ICC, Gwet's AC1)
- Expert-model agreement assessment
- Clinical expertise modeling
- Confidence-weighted analysis
- Experience-based weighting

Clinical Applications:
- Expert validation studies
- Clinical trial support
- Model benchmarking
- Regulatory submission
- Quality assurance

Usage:
    comparison = ExpertComparison()
    agreement = comparison.compare_with_experts(predictions, expert_annotations)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings
from scipy import stats
import itertools

@dataclass
class ExpertAnnotation:
    """Expert annotation with metadata."""
    
    expert_id: str
    annotations: np.ndarray
    confidence_scores: Optional[np.ndarray] = None
    annotation_time: Optional[np.ndarray] = None  # Time taken per annotation
    expertise_level: str = "experienced"  # "novice", "experienced", "expert", "specialist"
    years_experience: Optional[int] = None
    specialization: Optional[str] = None  # "cardiology", "electrophysiology", etc.
    institution: Optional[str] = None
    
@dataclass
class ExpertAgreement:
    """Expert agreement analysis results."""
    
    # Inter-expert agreement
    fleiss_kappa: float = 0.0
    light_kappa: float = 0.0
    gwets_ac1: float = 0.0
    intraclass_correlation: float = 0.0
    
    # Pairwise agreements
    pairwise_agreements: Dict[Tuple[str, str], float] = None
    pairwise_kappa: Dict[Tuple[str, str], float] = None
    
    # Model-expert agreements
    model_expert_agreements: Dict[str, float] = None
    model_expert_kappa: Dict[str, str] = None
    
    # Confidence analysis
    agreement_by_confidence: Dict[str, float] = None  # high, medium, low confidence
    confidence_accuracy_correlation: float = 0.0
    
    # Experience analysis
    agreement_by_experience: Dict[str, float] = None
    experience_accuracy_correlation: float = 0.0
    
    # Clinical analysis
    agreement_by_condition: Dict[str, float] = None
    critical_condition_agreement: float = 0.0
    
    # Statistical significance
    statistical_tests: Dict[str, Dict[str, float]] = None
    
    # Quality metrics
    annotation_consistency: Dict[str, float] = None
    expert_reliability: Dict[str, float] = None
    
    def __post_init__(self):
        if self.pairwise_agreements is None:
            self.pairwise_agreements = {}
        if self.pairwise_kappa is None:
            self.pairwise_kappa = {}
        if self.model_expert_agreements is None:
            self.model_expert_agreements = {}
        if self.model_expert_kappa is None:
            self.model_expert_kappa = {}
        if self.agreement_by_confidence is None:
            self.agreement_by_confidence = {}
        if self.agreement_by_experience is None:
            self.agreement_by_experience = {}
        if self.agreement_by_condition is None:
            self.agreement_by_condition = {}
        if self.statistical_tests is None:
            self.statistical_tests = {}
        if self.annotation_consistency is None:
            self.annotation_consistency = {}
        if self.expert_reliability is None:
            self.expert_reliability = {}

class ExpertComparison:
    """
    Expert comparison and agreement analysis for ECG classification.
    
    Provides comprehensive analysis of agreement between model predictions
    and expert annotations, as well as inter-expert agreement assessment.
    """
    
    def __init__(self,
                 class_names: Optional[List[str]] = None,
                 expertise_weights: Optional[Dict[str, float]] = None,
                 confidence_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize expert comparison analyzer.
        
        Args:
            class_names: Names of classification classes
            expertise_weights: Weights for different expertise levels
            confidence_thresholds: Thresholds for confidence categorization
        """
        
        self.class_names = class_names or []
        
        # Default expertise weights
        self.expertise_weights = expertise_weights or {
            'novice': 1.0,
            'experienced': 1.5,
            'expert': 2.0,
            'specialist': 2.5
        }
        
        # Default confidence thresholds
        self.confidence_thresholds = confidence_thresholds or {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.0
        }
        
        # Critical conditions requiring high agreement
        self.critical_conditions = ['MI', 'VT', 'VF', 'AV_BLOCK', 'STEMI']
    
    def compare_with_experts(self,
                           model_predictions: np.ndarray,
                           expert_annotations: List[ExpertAnnotation],
                           class_names: Optional[List[str]] = None) -> ExpertAgreement:
        """
        Compare model predictions with expert annotations.
        
        Args:
            model_predictions: Model prediction class indices
            expert_annotations: List of expert annotations
            class_names: Names of classes
            
        Returns:
            ExpertAgreement with comprehensive agreement analysis
        """
        
        print("Performing expert comparison analysis...")
        
        # Validate inputs
        if len(expert_annotations) == 0:
            raise ValueError("At least one expert annotation required")
        
        # Validate all annotations have same length
        n_samples = len(model_predictions)
        for annotation in expert_annotations:
            if len(annotation.annotations) != n_samples:
                raise ValueError(f"Expert {annotation.expert_id} annotations length mismatch")
        
        if class_names:
            self.class_names = class_names
        
        # Initialize agreement results
        agreement = ExpertAgreement()
        
        # Inter-expert agreement analysis
        if len(expert_annotations) > 1:
            agreement.fleiss_kappa = self._compute_fleiss_kappa(expert_annotations)
            agreement.light_kappa = self._compute_light_kappa(expert_annotations)
            agreement.gwets_ac1 = self._compute_gwets_ac1(expert_annotations)
            agreement.intraclass_correlation = self._compute_icc(expert_annotations)
            
            # Pairwise expert agreements
            agreement.pairwise_agreements = self._compute_pairwise_agreements(expert_annotations)
            agreement.pairwise_kappa = self._compute_pairwise_kappa(expert_annotations)
        
        # Model-expert agreements
        agreement.model_expert_agreements = self._compute_model_expert_agreements(
            model_predictions, expert_annotations
        )
        agreement.model_expert_kappa = self._compute_model_expert_kappa(
            model_predictions, expert_annotations
        )
        
        # Confidence analysis
        agreement.agreement_by_confidence = self._analyze_agreement_by_confidence(
            model_predictions, expert_annotations
        )
        agreement.confidence_accuracy_correlation = self._compute_confidence_accuracy_correlation(
            model_predictions, expert_annotations
        )
        
        # Experience analysis
        agreement.agreement_by_experience = self._analyze_agreement_by_experience(
            model_predictions, expert_annotations
        )
        agreement.experience_accuracy_correlation = self._compute_experience_accuracy_correlation(
            model_predictions, expert_annotations
        )
        
        # Clinical condition analysis
        agreement.agreement_by_condition = self._analyze_agreement_by_condition(
            model_predictions, expert_annotations
        )
        agreement.critical_condition_agreement = self._compute_critical_condition_agreement(
            model_predictions, expert_annotations
        )
        
        # Statistical significance tests
        agreement.statistical_tests = self._compute_statistical_tests(
            model_predictions, expert_annotations
        )
        
        # Quality metrics
        agreement.annotation_consistency = self._compute_annotation_consistency(expert_annotations)
        agreement.expert_reliability = self._compute_expert_reliability(expert_annotations)
        
        print(f"Expert comparison completed!")
        print(f"  Number of experts: {len(expert_annotations)}")
        if len(expert_annotations) > 1:
            print(f"  Fleiss' Kappa (inter-expert): {agreement.fleiss_kappa:.3f}")
        print(f"  Mean model-expert agreement: {np.mean(list(agreement.model_expert_agreements.values())):.3f}")
        print(f"  Critical condition agreement: {agreement.critical_condition_agreement:.3f}")
        
        return agreement
    
    def _compute_fleiss_kappa(self, annotations: List[ExpertAnnotation]) -> float:
        """Compute Fleiss' Kappa for inter-expert agreement."""
        
        try:
            n_subjects = len(annotations[0].annotations)
            n_experts = len(annotations)
            
            # Get unique categories
            all_annotations = np.concatenate([ann.annotations for ann in annotations])
            categories = np.unique(all_annotations)
            n_categories = len(categories)
            
            # Create annotation matrix (subjects x experts)
            annotation_matrix = np.zeros((n_subjects, n_experts))
            for i, annotation in enumerate(annotations):
                annotation_matrix[:, i] = annotation.annotations
            
            # Count matrix (subjects x categories)
            count_matrix = np.zeros((n_subjects, n_categories))
            for i in range(n_subjects):
                for j, category in enumerate(categories):
                    count_matrix[i, j] = np.sum(annotation_matrix[i, :] == category)
            
            # Calculate agreement
            P_i = np.sum(count_matrix * (count_matrix - 1), axis=1) / (n_experts * (n_experts - 1))
            P_bar = np.mean(P_i)
            
            # Calculate chance agreement
            p_j = np.sum(count_matrix, axis=0) / (n_subjects * n_experts)
            P_e = np.sum(p_j ** 2)
            
            # Fleiss' Kappa
            if P_e < 1.0:
                kappa = (P_bar - P_e) / (1 - P_e)
            else:
                kappa = 0.0
            
            return float(kappa)
        
        except Exception as e:
            warnings.warn(f"Fleiss' Kappa computation failed: {e}")
            return 0.0
    
    def _compute_light_kappa(self, annotations: List[ExpertAnnotation]) -> float:
        """Compute Light's Kappa (average pairwise kappa)."""
        
        if len(annotations) < 2:
            return 0.0
        
        try:
            kappas = []
            for i, j in itertools.combinations(range(len(annotations)), 2):
                kappa = self._cohen_kappa(annotations[i].annotations, annotations[j].annotations)
                kappas.append(kappa)
            
            return float(np.mean(kappas))
        
        except Exception as e:
            warnings.warn(f"Light's Kappa computation failed: {e}")
            return 0.0
    
    def _cohen_kappa(self, annotations1: np.ndarray, annotations2: np.ndarray) -> float:
        """Compute Cohen's Kappa between two annotators."""
        
        try:
            # Observed agreement
            po = np.mean(annotations1 == annotations2)
            
            # Expected agreement
            categories = np.unique(np.concatenate([annotations1, annotations2]))
            pe = 0.0
            
            for category in categories:
                p1 = np.mean(annotations1 == category)
                p2 = np.mean(annotations2 == category)
                pe += p1 * p2
            
            # Cohen's Kappa
            if pe < 1.0:
                kappa = (po - pe) / (1 - pe)
            else:
                kappa = 0.0
            
            return float(kappa)
        
        except Exception as e:
            warnings.warn(f"Cohen's Kappa computation failed: {e}")
            return 0.0
    
    def _compute_gwets_ac1(self, annotations: List[ExpertAnnotation]) -> float:
        """Compute Gwet's AC1 coefficient."""
        
        try:
            n_subjects = len(annotations[0].annotations)
            n_experts = len(annotations)
            
            # Create annotation matrix
            annotation_matrix = np.zeros((n_subjects, n_experts))
            for i, annotation in enumerate(annotations):
                annotation_matrix[:, i] = annotation.annotations
            
            # Observed agreement
            agreements = []
            for i in range(n_subjects):
                subject_annotations = annotation_matrix[i, :]
                # Count pairwise agreements
                total_pairs = n_experts * (n_experts - 1) / 2
                agreements_count = 0
                
                for j in range(n_experts):
                    for k in range(j + 1, n_experts):
                        if subject_annotations[j] == subject_annotations[k]:
                            agreements_count += 1
                
                agreements.append(agreements_count / total_pairs if total_pairs > 0 else 0)
            
            pa = np.mean(agreements)
            
            # Gwet's AC1 uses uniform chance agreement
            categories = np.unique(np.concatenate([ann.annotations for ann in annotations]))
            pe = 1.0 / len(categories)  # Uniform distribution
            
            # AC1 coefficient
            if pe < 1.0:
                ac1 = (pa - pe) / (1 - pe)
            else:
                ac1 = 0.0
            
            return float(ac1)
        
        except Exception as e:
            warnings.warn(f"Gwet's AC1 computation failed: {e}")
            return 0.0
    
    def _compute_icc(self, annotations: List[ExpertAnnotation]) -> float:
        """Compute Intraclass Correlation Coefficient."""
        
        try:
            # Convert to numeric data for ICC
            n_subjects = len(annotations[0].annotations)
            n_experts = len(annotations)
            
            data_matrix = np.zeros((n_subjects, n_experts))
            for i, annotation in enumerate(annotations):
                data_matrix[:, i] = annotation.annotations.astype(float)
            
            # ICC(2,1) - Two-way random effects, single measurement
            # Mean squares calculation
            subject_means = np.mean(data_matrix, axis=1)
            expert_means = np.mean(data_matrix, axis=0)
            grand_mean = np.mean(data_matrix)
            
            # Sum of squares
            ssb = n_experts * np.sum((subject_means - grand_mean) ** 2)  # Between subjects
            ssw = np.sum((data_matrix - subject_means[:, np.newaxis]) ** 2)  # Within subjects
            sst = np.sum((data_matrix - grand_mean) ** 2)  # Total
            
            # Mean squares
            msb = ssb / (n_subjects - 1)
            msw = ssw / (n_subjects * (n_experts - 1))
            
            # ICC calculation
            if (msb + (n_experts - 1) * msw) > 0:
                icc = (msb - msw) / (msb + (n_experts - 1) * msw)
            else:
                icc = 0.0
            
            return float(max(0.0, icc))  # ICC should be non-negative
        
        except Exception as e:
            warnings.warn(f"ICC computation failed: {e}")
            return 0.0
    
    def _compute_pairwise_agreements(self, annotations: List[ExpertAnnotation]) -> Dict[Tuple[str, str], float]:
        """Compute pairwise agreements between all expert pairs."""
        
        agreements = {}
        
        for i, j in itertools.combinations(range(len(annotations)), 2):
            expert1 = annotations[i]
            expert2 = annotations[j]
            
            agreement = np.mean(expert1.annotations == expert2.annotations)
            agreements[(expert1.expert_id, expert2.expert_id)] = float(agreement)
        
        return agreements
    
    def _compute_pairwise_kappa(self, annotations: List[ExpertAnnotation]) -> Dict[Tuple[str, str], float]:
        """Compute pairwise kappa between all expert pairs."""
        
        kappas = {}
        
        for i, j in itertools.combinations(range(len(annotations)), 2):
            expert1 = annotations[i]
            expert2 = annotations[j]
            
            kappa = self._cohen_kappa(expert1.annotations, expert2.annotations)
            kappas[(expert1.expert_id, expert2.expert_id)] = float(kappa)
        
        return kappas
    
    def _compute_model_expert_agreements(self,
                                       model_predictions: np.ndarray,
                                       annotations: List[ExpertAnnotation]) -> Dict[str, float]:
        """Compute agreement between model and each expert."""
        
        agreements = {}
        
        for annotation in annotations:
            agreement = np.mean(model_predictions == annotation.annotations)
            agreements[annotation.expert_id] = float(agreement)
        
        return agreements
    
    def _compute_model_expert_kappa(self,
                                  model_predictions: np.ndarray,
                                  annotations: List[ExpertAnnotation]) -> Dict[str, float]:
        """Compute kappa between model and each expert."""
        
        kappas = {}
        
        for annotation in annotations:
            kappa = self._cohen_kappa(model_predictions, annotation.annotations)
            kappas[annotation.expert_id] = float(kappa)
        
        return kappas
    
    def _analyze_agreement_by_confidence(self,
                                       model_predictions: np.ndarray,
                                       annotations: List[ExpertAnnotation]) -> Dict[str, float]:
        """Analyze agreement stratified by expert confidence."""
        
        confidence_agreements = {'high': [], 'medium': [], 'low': []}
        
        for annotation in annotations:
            if annotation.confidence_scores is not None:
                # Categorize by confidence
                high_conf = annotation.confidence_scores >= self.confidence_thresholds['high']
                medium_conf = ((annotation.confidence_scores >= self.confidence_thresholds['medium']) & 
                             (annotation.confidence_scores < self.confidence_thresholds['high']))
                low_conf = annotation.confidence_scores < self.confidence_thresholds['medium']
                
                # Calculate agreements for each confidence category
                if np.any(high_conf):
                    high_agreement = np.mean(model_predictions[high_conf] == annotation.annotations[high_conf])
                    confidence_agreements['high'].append(high_agreement)
                
                if np.any(medium_conf):
                    medium_agreement = np.mean(model_predictions[medium_conf] == annotation.annotations[medium_conf])
                    confidence_agreements['medium'].append(medium_agreement)
                
                if np.any(low_conf):
                    low_agreement = np.mean(model_predictions[low_conf] == annotation.annotations[low_conf])
                    confidence_agreements['low'].append(low_agreement)
        
        # Average across experts
        result = {}
        for conf_level, agreements in confidence_agreements.items():
            if agreements:
                result[conf_level] = float(np.mean(agreements))
            else:
                result[conf_level] = 0.0
        
        return result
    
    def _compute_confidence_accuracy_correlation(self,
                                               model_predictions: np.ndarray,
                                               annotations: List[ExpertAnnotation]) -> float:
        """Compute correlation between expert confidence and accuracy."""
        
        all_confidences = []
        all_accuracies = []
        
        for annotation in annotations:
            if annotation.confidence_scores is not None:
                accuracies = (model_predictions == annotation.annotations).astype(float)
                all_confidences.extend(annotation.confidence_scores)
                all_accuracies.extend(accuracies)
        
        if len(all_confidences) > 0:
            try:
                correlation, _ = stats.pearsonr(all_confidences, all_accuracies)
                return float(correlation)
            except:
                pass
        
        return 0.0
    
    def _analyze_agreement_by_experience(self,
                                       model_predictions: np.ndarray,
                                       annotations: List[ExpertAnnotation]) -> Dict[str, float]:
        """Analyze agreement by expert experience level."""
        
        experience_agreements = defaultdict(list)
        
        for annotation in annotations:
            agreement = np.mean(model_predictions == annotation.annotations)
            experience_agreements[annotation.expertise_level].append(agreement)
        
        # Average by experience level
        result = {}
        for exp_level, agreements in experience_agreements.items():
            result[exp_level] = float(np.mean(agreements))
        
        return result
    
    def _compute_experience_accuracy_correlation(self,
                                               model_predictions: np.ndarray,
                                               annotations: List[ExpertAnnotation]) -> float:
        """Compute correlation between years of experience and accuracy."""
        
        experience_years = []
        accuracies = []
        
        for annotation in annotations:
            if annotation.years_experience is not None:
                accuracy = np.mean(model_predictions == annotation.annotations)
                experience_years.append(annotation.years_experience)
                accuracies.append(accuracy)
        
        if len(experience_years) > 1:
            try:
                correlation, _ = stats.pearsonr(experience_years, accuracies)
                return float(correlation)
            except:
                pass
        
        return 0.0
    
    def _analyze_agreement_by_condition(self,
                                      model_predictions: np.ndarray,
                                      annotations: List[ExpertAnnotation]) -> Dict[str, float]:
        """Analyze agreement by medical condition."""
        
        if not self.class_names:
            return {}
        
        condition_agreements = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            # Find samples with this condition in ground truth
            condition_indices = []
            for annotation in annotations:
                condition_indices.extend(np.where(annotation.annotations == class_idx)[0])
            
            condition_indices = list(set(condition_indices))  # Remove duplicates
            
            if condition_indices:
                # Calculate agreement for this condition
                agreements = []
                for annotation in annotations:
                    if len(condition_indices) > 0:
                        agreement = np.mean(
                            model_predictions[condition_indices] == annotation.annotations[condition_indices]
                        )
                        agreements.append(agreement)
                
                if agreements:
                    condition_agreements[class_name] = float(np.mean(agreements))
        
        return condition_agreements
    
    def _compute_critical_condition_agreement(self,
                                            model_predictions: np.ndarray,
                                            annotations: List[ExpertAnnotation]) -> float:
        """Compute agreement for critical medical conditions."""
        
        if not self.class_names:
            return 0.0
        
        critical_agreements = []
        
        for class_idx, class_name in enumerate(self.class_names):
            if any(critical in class_name.upper() for critical in self.critical_conditions):
                # Find critical condition samples
                critical_indices = []
                for annotation in annotations:
                    critical_indices.extend(np.where(annotation.annotations == class_idx)[0])
                
                critical_indices = list(set(critical_indices))
                
                if critical_indices:
                    for annotation in annotations:
                        agreement = np.mean(
                            model_predictions[critical_indices] == annotation.annotations[critical_indices]
                        )
                        critical_agreements.append(agreement)
        
        return float(np.mean(critical_agreements)) if critical_agreements else 0.0
    
    def _compute_statistical_tests(self,
                                 model_predictions: np.ndarray,
                                 annotations: List[ExpertAnnotation]) -> Dict[str, Dict[str, float]]:
        """Compute statistical significance tests."""
        
        tests = {}
        
        # McNemar's test for model vs experts
        for annotation in annotations:
            try:
                # Create contingency table
                both_correct = np.sum((model_predictions == annotation.annotations))
                model_correct_expert_wrong = np.sum((model_predictions != annotation.annotations))
                
                # Simplified McNemar's test (would need proper implementation)
                tests[f'mcnemar_vs_{annotation.expert_id}'] = {
                    'statistic': 0.0,  # Placeholder
                    'p_value': 1.0     # Placeholder
                }
            
            except Exception as e:
                warnings.warn(f"Statistical test failed for {annotation.expert_id}: {e}")
        
        return tests
    
    def _compute_annotation_consistency(self, annotations: List[ExpertAnnotation]) -> Dict[str, float]:
        """Compute consistency of each expert's annotations."""
        
        consistency = {}
        
        for annotation in annotations:
            # Simple consistency measure: entropy of annotation distribution
            unique, counts = np.unique(annotation.annotations, return_counts=True)
            probabilities = counts / len(annotation.annotations)
            
            # Calculate entropy (lower = more consistent)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(len(unique))
            
            # Consistency score (higher = more consistent)
            if max_entropy > 0:
                consistency_score = 1 - (entropy / max_entropy)
            else:
                consistency_score = 1.0
            
            consistency[annotation.expert_id] = float(consistency_score)
        
        return consistency
    
    def _compute_expert_reliability(self, annotations: List[ExpertAnnotation]) -> Dict[str, float]:
        """Compute reliability scores for experts."""
        
        reliability = {}
        
        for annotation in annotations:
            # Combine multiple factors for reliability score
            reliability_score = 1.0
            
            # Experience factor
            experience_weight = self.expertise_weights.get(annotation.expertise_level, 1.0)
            reliability_score *= experience_weight / 2.5  # Normalize
            
            # Confidence factor (if available)
            if annotation.confidence_scores is not None:
                avg_confidence = np.mean(annotation.confidence_scores)
                reliability_score *= avg_confidence
            
            # Years of experience factor
            if annotation.years_experience is not None:
                exp_factor = min(1.0, annotation.years_experience / 20.0)  # Cap at 20 years
                reliability_score *= (0.5 + 0.5 * exp_factor)  # 0.5-1.0 range
            
            reliability[annotation.expert_id] = float(reliability_score)
        
        return reliability
    
    def generate_expert_comparison_report(self, agreement: ExpertAgreement) -> str:
        """Generate comprehensive expert comparison report."""
        
        report = f"""
EXPERT COMPARISON REPORT
========================

INTER-EXPERT AGREEMENT
----------------------
Fleiss' Kappa: {agreement.fleiss_kappa:.3f}
Light's Kappa: {agreement.light_kappa:.3f}
Gwet's AC1: {agreement.gwets_ac1:.3f}
Intraclass Correlation: {agreement.intraclass_correlation:.3f}

MODEL-EXPERT AGREEMENT
---------------------
"""
        
        for expert_id, agreement_score in agreement.model_expert_agreements.items():
            kappa = agreement.model_expert_kappa.get(expert_id, 0.0)
            report += f"{expert_id}: Agreement={agreement_score:.3f}, Kappa={kappa:.3f}\n"
        
        report += f"""
CONFIDENCE ANALYSIS
------------------
"""
        for conf_level, agreement_score in agreement.agreement_by_confidence.items():
            report += f"{conf_level.capitalize()} Confidence: {agreement_score:.3f}\n"
        
        report += f"Confidence-Accuracy Correlation: {agreement.confidence_accuracy_correlation:.3f}\n"
        
        report += f"""
EXPERIENCE ANALYSIS
------------------
"""
        for exp_level, agreement_score in agreement.agreement_by_experience.items():
            report += f"{exp_level.capitalize()}: {agreement_score:.3f}\n"
        
        report += f"Experience-Accuracy Correlation: {agreement.experience_accuracy_correlation:.3f}\n"
        
        report += f"""
CLINICAL CONDITIONS
------------------
"""
        for condition, agreement_score in agreement.agreement_by_condition.items():
            report += f"{condition}: {agreement_score:.3f}\n"
        
        report += f"Critical Condition Agreement: {agreement.critical_condition_agreement:.3f}\n"
        
        report += f"""
EXPERT RELIABILITY
-----------------
"""
        for expert_id, reliability_score in agreement.expert_reliability.items():
            consistency = agreement.annotation_consistency.get(expert_id, 0.0)
            report += f"{expert_id}: Reliability={reliability_score:.3f}, Consistency={consistency:.3f}\n"
        
        report += f"""
RECOMMENDATIONS
--------------
"""
        
        if agreement.fleiss_kappa < 0.6:
            report += "- Low inter-expert agreement detected\n"
            report += "- Consider expert training or guideline clarification\n"
        
        if agreement.critical_condition_agreement < 0.90:
            report += "- Critical condition agreement below threshold\n"
            report += "- Focus on high-risk condition classification\n"
        
        if agreement.confidence_accuracy_correlation < 0.3:
            report += "- Weak confidence-accuracy correlation\n"
            report += "- Expert confidence calibration may be needed\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    print("Expert Comparison Test")
    print("=" * 25)
    
    # Create test data
    np.random.seed(42)
    n_samples = 200
    n_classes = 5
    class_names = ['Normal', 'MI', 'AF', 'BBB', 'STTC']
    
    # Generate ground truth
    ground_truth = np.random.randint(0, n_classes, n_samples)
    
    # Generate model predictions (80% accuracy)
    model_predictions = ground_truth.copy()
    error_indices = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
    for idx in error_indices:
        model_predictions[idx] = np.random.randint(0, n_classes)
    
    # Create expert annotations
    expert_annotations = []
    
    # Expert 1: High expertise, high agreement with ground truth
    expert1_annotations = ground_truth.copy()
    error_indices = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    for idx in error_indices:
        expert1_annotations[idx] = np.random.randint(0, n_classes)
    
    expert1 = ExpertAnnotation(
        expert_id="Expert_Cardiologist_1",
        annotations=expert1_annotations,
        confidence_scores=np.random.beta(8, 2, n_samples),  # High confidence
        expertise_level="expert",
        years_experience=15,
        specialization="cardiology"
    )
    
    # Expert 2: Medium expertise, medium agreement
    expert2_annotations = ground_truth.copy()
    error_indices = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
    for idx in error_indices:
        expert2_annotations[idx] = np.random.randint(0, n_classes)
    
    expert2 = ExpertAnnotation(
        expert_id="Experienced_Physician_1",
        annotations=expert2_annotations,
        confidence_scores=np.random.beta(5, 3, n_samples),  # Medium confidence
        expertise_level="experienced",
        years_experience=8,
        specialization="internal_medicine"
    )
    
    expert_annotations = [expert1, expert2]
    
    print(f"Test data:")
    print(f"  Samples: {n_samples}")
    print(f"  Classes: {n_classes}")
    print(f"  Class names: {class_names}")
    print(f"  Experts: {len(expert_annotations)}")
    
    # Create expert comparison analyzer
    comparison = ExpertComparison(class_names=class_names)
    
    # Perform expert comparison
    print(f"\nPerforming expert comparison...")
    agreement = comparison.compare_with_experts(model_predictions, expert_annotations, class_names)
    
    # Display key results
    print(f"\nExpert Comparison Results:")
    print(f"  Fleiss' Kappa: {agreement.fleiss_kappa:.3f}")
    print(f"  Model-Expert Agreement (mean): {np.mean(list(agreement.model_expert_agreements.values())):.3f}")
    print(f"  Critical Condition Agreement: {agreement.critical_condition_agreement:.3f}")
    print(f"  Confidence-Accuracy Correlation: {agreement.confidence_accuracy_correlation:.3f}")
    
    # Generate report
    print(f"\nGenerating expert comparison report...")
    report = comparison.generate_expert_comparison_report(agreement)
    
    # Save report
    with open("expert_comparison_report.txt", "w") as f:
        f.write(report)
    
    print(f"Expert comparison report saved to: expert_comparison_report.txt")
    print("\nExpert Comparison Test Complete!")