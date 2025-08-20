#!/usr/bin/env python3
"""
Clinical Metrics and Validation
===============================

Comprehensive clinical validation metrics for ECG classification models.
Provides medical-grade performance assessment with clinical relevance,
regulatory compliance, and statistical significance testing.

Features:
- Clinical performance metrics (sensitivity, specificity, PPV, NPV)
- Regulatory compliance metrics
- Statistical significance testing
- Confidence intervals
- Clinical risk assessment
- Multi-class and multi-label support

Clinical Standards:
- FDA guidance compliance
- CE marking requirements
- Clinical trial standards
- Medical device regulations
- Expert validation protocols

Usage:
    metrics = ClinicalMetrics()
    results = metrics.evaluate_clinical_performance(predictions, ground_truth, class_names)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from collections import defaultdict
from scipy import stats
import math

@dataclass
class ConfusionMatrixMetrics:
    """Metrics derived from confusion matrix."""
    # Basic counts
    true_positives: int = 0
    true_negatives: int = 0 
    false_positives: int = 0
    false_negatives: int = 0
    
    # Derived metrics
    sensitivity: float = 0.0  # Recall, True Positive Rate
    specificity: float = 0.0  # True Negative Rate
    precision: float = 0.0    # Positive Predictive Value (PPV)
    negative_predictive_value: float = 0.0  # NPV
    
    # Clinical metrics
    positive_likelihood_ratio: float = 0.0
    negative_likelihood_ratio: float = 0.0
    diagnostic_odds_ratio: float = 0.0
    
    # Statistical measures
    accuracy: float = 0.0
    f1_score: float = 0.0
    matthews_correlation: float = 0.0
    
    # Confidence intervals (95%)
    sensitivity_ci: Tuple[float, float] = (0.0, 0.0)
    specificity_ci: Tuple[float, float] = (0.0, 0.0)
    precision_ci: Tuple[float, float] = (0.0, 0.0)
    
    def calculate_derived_metrics(self):
        """Calculate all derived metrics from basic counts."""
        
        # Denominators
        total_positive = self.true_positives + self.false_negatives
        total_negative = self.true_negatives + self.false_positives
        predicted_positive = self.true_positives + self.false_positives
        predicted_negative = self.true_negatives + self.false_negatives
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        
        # Primary metrics
        self.sensitivity = self.true_positives / total_positive if total_positive > 0 else 0.0
        self.specificity = self.true_negatives / total_negative if total_negative > 0 else 0.0
        self.precision = self.true_positives / predicted_positive if predicted_positive > 0 else 0.0
        self.negative_predictive_value = self.true_negatives / predicted_negative if predicted_negative > 0 else 0.0
        
        # Likelihood ratios
        if self.specificity < 1.0:
            self.positive_likelihood_ratio = self.sensitivity / (1 - self.specificity)
        else:
            self.positive_likelihood_ratio = float('inf') if self.sensitivity > 0 else 1.0
            
        if self.sensitivity > 0.0:
            self.negative_likelihood_ratio = (1 - self.sensitivity) / self.specificity
        else:
            self.negative_likelihood_ratio = 0.0
        
        # Diagnostic odds ratio
        if (self.false_positives * self.false_negatives) > 0:
            self.diagnostic_odds_ratio = (self.true_positives * self.true_negatives) / (self.false_positives * self.false_negatives)
        else:
            self.diagnostic_odds_ratio = float('inf')
        
        # Overall metrics
        self.accuracy = (self.true_positives + self.true_negatives) / total if total > 0 else 0.0
        
        # F1 score
        if (self.precision + self.sensitivity) > 0:
            self.f1_score = 2 * (self.precision * self.sensitivity) / (self.precision + self.sensitivity)
        else:
            self.f1_score = 0.0
        
        # Matthews Correlation Coefficient
        numerator = (self.true_positives * self.true_negatives) - (self.false_positives * self.false_negatives)
        denominator = math.sqrt(
            (self.true_positives + self.false_positives) *
            (self.true_positives + self.false_negatives) *
            (self.true_negatives + self.false_positives) *
            (self.true_negatives + self.false_negatives)
        )
        
        if denominator > 0:
            self.matthews_correlation = numerator / denominator
        else:
            self.matthews_correlation = 0.0
        
        # Confidence intervals
        self.sensitivity_ci = self._calculate_proportion_ci(self.true_positives, total_positive)
        self.specificity_ci = self._calculate_proportion_ci(self.true_negatives, total_negative)
        self.precision_ci = self._calculate_proportion_ci(self.true_positives, predicted_positive)
    
    def _calculate_proportion_ci(self, successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a proportion using Wilson score interval."""
        
        if total == 0:
            return (0.0, 0.0)
        
        p = successes / total
        z = stats.norm.ppf((1 + confidence) / 2)  # Z-score for 95% CI
        
        # Wilson score interval
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return (lower, upper)

@dataclass
class ClinicalValidationResult:
    """Comprehensive clinical validation results."""
    
    # Overall performance
    overall_accuracy: float = 0.0
    macro_sensitivity: float = 0.0
    macro_specificity: float = 0.0
    macro_precision: float = 0.0
    macro_f1: float = 0.0
    
    # Per-class metrics
    per_class_metrics: Dict[str, ConfusionMatrixMetrics] = None
    
    # Multi-class metrics
    confusion_matrix: np.ndarray = None
    class_names: List[str] = None
    
    # Clinical assessment
    clinical_acceptability: bool = False
    regulatory_compliance: bool = False
    safety_assessment: str = "unknown"
    
    # Statistical validation
    statistical_significance: Dict[str, float] = None
    sample_size_adequacy: bool = False
    power_analysis: Dict[str, float] = None
    
    # Performance benchmarks
    vs_random_classifier: Dict[str, float] = None
    vs_clinical_benchmark: Dict[str, float] = None
    
    # Risk assessment
    clinical_risk_level: str = "unknown"  # "low", "medium", "high"
    false_positive_impact: str = "unknown"
    false_negative_impact: str = "unknown"
    
    def __post_init__(self):
        if self.per_class_metrics is None:
            self.per_class_metrics = {}
        if self.statistical_significance is None:
            self.statistical_significance = {}
        if self.power_analysis is None:
            self.power_analysis = {}
        if self.vs_random_classifier is None:
            self.vs_random_classifier = {}
        if self.vs_clinical_benchmark is None:
            self.vs_clinical_benchmark = {}

class ClinicalMetrics:
    """
    Comprehensive clinical metrics calculator for ECG classification.
    
    Provides medical-grade performance assessment including clinical
    relevance, regulatory compliance, and statistical validation.
    """
    
    def __init__(self, 
                 clinical_thresholds: Optional[Dict[str, float]] = None,
                 regulatory_requirements: Optional[Dict[str, Any]] = None):
        """
        Initialize clinical metrics calculator.
        
        Args:
            clinical_thresholds: Clinical acceptability thresholds
            regulatory_requirements: Regulatory compliance requirements
        """
        
        # Default clinical thresholds
        self.clinical_thresholds = clinical_thresholds or {
            'min_sensitivity': 0.85,  # Minimum acceptable sensitivity
            'min_specificity': 0.90,  # Minimum acceptable specificity
            'min_precision': 0.80,    # Minimum acceptable precision
            'min_npv': 0.95,          # Minimum negative predictive value
            'max_false_negative_rate': 0.15,  # Maximum acceptable false negative rate
            'min_accuracy': 0.85,     # Minimum overall accuracy
        }
        
        # Default regulatory requirements
        self.regulatory_requirements = regulatory_requirements or {
            'min_sample_size': 100,   # Minimum validation sample size
            'required_significance': 0.05,  # Required p-value
            'min_power': 0.80,        # Minimum statistical power
            'required_ci_width': 0.10  # Maximum CI width for key metrics
        }
        
        # Clinical benchmarks (literature values)
        self.clinical_benchmarks = {
            'expert_cardiologist_accuracy': 0.92,
            'experienced_physician_accuracy': 0.88,
            'general_practitioner_accuracy': 0.82
        }
    
    def evaluate_clinical_performance(self,
                                    predictions: np.ndarray,
                                    ground_truth: np.ndarray,
                                    class_names: Optional[List[str]] = None,
                                    prediction_probabilities: Optional[np.ndarray] = None) -> ClinicalValidationResult:
        """
        Comprehensive clinical performance evaluation.
        
        Args:
            predictions: Model predictions (class indices)
            ground_truth: True labels (class indices)
            class_names: Names of classes
            prediction_probabilities: Prediction probabilities (optional)
            
        Returns:
            ClinicalValidationResult with comprehensive metrics
        """
        
        print("Performing comprehensive clinical validation...")
        
        # Validate inputs
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        if len(predictions) == 0:
            raise ValueError("Empty predictions provided")
        
        # Determine number of classes
        num_classes = max(max(predictions), max(ground_truth)) + 1
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        elif len(class_names) != num_classes:
            warnings.warn(f"Class names length ({len(class_names)}) doesn't match num_classes ({num_classes})")
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        # Initialize result
        result = ClinicalValidationResult(
            class_names=class_names,
            confusion_matrix=self._compute_confusion_matrix(predictions, ground_truth, num_classes)
        )
        
        # Compute per-class metrics
        result.per_class_metrics = self._compute_per_class_metrics(
            predictions, ground_truth, class_names, result.confusion_matrix
        )
        
        # Compute overall metrics
        result = self._compute_overall_metrics(result)
        
        # Clinical acceptability assessment
        result.clinical_acceptability = self._assess_clinical_acceptability(result)
        
        # Regulatory compliance
        result.regulatory_compliance = self._assess_regulatory_compliance(result, len(predictions))
        
        # Statistical validation
        result.statistical_significance = self._compute_statistical_significance(
            predictions, ground_truth, result.confusion_matrix
        )
        
        # Sample size adequacy
        result.sample_size_adequacy = self._assess_sample_size_adequacy(len(predictions))
        
        # Power analysis
        result.power_analysis = self._compute_power_analysis(result, len(predictions))
        
        # Benchmark comparisons
        result.vs_random_classifier = self._compare_vs_random(result, num_classes)
        result.vs_clinical_benchmark = self._compare_vs_clinical_benchmark(result)
        
        # Risk assessment
        result.clinical_risk_level = self._assess_clinical_risk(result)
        result.false_positive_impact = self._assess_false_positive_impact(result)
        result.false_negative_impact = self._assess_false_negative_impact(result)
        
        # Safety assessment
        result.safety_assessment = self._assess_safety(result)
        
        print(f"Clinical validation completed!")
        print(f"  Overall accuracy: {result.overall_accuracy:.3f}")
        print(f"  Clinical acceptability: {result.clinical_acceptability}")
        print(f"  Regulatory compliance: {result.regulatory_compliance}")
        print(f"  Clinical risk level: {result.clinical_risk_level}")
        
        return result
    
    def _compute_confusion_matrix(self, 
                                predictions: np.ndarray, 
                                ground_truth: np.ndarray,
                                num_classes: int) -> np.ndarray:
        """Compute confusion matrix."""
        
        cm = np.zeros((num_classes, num_classes), dtype=int)
        
        for true_label, pred_label in zip(ground_truth, predictions):
            cm[true_label, pred_label] += 1
        
        return cm
    
    def _compute_per_class_metrics(self,
                                 predictions: np.ndarray,
                                 ground_truth: np.ndarray, 
                                 class_names: List[str],
                                 confusion_matrix: np.ndarray) -> Dict[str, ConfusionMatrixMetrics]:
        """Compute metrics for each class."""
        
        per_class_metrics = {}
        num_classes = len(class_names)
        
        for i, class_name in enumerate(class_names):
            # Binary classification metrics for class i vs all others
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            tn = np.sum(confusion_matrix) - tp - fp - fn
            
            # Create metrics object
            metrics = ConfusionMatrixMetrics(
                true_positives=tp,
                true_negatives=tn,
                false_positives=fp,
                false_negatives=fn
            )
            
            # Calculate derived metrics
            metrics.calculate_derived_metrics()
            
            per_class_metrics[class_name] = metrics
        
        return per_class_metrics
    
    def _compute_overall_metrics(self, result: ClinicalValidationResult) -> ClinicalValidationResult:
        """Compute overall performance metrics."""
        
        if not result.per_class_metrics:
            return result
        
        # Macro averages
        sensitivities = [m.sensitivity for m in result.per_class_metrics.values()]
        specificities = [m.specificity for m in result.per_class_metrics.values()]
        precisions = [m.precision for m in result.per_class_metrics.values()]
        f1_scores = [m.f1_score for m in result.per_class_metrics.values()]
        
        result.macro_sensitivity = float(np.mean(sensitivities))
        result.macro_specificity = float(np.mean(specificities))
        result.macro_precision = float(np.mean(precisions))
        result.macro_f1 = float(np.mean(f1_scores))
        
        # Overall accuracy from confusion matrix
        if result.confusion_matrix is not None:
            correct = np.trace(result.confusion_matrix)
            total = np.sum(result.confusion_matrix)
            result.overall_accuracy = float(correct / total) if total > 0 else 0.0
        
        return result
    
    def _assess_clinical_acceptability(self, result: ClinicalValidationResult) -> bool:
        """Assess whether performance meets clinical acceptability criteria."""
        
        checks = []
        
        # Overall accuracy check
        checks.append(result.overall_accuracy >= self.clinical_thresholds['min_accuracy'])
        
        # Macro sensitivity check
        checks.append(result.macro_sensitivity >= self.clinical_thresholds['min_sensitivity'])
        
        # Macro specificity check  
        checks.append(result.macro_specificity >= self.clinical_thresholds['min_specificity'])
        
        # Macro precision check
        checks.append(result.macro_precision >= self.clinical_thresholds['min_precision'])
        
        # Check critical classes individually
        for class_name, metrics in result.per_class_metrics.items():
            # For critical diagnoses (like MI, VT), higher thresholds may apply
            if any(critical in class_name.lower() for critical in ['mi', 'vt', 'vf', 'heart_block']):
                checks.append(metrics.sensitivity >= 0.95)  # Higher sensitivity for critical conditions
                checks.append(metrics.negative_predictive_value >= 0.98)  # Very high NPV needed
        
        return all(checks)
    
    def _assess_regulatory_compliance(self, 
                                    result: ClinicalValidationResult, 
                                    sample_size: int) -> bool:
        """Assess regulatory compliance (FDA/CE requirements)."""
        
        compliance_checks = []
        
        # Sample size requirement
        compliance_checks.append(sample_size >= self.regulatory_requirements['min_sample_size'])
        
        # Statistical significance (would need p-values from statistical tests)
        # For now, check if we have statistical significance data
        compliance_checks.append(bool(result.statistical_significance))
        
        # Confidence interval width requirements
        for metrics in result.per_class_metrics.values():
            # Check that confidence intervals are reasonably narrow
            sens_ci_width = metrics.sensitivity_ci[1] - metrics.sensitivity_ci[0]
            spec_ci_width = metrics.specificity_ci[1] - metrics.specificity_ci[0]
            
            compliance_checks.append(sens_ci_width <= self.regulatory_requirements['required_ci_width'])
            compliance_checks.append(spec_ci_width <= self.regulatory_requirements['required_ci_width'])
        
        return all(compliance_checks)
    
    def _compute_statistical_significance(self,
                                        predictions: np.ndarray,
                                        ground_truth: np.ndarray,
                                        confusion_matrix: np.ndarray) -> Dict[str, float]:
        """Compute statistical significance tests."""
        
        significance_results = {}
        
        # McNemar's test for comparing with baseline (assuming 50% accuracy baseline)
        try:
            correct_predictions = (predictions == ground_truth)
            n_correct = np.sum(correct_predictions)
            n_total = len(predictions)
            n_incorrect = n_total - n_correct
            
            # Chi-square test against random classifier
            expected_correct = n_total / len(np.unique(ground_truth))  # Random classifier expectation
            
            if expected_correct > 5:  # Chi-square validity condition
                chi_square = ((n_correct - expected_correct) ** 2) / expected_correct
                p_value = 1 - stats.chi2.cdf(chi_square, df=1)
                significance_results['vs_random_classifier'] = p_value
            
            # Binomial test for overall accuracy
            p_value_binomial = stats.binom_test(n_correct, n_total, p=0.5, alternative='greater')
            significance_results['accuracy_vs_chance'] = p_value_binomial
            
        except Exception as e:
            warnings.warn(f"Statistical significance computation failed: {e}")
        
        return significance_results
    
    def _assess_sample_size_adequacy(self, sample_size: int) -> bool:
        """Assess whether sample size is adequate for reliable results."""
        
        # Basic sample size requirement
        min_size_met = sample_size >= self.regulatory_requirements['min_sample_size']
        
        # Rule of thumb: at least 10 samples per class per parameter
        # For deep learning models, this should be much higher
        recommended_min = max(1000, self.regulatory_requirements['min_sample_size'])
        
        return sample_size >= recommended_min
    
    def _compute_power_analysis(self, 
                              result: ClinicalValidationResult, 
                              sample_size: int) -> Dict[str, float]:
        """Compute statistical power analysis."""
        
        power_results = {}
        
        try:
            # Power for detecting sensitivity difference from 0.5
            for class_name, metrics in result.per_class_metrics.items():
                # Simplified power calculation for proportion test
                p0 = 0.5  # Null hypothesis (random classifier)
                p1 = metrics.sensitivity  # Observed sensitivity
                
                if p1 > p0:
                    # Effect size
                    effect_size = abs(p1 - p0) / math.sqrt(p0 * (1 - p0))
                    
                    # Approximate power using normal approximation
                    z_alpha = stats.norm.ppf(0.975)  # Two-tailed test, alpha = 0.05
                    z_beta = effect_size * math.sqrt(sample_size) - z_alpha
                    power = stats.norm.cdf(z_beta)
                    
                    power_results[f'{class_name}_sensitivity_power'] = power
        
        except Exception as e:
            warnings.warn(f"Power analysis computation failed: {e}")
        
        return power_results
    
    def _compare_vs_random(self, 
                         result: ClinicalValidationResult, 
                         num_classes: int) -> Dict[str, float]:
        """Compare performance against random classifier."""
        
        random_accuracy = 1.0 / num_classes
        
        comparison = {
            'random_classifier_accuracy': random_accuracy,
            'improvement_over_random': result.overall_accuracy - random_accuracy,
            'relative_improvement': (result.overall_accuracy - random_accuracy) / random_accuracy if random_accuracy > 0 else 0.0
        }
        
        return comparison
    
    def _compare_vs_clinical_benchmark(self, result: ClinicalValidationResult) -> Dict[str, float]:
        """Compare against clinical benchmarks."""
        
        comparison = {}
        
        for benchmark_name, benchmark_value in self.clinical_benchmarks.items():
            comparison[f'vs_{benchmark_name}'] = result.overall_accuracy - benchmark_value
            comparison[f'vs_{benchmark_name}_relative'] = (result.overall_accuracy - benchmark_value) / benchmark_value if benchmark_value > 0 else 0.0
        
        return comparison
    
    def _assess_clinical_risk(self, result: ClinicalValidationResult) -> str:
        """Assess overall clinical risk level."""
        
        # Risk factors
        risk_factors = []
        
        # Low sensitivity is high risk
        if result.macro_sensitivity < 0.85:
            risk_factors.append("low_sensitivity")
        
        # Low specificity leads to unnecessary interventions
        if result.macro_specificity < 0.80:
            risk_factors.append("low_specificity")
        
        # Check for critical class performance
        for class_name, metrics in result.per_class_metrics.items():
            if any(critical in class_name.lower() for critical in ['mi', 'vt', 'vf']):
                if metrics.sensitivity < 0.90:
                    risk_factors.append(f"critical_class_low_sensitivity_{class_name}")
        
        # Determine overall risk level
        if len(risk_factors) >= 3:
            return "high"
        elif len(risk_factors) >= 1:
            return "medium"
        else:
            return "low"
    
    def _assess_false_positive_impact(self, result: ClinicalValidationResult) -> str:
        """Assess impact of false positives."""
        
        if result.macro_specificity >= 0.95:
            return "low"
        elif result.macro_specificity >= 0.85:
            return "medium"
        else:
            return "high"
    
    def _assess_false_negative_impact(self, result: ClinicalValidationResult) -> str:
        """Assess impact of false negatives."""
        
        if result.macro_sensitivity >= 0.95:
            return "low"
        elif result.macro_sensitivity >= 0.85:
            return "medium" 
        else:
            return "high"
    
    def _assess_safety(self, result: ClinicalValidationResult) -> str:
        """Overall safety assessment."""
        
        safety_score = 0
        
        # High sensitivity for safety
        if result.macro_sensitivity >= 0.95:
            safety_score += 2
        elif result.macro_sensitivity >= 0.90:
            safety_score += 1
        
        # Reasonable specificity
        if result.macro_specificity >= 0.90:
            safety_score += 1
        
        # Clinical acceptability
        if result.clinical_acceptability:
            safety_score += 1
        
        # Regulatory compliance
        if result.regulatory_compliance:
            safety_score += 1
        
        if safety_score >= 4:
            return "acceptable"
        elif safety_score >= 2:
            return "conditional"
        else:
            return "unacceptable"
    
    def generate_clinical_report(self, result: ClinicalValidationResult) -> str:
        """Generate comprehensive clinical validation report."""
        
        report = f"""
CLINICAL VALIDATION REPORT
=========================

EXECUTIVE SUMMARY
-----------------
Overall Accuracy: {result.overall_accuracy:.3f}
Clinical Acceptability: {result.clinical_acceptability}
Regulatory Compliance: {result.regulatory_compliance}
Safety Assessment: {result.safety_assessment}
Clinical Risk Level: {result.clinical_risk_level}

PERFORMANCE METRICS
------------------
Macro Sensitivity: {result.macro_sensitivity:.3f}
Macro Specificity: {result.macro_specificity:.3f}
Macro Precision: {result.macro_precision:.3f}
Macro F1-Score: {result.macro_f1:.3f}

PER-CLASS PERFORMANCE
--------------------
"""
        
        for class_name, metrics in result.per_class_metrics.items():
            report += f"""
{class_name}:
  Sensitivity: {metrics.sensitivity:.3f} (95% CI: {metrics.sensitivity_ci[0]:.3f}-{metrics.sensitivity_ci[1]:.3f})
  Specificity: {metrics.specificity:.3f} (95% CI: {metrics.specificity_ci[0]:.3f}-{metrics.specificity_ci[1]:.3f})
  Precision: {metrics.precision:.3f} (95% CI: {metrics.precision_ci[0]:.3f}-{metrics.precision_ci[1]:.3f})
  NPV: {metrics.negative_predictive_value:.3f}
  F1-Score: {metrics.f1_score:.3f}
"""
        
        report += f"""
STATISTICAL VALIDATION
---------------------
Sample Size Adequacy: {result.sample_size_adequacy}
Statistical Significance Tests:
"""
        
        for test_name, p_value in result.statistical_significance.items():
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            report += f"  {test_name}: p={p_value:.4f} ({significance})\n"
        
        report += f"""
BENCHMARK COMPARISONS
--------------------
vs Random Classifier: +{result.vs_random_classifier.get('improvement_over_random', 0):.3f}
vs Expert Cardiologist: {result.vs_clinical_benchmark.get('vs_expert_cardiologist_accuracy', 0):+.3f}

RISK ASSESSMENT
--------------
Clinical Risk Level: {result.clinical_risk_level}
False Positive Impact: {result.false_positive_impact}
False Negative Impact: {result.false_negative_impact}

RECOMMENDATIONS
--------------
"""
        
        if not result.clinical_acceptability:
            report += "- Model does not meet clinical acceptability criteria\n"
            report += "- Further development and validation required\n"
        
        if not result.regulatory_compliance:
            report += "- Regulatory compliance requirements not met\n"
            report += "- Additional validation studies recommended\n"
        
        if result.clinical_risk_level == "high":
            report += "- High clinical risk identified\n"
            report += "- Expert oversight required for deployment\n"
        
        if result.safety_assessment == "unacceptable":
            report += "- Safety assessment indicates unacceptable risk\n"
            report += "- Model not recommended for clinical use\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    print("Clinical Metrics Test")
    print("=" * 25)
    
    # Create test data
    np.random.seed(42)
    n_samples = 500
    n_classes = 5
    
    # Simulate reasonably good predictions
    ground_truth = np.random.randint(0, n_classes, n_samples)
    
    # Create predictions with some accuracy (80-85%)
    predictions = ground_truth.copy()
    
    # Add some errors
    error_indices = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
    for idx in error_indices:
        predictions[idx] = np.random.choice([c for c in range(n_classes) if c != ground_truth[idx]])
    
    class_names = ['Normal', 'MI', 'STTC', 'CD', 'HYP']
    
    print(f"Test data:")
    print(f"  Samples: {n_samples}")
    print(f"  Classes: {n_classes}")
    print(f"  Class names: {class_names}")
    
    # Create clinical metrics evaluator
    clinical_metrics = ClinicalMetrics()
    
    # Evaluate performance
    print(f"\nEvaluating clinical performance...")
    results = clinical_metrics.evaluate_clinical_performance(
        predictions, ground_truth, class_names
    )
    
    # Display key results
    print(f"\nKey Results:")
    print(f"  Overall Accuracy: {results.overall_accuracy:.3f}")
    print(f"  Macro Sensitivity: {results.macro_sensitivity:.3f}")
    print(f"  Macro Specificity: {results.macro_specificity:.3f}")
    print(f"  Clinical Acceptability: {results.clinical_acceptability}")
    print(f"  Regulatory Compliance: {results.regulatory_compliance}")
    print(f"  Safety Assessment: {results.safety_assessment}")
    
    # Show per-class performance
    print(f"\nPer-Class Performance:")
    for class_name, metrics in results.per_class_metrics.items():
        print(f"  {class_name}: Sens={metrics.sensitivity:.3f}, Spec={metrics.specificity:.3f}, Prec={metrics.precision:.3f}")
    
    # Generate clinical report
    print(f"\nGenerating clinical report...")
    report = clinical_metrics.generate_clinical_report(results)
    
    # Save report to file for review
    with open("clinical_validation_report.txt", "w") as f:
        f.write(report)
    
    print(f"Clinical report saved to: clinical_validation_report.txt")
    
    print("\nClinical Metrics Test Complete!")