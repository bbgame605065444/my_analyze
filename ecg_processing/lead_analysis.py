#!/usr/bin/env python3
"""
Multi-Lead ECG Analysis Module
=============================

Specialized analysis for 12-lead ECG data in CoT-RAG Stage 4.
Provides lead-specific analysis, cross-lead correlation, and
comprehensive cardiac assessment using multiple ECG perspectives.

Features:
- 12-lead ECG processing and validation
- Lead-specific morphology analysis  
- Cross-lead correlation and consistency
- Cardiac axis calculation
- Localized ischemia detection
- Bundle branch block analysis
- Comprehensive cardiac assessment

Clinical Applications:
- ST-elevation myocardial infarction (STEMI) localization
- Bundle branch block detection and classification
- Cardiac axis deviation assessment
- Multi-lead arrhythmia analysis
- Lead quality assessment and artifact detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import warnings

from .signal_preprocessing import ECGLead, ECGSignalProcessor
from .feature_extraction import ECGFeatureExtractor, ECGFeatures

class CardiacAxis(Enum):
    """Cardiac axis classification."""
    NORMAL = "normal"           # -30° to +90°
    LEFT_DEVIATION = "left_deviation"      # -30° to -90°
    RIGHT_DEVIATION = "right_deviation"     # +90° to +180°
    EXTREME_DEVIATION = "extreme_deviation"  # -90° to -180°
    INDETERMINATE = "indeterminate"

class LeadGroup(Enum):
    """ECG lead groupings for cardiac region analysis."""
    LIMB_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]
    PRECORDIAL_LEADS = ["V1", "V2", "V3", "V4", "V5", "V6"]
    INFERIOR_LEADS = ["II", "III", "aVF"]
    LATERAL_LEADS = ["I", "aVL", "V5", "V6"]
    ANTERIOR_LEADS = ["V1", "V2", "V3", "V4"]
    SEPTAL_LEADS = ["V1", "V2"]
    LATERAL_WALL = ["I", "aVL", "V5", "V6"]
    INFERIOR_WALL = ["II", "III", "aVF"]
    ANTERIOR_WALL = ["V3", "V4"]

@dataclass
class LeadAnalysisResult:
    """Results from single lead analysis."""
    lead_name: str
    signal_quality: float
    features: ECGFeatures
    morphology_score: float
    clinical_findings: Dict[str, float]
    artifacts_detected: List[str]
    analysis_confidence: float

@dataclass
class MultiLeadAnalysisResult:
    """Results from multi-lead ECG analysis."""
    lead_results: Dict[str, LeadAnalysisResult]
    cross_lead_correlations: Dict[Tuple[str, str], float]
    cardiac_axis: CardiacAxis
    axis_degrees: float
    bundle_branch_analysis: Dict[str, any]
    ischemia_analysis: Dict[str, any]
    overall_rhythm_assessment: Dict[str, any]
    lead_concordance: float
    clinical_summary: Dict[str, any]
    analysis_metadata: Dict[str, any]

class MultiLeadAnalyzer:
    """
    Comprehensive multi-lead ECG analysis system.
    
    Analyzes standard 12-lead ECG data for clinical diagnosis,
    providing lead-specific analysis and cross-lead integration.
    """
    
    def __init__(self, 
                 sampling_rate: float = 500.0,
                 enable_advanced_analysis: bool = True):
        """
        Initialize multi-lead ECG analyzer.
        
        Args:
            sampling_rate: ECG sampling rate (Hz)
            enable_advanced_analysis: Enable advanced clinical analysis
        """
        self.sampling_rate = sampling_rate
        self.enable_advanced_analysis = enable_advanced_analysis
        
        # Initialize processing components
        self.signal_processor = ECGSignalProcessor(sampling_rate=sampling_rate)
        self.feature_extractor = ECGFeatureExtractor(sampling_rate=sampling_rate)
        
        # Analysis statistics
        self.analysis_stats = {}
        
        # Standard 12-lead configuration
        self.standard_leads = [lead.value for lead in ECGLead]
        self.lead_groups = {group.name: group.value for group in LeadGroup}
    
    def analyze_multi_lead_ecg(self, 
                              ecg_data: Dict[str, np.ndarray],
                              patient_id: Optional[str] = None) -> MultiLeadAnalysisResult:
        """
        Perform comprehensive multi-lead ECG analysis.
        
        Args:
            ecg_data: Dictionary of lead_name -> signal_array
            patient_id: Optional patient identifier
            
        Returns:
            MultiLeadAnalysisResult with comprehensive analysis
        """
        print(f"Analyzing multi-lead ECG with {len(ecg_data)} leads...")
        
        try:
            # Validate input leads
            available_leads = set(ecg_data.keys())
            standard_leads_set = set(self.standard_leads)
            
            # Check for standard 12-lead configuration
            missing_leads = standard_leads_set - available_leads
            extra_leads = available_leads - standard_leads_set
            
            if missing_leads:
                warnings.warn(f"Missing standard leads: {missing_leads}")
            if extra_leads:
                print(f"Additional leads detected: {extra_leads}")
            
            # Process each lead individually
            lead_results = {}
            processed_signals = {}
            
            for lead_name, signal in ecg_data.items():
                try:
                    lead_result = self._analyze_single_lead(signal, lead_name)
                    lead_results[lead_name] = lead_result
                    
                    # Store processed signal for cross-lead analysis
                    processed_result = self.signal_processor.preprocess_signal(
                        {lead_name: signal}, sampling_rate=self.sampling_rate
                    )
                    if processed_result['processing_successful']:
                        processed_signals[lead_name] = processed_result['processed_ecg'][lead_name]
                    
                except Exception as e:
                    warnings.warn(f"Failed to analyze lead {lead_name}: {e}")
                    # Create minimal result for failed lead
                    lead_results[lead_name] = self._create_failed_lead_result(lead_name, str(e))
            
            # Cross-lead analysis
            cross_correlations = self._compute_cross_lead_correlations(processed_signals)
            
            # Cardiac axis analysis
            cardiac_axis, axis_degrees = self._analyze_cardiac_axis(lead_results)
            
            # Bundle branch analysis
            bundle_branch_analysis = self._analyze_bundle_branches(lead_results, processed_signals)
            
            # Ischemia analysis
            ischemia_analysis = self._analyze_ischemia_patterns(lead_results, processed_signals)
            
            # Overall rhythm assessment
            rhythm_assessment = self._assess_overall_rhythm(lead_results, processed_signals)
            
            # Lead concordance assessment
            lead_concordance = self._assess_lead_concordance(lead_results, cross_correlations)
            
            # Clinical summary
            clinical_summary = self._generate_clinical_summary(
                lead_results, cardiac_axis, bundle_branch_analysis, 
                ischemia_analysis, rhythm_assessment
            )
            
            # Analysis metadata
            metadata = {
                'patient_id': patient_id,
                'total_leads_analyzed': len(lead_results),
                'successful_analyses': sum(1 for r in lead_results.values() 
                                         if r.analysis_confidence > 0.5),
                'standard_12_lead_complete': len(missing_leads) == 0,
                'analysis_timestamp': 'current_time',  # Would use datetime in production
                'processing_version': '1.0.0'
            }
            
            # Create comprehensive result
            result = MultiLeadAnalysisResult(
                lead_results=lead_results,
                cross_lead_correlations=cross_correlations,
                cardiac_axis=cardiac_axis,
                axis_degrees=axis_degrees,
                bundle_branch_analysis=bundle_branch_analysis,
                ischemia_analysis=ischemia_analysis,
                overall_rhythm_assessment=rhythm_assessment,
                lead_concordance=lead_concordance,
                clinical_summary=clinical_summary,
                analysis_metadata=metadata
            )
            
            self._update_analysis_stats(result, success=True)
            return result
            
        except Exception as e:
            print(f"Multi-lead analysis failed: {e}")
            # Return minimal failed result
            failed_result = self._create_failed_analysis_result(str(e))
            self._update_analysis_stats(failed_result, success=False)
            return failed_result
    
    def _analyze_single_lead(self, signal: np.ndarray, lead_name: str) -> LeadAnalysisResult:
        """Analyze a single ECG lead comprehensively."""
        
        # Preprocess signal
        processed_result = self.signal_processor.preprocess_signal(
            {lead_name: signal}, sampling_rate=self.sampling_rate
        )
        
        if not processed_result['processing_successful']:
            return self._create_failed_lead_result(lead_name, "Signal processing failed")
        
        processed_signal = processed_result['processed_ecg'][lead_name]
        signal_quality = processed_result['metadata']['signal_quality'][lead_name]
        r_peaks = processed_result['metadata']['r_peaks'].get(lead_name)
        
        # Extract features
        features = self.feature_extractor.extract_features(
            processed_signal, r_peaks, lead_name
        )
        
        if not features.metadata['feature_extraction_success']:
            return self._create_failed_lead_result(lead_name, "Feature extraction failed")
        
        # Lead-specific morphology analysis
        morphology_score = self._assess_lead_morphology(processed_signal, r_peaks, lead_name)
        
        # Clinical findings specific to this lead
        clinical_findings = self._extract_lead_clinical_findings(
            processed_signal, features, lead_name
        )
        
        # Artifact detection
        artifacts_detected = self._detect_lead_artifacts(processed_signal, lead_name)
        
        # Overall analysis confidence
        analysis_confidence = self._calculate_lead_confidence(
            signal_quality, morphology_score, len(artifacts_detected)
        )
        
        return LeadAnalysisResult(
            lead_name=lead_name,
            signal_quality=signal_quality,
            features=features,
            morphology_score=morphology_score,
            clinical_findings=clinical_findings,
            artifacts_detected=artifacts_detected,
            analysis_confidence=analysis_confidence
        )
    
    def _compute_cross_lead_correlations(self, 
                                       processed_signals: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
        """Compute cross-correlations between ECG leads."""
        correlations = {}
        
        lead_names = list(processed_signals.keys())
        
        for i, lead1 in enumerate(lead_names):
            for j, lead2 in enumerate(lead_names[i+1:], i+1):
                try:
                    # Ensure signals are same length
                    min_len = min(len(processed_signals[lead1]), len(processed_signals[lead2]))
                    signal1 = processed_signals[lead1][:min_len]
                    signal2 = processed_signals[lead2][:min_len]
                    
                    # Compute Pearson correlation
                    correlation = np.corrcoef(signal1, signal2)[0, 1]
                    
                    if not np.isnan(correlation):
                        correlations[(lead1, lead2)] = float(correlation)
                        correlations[(lead2, lead1)] = float(correlation)  # Symmetric
                    
                except Exception:
                    correlations[(lead1, lead2)] = 0.0
                    correlations[(lead2, lead1)] = 0.0
        
        return correlations
    
    def _analyze_cardiac_axis(self, 
                            lead_results: Dict[str, LeadAnalysisResult]) -> Tuple[CardiacAxis, float]:
        """Analyze cardiac electrical axis using limb leads."""
        
        # Need leads I and aVF for axis calculation
        if 'I' not in lead_results or 'aVF' not in lead_results:
            return CardiacAxis.INDETERMINATE, 0.0
        
        try:
            # Get QRS amplitude in leads I and aVF
            lead_i_qrs = lead_results['I'].features.morphological.get('qrs_amplitude_mean', 0.0)
            lead_avf_qrs = lead_results['aVF'].features.morphological.get('qrs_amplitude_mean', 0.0)
            
            # Calculate axis in degrees
            # Simplified axis calculation using lead I and aVF
            if lead_i_qrs == 0 and lead_avf_qrs == 0:
                return CardiacAxis.INDETERMINATE, 0.0
            
            axis_radians = np.arctan2(lead_avf_qrs, lead_i_qrs)
            axis_degrees = np.degrees(axis_radians)
            
            # Normalize to -180 to +180 range
            if axis_degrees > 180:
                axis_degrees -= 360
            elif axis_degrees < -180:
                axis_degrees += 360
            
            # Classify axis deviation
            if -30 <= axis_degrees <= 90:
                cardiac_axis = CardiacAxis.NORMAL
            elif -30 > axis_degrees >= -90:
                cardiac_axis = CardiacAxis.LEFT_DEVIATION
            elif 90 < axis_degrees <= 180:
                cardiac_axis = CardiacAxis.RIGHT_DEVIATION
            else:  # -90 > axis_degrees >= -180
                cardiac_axis = CardiacAxis.EXTREME_DEVIATION
            
            return cardiac_axis, float(axis_degrees)
            
        except Exception:
            return CardiacAxis.INDETERMINATE, 0.0
    
    def _analyze_bundle_branches(self, 
                               lead_results: Dict[str, LeadAnalysisResult],
                               processed_signals: Dict[str, np.ndarray]) -> Dict[str, any]:
        """Analyze for bundle branch blocks."""
        
        analysis = {
            'right_bundle_branch_block': False,
            'left_bundle_branch_block': False,
            'incomplete_rbbb': False,
            'incomplete_lbbb': False,
            'qrs_duration_ms': 0.0,
            'bundle_block_confidence': 0.0,
            'morphology_patterns': {}
        }
        
        try:
            # Calculate average QRS duration across leads
            qrs_durations = []
            for lead_name, result in lead_results.items():
                if result.analysis_confidence > 0.5:
                    qrs_width = result.features.morphological.get('qrs_width_mean', 0.0)
                    qrs_durations.append(qrs_width * 1000)  # Convert to ms
            
            if qrs_durations:
                avg_qrs_duration = np.mean(qrs_durations)
                analysis['qrs_duration_ms'] = float(avg_qrs_duration)
                
                # Bundle branch block criteria (simplified)
                if avg_qrs_duration >= 120:  # ms
                    # Check for RBBB pattern (rsR' in V1, wide S in I, V6)
                    if self._check_rbbb_pattern(lead_results):
                        analysis['right_bundle_branch_block'] = True
                        analysis['bundle_block_confidence'] = 0.8
                    
                    # Check for LBBB pattern (wide R in I, V6, deep S in V1)
                    elif self._check_lbbb_pattern(lead_results):
                        analysis['left_bundle_branch_block'] = True
                        analysis['bundle_block_confidence'] = 0.8
                
                elif 100 <= avg_qrs_duration < 120:  # Incomplete blocks
                    if self._check_rbbb_pattern(lead_results, incomplete=True):
                        analysis['incomplete_rbbb'] = True
                        analysis['bundle_block_confidence'] = 0.6
                    elif self._check_lbbb_pattern(lead_results, incomplete=True):
                        analysis['incomplete_lbbb'] = True
                        analysis['bundle_block_confidence'] = 0.6
                
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_ischemia_patterns(self, 
                                 lead_results: Dict[str, LeadAnalysisResult],
                                 processed_signals: Dict[str, np.ndarray]) -> Dict[str, any]:
        """Analyze for ischemia and infarction patterns."""
        
        analysis = {
            'st_elevation': {},
            'st_depression': {},
            't_wave_inversion': {},
            'q_waves': {},
            'regional_findings': {},
            'acute_changes': False,
            'chronic_changes': False,
            'ischemia_confidence': 0.0
        }
        
        try:
            # Analyze ST segment changes in each lead
            for lead_name, result in lead_results.items():
                if result.analysis_confidence > 0.5:
                    st_deviation = result.clinical_findings.get('st_deviation', 0.0)
                    
                    # ST elevation threshold (simplified)
                    if st_deviation > 0.1:  # 1 mV
                        analysis['st_elevation'][lead_name] = float(st_deviation)
                    
                    # ST depression threshold
                    elif st_deviation < -0.05:  # -0.5 mV
                        analysis['st_depression'][lead_name] = abs(float(st_deviation))
            
            # Regional analysis based on lead groups
            analysis['regional_findings'] = self._analyze_regional_ischemia(
                analysis['st_elevation'], analysis['st_depression']
            )
            
            # Determine if changes suggest acute vs chronic
            total_st_changes = len(analysis['st_elevation']) + len(analysis['st_depression'])
            if total_st_changes >= 2:  # Changes in multiple leads
                analysis['acute_changes'] = True
                analysis['ischemia_confidence'] = min(0.9, total_st_changes / 6.0)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _assess_overall_rhythm(self, 
                             lead_results: Dict[str, LeadAnalysisResult],
                             processed_signals: Dict[str, np.ndarray]) -> Dict[str, any]:
        """Assess overall cardiac rhythm across all leads."""
        
        rhythm_assessment = {
            'primary_rhythm': 'sinus_rhythm',
            'rhythm_regularity': 0.0,
            'average_heart_rate': 0.0,
            'arrhythmia_detected': False,
            'arrhythmia_type': None,
            'rhythm_confidence': 0.0,
            'p_wave_analysis': {},
            'atrial_fibrillation_probability': 0.0
        }
        
        try:
            # Collect rhythm metrics from all leads
            heart_rates = []
            regularity_scores = []
            
            for lead_name, result in lead_results.items():
                if result.analysis_confidence > 0.5:
                    hr = result.features.time_domain.get('mean_hr', 75.0)
                    regularity = result.features.clinical.get('rhythm_regularity', 0.5)
                    
                    heart_rates.append(hr)
                    regularity_scores.append(regularity)
            
            if heart_rates:
                rhythm_assessment['average_heart_rate'] = float(np.mean(heart_rates))
                rhythm_assessment['rhythm_regularity'] = float(np.mean(regularity_scores))
                
                # Rhythm classification (simplified)
                avg_regularity = np.mean(regularity_scores)
                if avg_regularity < 0.3:  # Very irregular
                    rhythm_assessment['arrhythmia_detected'] = True
                    rhythm_assessment['arrhythmia_type'] = 'atrial_fibrillation'
                    rhythm_assessment['atrial_fibrillation_probability'] = 1.0 - avg_regularity
                    rhythm_assessment['primary_rhythm'] = 'atrial_fibrillation'
                elif avg_regularity < 0.7:  # Moderately irregular
                    rhythm_assessment['arrhythmia_detected'] = True
                    rhythm_assessment['arrhythmia_type'] = 'irregular_rhythm'
                    rhythm_assessment['primary_rhythm'] = 'irregular_sinus'
                
                # Rhythm confidence based on consistency across leads
                hr_consistency = 1.0 - (np.std(heart_rates) / (np.mean(heart_rates) + 1e-10))
                reg_consistency = 1.0 - np.std(regularity_scores)
                rhythm_assessment['rhythm_confidence'] = float((hr_consistency + reg_consistency) / 2.0)
            
        except Exception as e:
            rhythm_assessment['error'] = str(e)
        
        return rhythm_assessment
    
    def _assess_lead_concordance(self, 
                               lead_results: Dict[str, LeadAnalysisResult],
                               cross_correlations: Dict[Tuple[str, str], float]) -> float:
        """Assess concordance between leads."""
        
        try:
            # Calculate overall concordance based on correlations
            correlations = [abs(corr) for corr in cross_correlations.values()]
            
            if correlations:
                # Higher correlation indicates better concordance
                mean_correlation = np.mean(correlations)
                concordance_score = mean_correlation
            else:
                concordance_score = 0.5  # Default when no correlations available
            
            # Adjust based on lead quality consistency
            quality_scores = [result.signal_quality for result in lead_results.values()]
            if quality_scores:
                quality_consistency = 1.0 - (np.std(quality_scores) / (np.mean(quality_scores) + 1e-10))
                concordance_score = (concordance_score + quality_consistency) / 2.0
            
            return float(np.clip(concordance_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default concordance score
    
    def _generate_clinical_summary(self, 
                                 lead_results: Dict[str, LeadAnalysisResult],
                                 cardiac_axis: CardiacAxis,
                                 bundle_branch_analysis: Dict,
                                 ischemia_analysis: Dict,
                                 rhythm_assessment: Dict) -> Dict[str, any]:
        """Generate comprehensive clinical summary."""
        
        summary = {
            'overall_interpretation': 'normal_ecg',
            'abnormal_findings': [],
            'clinical_significance': [],
            'recommendations': [],
            'diagnostic_confidence': 0.0,
            'key_measurements': {},
            'risk_assessment': 'low'
        }
        
        try:
            # Collect key measurements
            hr_values = [result.features.time_domain.get('mean_hr', 75.0) 
                        for result in lead_results.values() if result.analysis_confidence > 0.5]
            
            if hr_values:
                summary['key_measurements']['heart_rate'] = {
                    'value': float(np.mean(hr_values)),
                    'unit': 'bpm',
                    'range': 'normal' if 60 <= np.mean(hr_values) <= 100 else 'abnormal'
                }
            
            # QRS duration
            qrs_duration = bundle_branch_analysis.get('qrs_duration_ms', 0.0)
            summary['key_measurements']['qrs_duration'] = {
                'value': float(qrs_duration),
                'unit': 'ms',
                'range': 'normal' if qrs_duration < 120 else 'prolonged'
            }
            
            # Cardiac axis
            summary['key_measurements']['cardiac_axis'] = {
                'value': cardiac_axis.value,
                'degrees': bundle_branch_analysis.get('axis_degrees', 0.0),
                'range': 'normal' if cardiac_axis == CardiacAxis.NORMAL else 'deviated'
            }
            
            # Identify abnormal findings
            abnormal_findings = []
            
            # Rhythm abnormalities
            if rhythm_assessment['arrhythmia_detected']:
                abnormal_findings.append(f"Arrhythmia: {rhythm_assessment['arrhythmia_type']}")
                summary['overall_interpretation'] = 'abnormal_ecg'
                
            # Bundle branch blocks
            if bundle_branch_analysis['right_bundle_branch_block']:
                abnormal_findings.append("Right bundle branch block")
                summary['overall_interpretation'] = 'abnormal_ecg'
            if bundle_branch_analysis['left_bundle_branch_block']:
                abnormal_findings.append("Left bundle branch block")
                summary['overall_interpretation'] = 'abnormal_ecg'
                
            # Ischemia findings
            if ischemia_analysis['acute_changes']:
                abnormal_findings.append("Acute ischemic changes")
                summary['overall_interpretation'] = 'abnormal_ecg'
                summary['risk_assessment'] = 'high'
                
            # Axis deviation
            if cardiac_axis != CardiacAxis.NORMAL:
                abnormal_findings.append(f"Cardiac axis: {cardiac_axis.value}")
                if summary['overall_interpretation'] == 'normal_ecg':
                    summary['overall_interpretation'] = 'borderline_ecg'
            
            summary['abnormal_findings'] = abnormal_findings
            
            # Calculate overall diagnostic confidence
            lead_confidences = [result.analysis_confidence for result in lead_results.values()]
            if lead_confidences:
                summary['diagnostic_confidence'] = float(np.mean(lead_confidences))
            
            # Generate recommendations
            recommendations = []
            if summary['overall_interpretation'] == 'abnormal_ecg':
                recommendations.append("Clinical correlation recommended")
                if ischemia_analysis['acute_changes']:
                    recommendations.append("Consider urgent cardiology consultation")
                if rhythm_assessment['arrhythmia_detected']:
                    recommendations.append("Consider Holter monitoring")
            
            summary['recommendations'] = recommendations
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def _check_rbbb_pattern(self, 
                          lead_results: Dict[str, LeadAnalysisResult],
                          incomplete: bool = False) -> bool:
        """Check for right bundle branch block pattern."""
        
        # Simplified RBBB detection
        # Requires wide QRS and specific morphology patterns
        
        try:
            # Need V1 and V6 for RBBB diagnosis
            if 'V1' not in lead_results or 'V6' not in lead_results:
                return False
            
            v1_result = lead_results['V1']
            v6_result = lead_results['V6']
            
            if (v1_result.analysis_confidence < 0.5 or 
                v6_result.analysis_confidence < 0.5):
                return False
            
            # Check QRS morphology patterns (simplified)
            # In RBBB: V1 shows rsR' pattern, V6 shows wide S
            v1_qrs_width = v1_result.features.morphological.get('qrs_width_mean', 0.0) * 1000
            v6_qrs_width = v6_result.features.morphological.get('qrs_width_mean', 0.0) * 1000
            
            threshold = 100 if incomplete else 120  # ms
            
            return (v1_qrs_width >= threshold or v6_qrs_width >= threshold)
            
        except Exception:
            return False
    
    def _check_lbbb_pattern(self, 
                          lead_results: Dict[str, LeadAnalysisResult],
                          incomplete: bool = False) -> bool:
        """Check for left bundle branch block pattern."""
        
        # Simplified LBBB detection
        try:
            # Need I, V1, V6 for LBBB diagnosis
            required_leads = ['I', 'V1', 'V6']
            if not all(lead in lead_results for lead in required_leads):
                return False
            
            # Check QRS width in lateral leads
            lateral_width = 0.0
            count = 0
            for lead in ['I', 'V6']:
                if (lead in lead_results and 
                    lead_results[lead].analysis_confidence > 0.5):
                    width = lead_results[lead].features.morphological.get('qrs_width_mean', 0.0) * 1000
                    lateral_width += width
                    count += 1
            
            if count > 0:
                avg_lateral_width = lateral_width / count
                threshold = 100 if incomplete else 120  # ms
                return avg_lateral_width >= threshold
            
            return False
            
        except Exception:
            return False
    
    def _analyze_regional_ischemia(self, 
                                 st_elevation: Dict[str, float],
                                 st_depression: Dict[str, float]) -> Dict[str, any]:
        """Analyze regional ischemia patterns based on lead groups."""
        
        regional_analysis = {}
        
        # Define coronary territories (simplified)
        territories = {
            'anterior': ['V3', 'V4'],
            'lateral': ['I', 'aVL', 'V5', 'V6'],
            'inferior': ['II', 'III', 'aVF'],
            'septal': ['V1', 'V2']
        }
        
        for territory, leads in territories.items():
            # Count ST changes in this territory
            elevation_count = sum(1 for lead in leads if lead in st_elevation)
            depression_count = sum(1 for lead in leads if lead in st_depression)
            
            if elevation_count >= 2:  # At least 2 leads with ST elevation
                regional_analysis[territory] = {
                    'finding': 'st_elevation',
                    'leads_affected': [lead for lead in leads if lead in st_elevation],
                    'severity': 'significant' if elevation_count >= len(leads) / 2 else 'mild'
                }
            elif depression_count >= 2:
                regional_analysis[territory] = {
                    'finding': 'st_depression',
                    'leads_affected': [lead for lead in leads if lead in st_depression],
                    'severity': 'significant' if depression_count >= len(leads) / 2 else 'mild'
                }
        
        return regional_analysis
    
    def _assess_lead_morphology(self, 
                              signal: np.ndarray, 
                              r_peaks: Optional[np.ndarray],
                              lead_name: str) -> float:
        """Assess morphological quality of ECG lead."""
        
        try:
            morphology_score = 0.0
            
            # Basic waveform quality
            if len(signal) > 0:
                # Signal stability
                signal_stability = 1.0 / (1.0 + np.std(signal) / (np.mean(np.abs(signal)) + 1e-10))
                morphology_score += signal_stability * 0.3
                
                # Dynamic range
                signal_range = np.ptp(signal)
                dynamic_range_score = min(1.0, signal_range / (4 * np.std(signal)))
                morphology_score += dynamic_range_score * 0.2
                
                # R-peak quality
                if r_peaks is not None and len(r_peaks) > 0:
                    # Regular R-peak spacing
                    if len(r_peaks) > 1:
                        rr_intervals = np.diff(r_peaks)
                        rr_regularity = 1.0 - (np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-10))
                        morphology_score += rr_regularity * 0.3
                    
                    # R-peak prominence
                    peak_amplitudes = signal[r_peaks]
                    mean_amplitude = np.mean(peak_amplitudes)
                    prominence_score = min(1.0, mean_amplitude / (np.std(signal) + 1e-10))
                    morphology_score += prominence_score * 0.2
                else:
                    # Penalize missing R-peaks
                    morphology_score += 0.25  # Half points for missing peaks
            
            return float(np.clip(morphology_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default score
    
    def _extract_lead_clinical_findings(self, 
                                      signal: np.ndarray,
                                      features: ECGFeatures,
                                      lead_name: str) -> Dict[str, float]:
        """Extract clinical findings specific to this lead."""
        
        findings = {}
        
        try:
            # Heart rate from this lead
            findings['heart_rate'] = features.time_domain.get('mean_hr', 75.0)
            
            # Rhythm regularity
            findings['rhythm_regularity'] = features.clinical.get('rhythm_regularity', 0.5)
            
            # ST segment deviation
            findings['st_deviation'] = features.clinical.get('st_deviation', 0.0)
            
            # QT interval
            findings['qt_interval'] = features.clinical.get('qt_interval', 0.4)
            
            # Lead-specific findings
            if lead_name in ['V1', 'V2']:  # Septal leads
                findings['septal_q_waves'] = self._detect_q_waves(signal)
                findings['r_wave_progression'] = self._assess_r_progression(signal, lead_name)
            elif lead_name in ['II', 'III', 'aVF']:  # Inferior leads
                findings['inferior_changes'] = findings['st_deviation']
            elif lead_name in ['I', 'aVL', 'V5', 'V6']:  # Lateral leads
                findings['lateral_changes'] = findings['st_deviation']
            
        except Exception:
            # Default findings if extraction fails
            findings = {
                'heart_rate': 75.0,
                'rhythm_regularity': 0.5,
                'st_deviation': 0.0,
                'qt_interval': 0.4
            }
        
        return findings
    
    def _detect_lead_artifacts(self, signal: np.ndarray, lead_name: str) -> List[str]:
        """Detect artifacts in ECG lead."""
        
        artifacts = []
        
        try:
            # Baseline wander
            if self._detect_baseline_wander(signal):
                artifacts.append('baseline_wander')
            
            # Muscle artifact
            if self._detect_muscle_artifact(signal):
                artifacts.append('muscle_artifact')
            
            # Electrical interference
            if self._detect_electrical_interference(signal):
                artifacts.append('electrical_interference')
            
            # Lead disconnection
            if self._detect_lead_disconnection(signal):
                artifacts.append('lead_disconnection')
            
        except Exception:
            pass  # Ignore artifact detection errors
        
        return artifacts
    
    def _detect_baseline_wander(self, signal: np.ndarray) -> bool:
        """Detect baseline wander artifact."""
        
        # Low frequency baseline variation
        try:
            # Detrend signal and check for low-frequency components
            detrended = scipy.signal.detrend(signal)
            baseline_variation = np.std(signal - detrended)
            signal_std = np.std(signal)
            
            return baseline_variation > 0.2 * signal_std
        except:
            return False
    
    def _detect_muscle_artifact(self, signal: np.ndarray) -> bool:
        """Detect muscle artifact."""
        
        try:
            # High frequency noise characteristic of muscle artifact
            high_freq = scipy.signal.butter(4, 20/(self.sampling_rate/2), btype='high')
            filtered = scipy.signal.filtfilt(high_freq[0], high_freq[1], signal)
            
            noise_power = np.var(filtered)
            signal_power = np.var(signal)
            
            return noise_power > 0.1 * signal_power
        except:
            return False
    
    def _detect_electrical_interference(self, signal: np.ndarray) -> bool:
        """Detect electrical interference (50/60 Hz)."""
        
        try:
            # Check for 50/60 Hz interference
            freqs, psd = scipy.signal.periodogram(signal, fs=self.sampling_rate)
            
            # Look for peaks at 50 or 60 Hz
            freq_50 = np.argmin(np.abs(freqs - 50))
            freq_60 = np.argmin(np.abs(freqs - 60))
            
            peak_50 = psd[freq_50] if freq_50 < len(psd) else 0
            peak_60 = psd[freq_60] if freq_60 < len(psd) else 0
            
            mean_psd = np.mean(psd)
            
            return (peak_50 > 5 * mean_psd) or (peak_60 > 5 * mean_psd)
        except:
            return False
    
    def _detect_lead_disconnection(self, signal: np.ndarray) -> bool:
        """Detect lead disconnection."""
        
        try:
            # Very low amplitude or constant signal suggests disconnection
            signal_range = np.ptp(signal)
            signal_std = np.std(signal)
            
            return (signal_range < 0.01) or (signal_std < 0.005)
        except:
            return False
    
    def _detect_q_waves(self, signal: np.ndarray) -> float:
        """Detect pathological Q waves (simplified)."""
        
        try:
            # Look for negative deflections at beginning of QRS
            # This is a simplified implementation
            negative_peaks, _ = scipy.signal.find_peaks(-signal, height=0.1*np.std(signal))
            
            if len(negative_peaks) > 0:
                return float(len(negative_peaks) / (len(signal) / self.sampling_rate / 60))  # Q waves per minute
            else:
                return 0.0
        except:
            return 0.0
    
    def _assess_r_progression(self, signal: np.ndarray, lead_name: str) -> float:
        """Assess R wave progression in precordial leads."""
        
        try:
            # Simplified R wave progression assessment
            positive_peaks, _ = scipy.signal.find_peaks(signal, height=0.1*np.std(signal))
            
            if len(positive_peaks) > 0:
                mean_r_amplitude = np.mean(signal[positive_peaks])
                return float(mean_r_amplitude / (np.std(signal) + 1e-10))
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_lead_confidence(self, 
                                 signal_quality: float,
                                 morphology_score: float,
                                 num_artifacts: int) -> float:
        """Calculate overall confidence for lead analysis."""
        
        # Combine quality metrics
        confidence = (signal_quality * 0.4 + 
                     morphology_score * 0.4 + 
                     max(0.0, (1.0 - num_artifacts * 0.1)) * 0.2)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _create_failed_lead_result(self, lead_name: str, error_msg: str) -> LeadAnalysisResult:
        """Create failed lead analysis result."""
        
        from .feature_extraction import ECGFeatures
        
        empty_features = ECGFeatures(
            time_domain={}, frequency_domain={}, morphological={},
            statistical={}, clinical={}, wavelet={},
            metadata={'lead_name': lead_name, 'error': error_msg}
        )
        
        return LeadAnalysisResult(
            lead_name=lead_name,
            signal_quality=0.0,
            features=empty_features,
            morphology_score=0.0,
            clinical_findings={},
            artifacts_detected=[],
            analysis_confidence=0.0
        )
    
    def _create_failed_analysis_result(self, error_msg: str) -> MultiLeadAnalysisResult:
        """Create failed multi-lead analysis result."""
        
        return MultiLeadAnalysisResult(
            lead_results={},
            cross_lead_correlations={},
            cardiac_axis=CardiacAxis.INDETERMINATE,
            axis_degrees=0.0,
            bundle_branch_analysis={'error': error_msg},
            ischemia_analysis={'error': error_msg},
            overall_rhythm_assessment={'error': error_msg},
            lead_concordance=0.0,
            clinical_summary={'error': error_msg},
            analysis_metadata={'error': error_msg, 'analysis_successful': False}
        )
    
    def _update_analysis_stats(self, result: MultiLeadAnalysisResult, success: bool):
        """Update analysis statistics."""
        
        if not hasattr(self, 'analysis_stats'):
            self.analysis_stats = {
                'analyses_performed': 0,
                'successful_analyses': 0,
                'failed_analyses': 0,
                'total_leads_processed': 0
            }
        
        self.analysis_stats['analyses_performed'] += 1
        
        if success:
            self.analysis_stats['successful_analyses'] += 1
            self.analysis_stats['total_leads_processed'] += len(result.lead_results)
        else:
            self.analysis_stats['failed_analyses'] += 1
    
    def get_analysis_stats(self) -> Dict:
        """Get analysis statistics."""
        if not hasattr(self, 'analysis_stats'):
            return {}
        
        stats = self.analysis_stats.copy()
        if stats['analyses_performed'] > 0:
            stats['success_rate'] = stats['successful_analyses'] / stats['analyses_performed']
            if stats['successful_analyses'] > 0:
                stats['avg_leads_per_analysis'] = (stats['total_leads_processed'] / 
                                                 stats['successful_analyses'])
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("Multi-Lead ECG Analysis Module Test")
    print("=" * 50)
    
    # Import signal generation
    from signal_preprocessing import create_test_ecg_signal
    
    # Generate 12-lead ECG test data
    standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    test_ecg_data = create_test_ecg_signal(
        duration=10.0, 
        sampling_rate=500.0, 
        leads=standard_leads
    )
    
    print(f"Generated 12-lead ECG test data:")
    for lead, signal in test_ecg_data.items():
        print(f"  {lead}: {len(signal)} samples")
    
    # Initialize multi-lead analyzer
    analyzer = MultiLeadAnalyzer(sampling_rate=500.0)
    
    # Perform comprehensive analysis
    print("\nPerforming multi-lead ECG analysis...")
    analysis_result = analyzer.analyze_multi_lead_ecg(test_ecg_data, patient_id="test_001")
    
    # Display results
    print(f"\nAnalysis Results:")
    print(f"  Total leads analyzed: {analysis_result.analysis_metadata['total_leads_analyzed']}")
    print(f"  Successful analyses: {analysis_result.analysis_metadata['successful_analyses']}")
    print(f"  Lead concordance: {analysis_result.lead_concordance:.3f}")
    
    # Cardiac axis
    print(f"\nCardiac Assessment:")
    print(f"  Cardiac axis: {analysis_result.cardiac_axis.value} ({analysis_result.axis_degrees:.1f}°)")
    print(f"  Primary rhythm: {analysis_result.overall_rhythm_assessment['primary_rhythm']}")
    print(f"  Average heart rate: {analysis_result.overall_rhythm_assessment['average_heart_rate']:.1f} BPM")
    
    # Clinical summary
    print(f"\nClinical Summary:")
    print(f"  Interpretation: {analysis_result.clinical_summary['overall_interpretation']}")
    print(f"  Risk assessment: {analysis_result.clinical_summary['risk_assessment']}")
    print(f"  Abnormal findings: {len(analysis_result.clinical_summary['abnormal_findings'])}")
    
    if analysis_result.clinical_summary['abnormal_findings']:
        for finding in analysis_result.clinical_summary['abnormal_findings']:
            print(f"    - {finding}")
    
    # Cross-lead correlations
    print(f"\nCross-Lead Analysis:")
    correlations = list(analysis_result.cross_lead_correlations.values())
    if correlations:
        print(f"  Mean correlation: {np.mean(correlations):.3f}")
        print(f"  Min correlation: {np.min(correlations):.3f}")
        print(f"  Max correlation: {np.max(correlations):.3f}")
    
    # Analysis statistics
    stats = analyzer.get_analysis_stats()
    print(f"\nAnalyzer Statistics:")
    print(f"  Analyses performed: {stats.get('analyses_performed', 0)}")
    print(f"  Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    print(f"  Avg leads per analysis: {stats.get('avg_leads_per_analysis', 0):.1f}")
    
    print("\nMulti-Lead ECG Analysis Module Test Complete!")