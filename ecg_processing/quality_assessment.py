#!/usr/bin/env python3
"""
ECG Signal Quality Assessment Module
===================================

Comprehensive ECG signal quality validation for CoT-RAG Stage 4.
Provides automated quality assessment, artifact detection, and
clinical validity scoring for ECG signals before AI model processing.

Features:
- Multi-dimensional quality scoring
- Artifact detection and classification
- Clinical validity assessment
- Real-time quality monitoring
- Quality-based processing recommendations
- Regulatory compliance validation

Quality Metrics:
- Signal-to-noise ratio (SNR)
- Baseline stability
- Artifact detection (muscle, electrical, motion)
- Lead contact quality
- Morphological consistency
- Temporal stability
- Clinical validity score

Clinical Applications:
- Pre-processing quality gates
- Real-time monitoring alerts
- Automated quality reporting
- Model confidence adjustment
- Clinical decision support validation
"""

import numpy as np
import scipy.signal
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings

class QualityLevel(Enum):
    """ECG signal quality classification levels."""
    EXCELLENT = "excellent"     # >0.9
    GOOD = "good"              # 0.7-0.9
    ACCEPTABLE = "acceptable"   # 0.5-0.7
    POOR = "poor"              # 0.3-0.5
    UNACCEPTABLE = "unacceptable"  # <0.3

class ArtifactType(Enum):
    """Types of ECG artifacts that can be detected."""
    BASELINE_WANDER = "baseline_wander"
    MUSCLE_ARTIFACT = "muscle_artifact"
    ELECTRICAL_INTERFERENCE = "electrical_interference"
    MOTION_ARTIFACT = "motion_artifact"
    LEAD_DISCONNECTION = "lead_disconnection"
    SATURATION = "saturation"
    HIGH_FREQUENCY_NOISE = "high_frequency_noise"
    POWERLINE_INTERFERENCE = "powerline_interference"

@dataclass
class ArtifactDetection:
    """Results from artifact detection analysis."""
    artifact_type: ArtifactType
    severity: float  # 0.0 (none) to 1.0 (severe)
    confidence: float  # Detection confidence
    location_indices: Optional[np.ndarray] = None  # Sample indices where artifact occurs
    description: str = ""
    recommendation: str = ""

@dataclass
class QualityMetrics:
    """Comprehensive ECG signal quality metrics."""
    signal_to_noise_ratio: float
    baseline_stability: float
    morphological_consistency: float
    temporal_stability: float
    dynamic_range_score: float
    lead_contact_quality: float
    spectral_purity: float
    clinical_validity: float
    overall_quality_score: float
    quality_level: QualityLevel
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            'signal_to_noise_ratio': self.signal_to_noise_ratio,
            'baseline_stability': self.baseline_stability,
            'morphological_consistency': self.morphological_consistency,
            'temporal_stability': self.temporal_stability,
            'dynamic_range_score': self.dynamic_range_score,
            'lead_contact_quality': self.lead_contact_quality,
            'spectral_purity': self.spectral_purity,
            'clinical_validity': self.clinical_validity,
            'overall_quality_score': self.overall_quality_score,
            'quality_level': self.quality_level.value
        }

@dataclass
class QualityAssessmentResult:
    """Complete quality assessment result for ECG signal."""
    lead_name: str
    signal_length: int
    sampling_rate: float
    duration_seconds: float
    quality_metrics: QualityMetrics
    artifacts_detected: List[ArtifactDetection]
    processing_recommendations: List[str]
    clinical_warnings: List[str]
    usable_for_diagnosis: bool
    confidence_adjustment: float  # Factor to adjust AI model confidence
    timestamp: str
    
    def get_quality_report(self) -> str:
        """Generate human-readable quality report."""
        report = f"""
ECG Signal Quality Assessment Report
===================================

Lead: {self.lead_name}
Duration: {self.duration_seconds:.1f} seconds ({self.signal_length} samples)
Sampling Rate: {self.sampling_rate} Hz

Overall Quality: {self.quality_metrics.quality_level.value.upper()} 
                ({self.quality_metrics.overall_quality_score:.3f})

Quality Metrics:
- Signal-to-Noise Ratio: {self.quality_metrics.signal_to_noise_ratio:.2f} dB
- Baseline Stability: {self.quality_metrics.baseline_stability:.3f}
- Morphological Consistency: {self.quality_metrics.morphological_consistency:.3f}
- Temporal Stability: {self.quality_metrics.temporal_stability:.3f}
- Clinical Validity: {self.quality_metrics.clinical_validity:.3f}

Artifacts Detected: {len(self.artifacts_detected)}
"""
        
        if self.artifacts_detected:
            report += "\nArtifact Details:\n"
            for artifact in self.artifacts_detected:
                report += f"- {artifact.artifact_type.value}: Severity {artifact.severity:.2f}, "
                report += f"Confidence {artifact.confidence:.2f}\n"
        
        if self.processing_recommendations:
            report += "\nProcessing Recommendations:\n"
            for rec in self.processing_recommendations:
                report += f"- {rec}\n"
        
        if self.clinical_warnings:
            report += "\nClinical Warnings:\n"
            for warning in self.clinical_warnings:
                report += f"- {warning}\n"
        
        report += f"\nUsable for Clinical Diagnosis: {'Yes' if self.usable_for_diagnosis else 'No'}\n"
        report += f"AI Model Confidence Adjustment: {self.confidence_adjustment:.3f}\n"
        
        return report

class SignalQualityAssessor:
    """
    Comprehensive ECG signal quality assessment system.
    
    Provides multi-dimensional quality analysis with clinical validation
    and automated recommendations for ECG signal processing.
    """
    
    def __init__(self, 
                 sampling_rate: float = 500.0,
                 clinical_validation: bool = True):
        """
        Initialize signal quality assessor.
        
        Args:
            sampling_rate: ECG sampling rate (Hz)
            clinical_validation: Enable clinical validity checks
        """
        self.sampling_rate = sampling_rate
        self.clinical_validation = clinical_validation
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5,
            'poor': 0.3
        }
        
        # Artifact detection parameters
        self.artifact_params = {
            'baseline_wander_freq': 0.5,  # Hz
            'muscle_artifact_freq': 20.0,  # Hz
            'powerline_freq': [50.0, 60.0],  # Hz
            'noise_threshold': 0.1,
            'saturation_threshold': 0.95
        }
        
        # Assessment statistics
        self.assessment_stats = {}
    
    def assess_signal_quality(self, 
                            ecg_signal: np.ndarray,
                            lead_name: str = "unknown",
                            r_peaks: Optional[np.ndarray] = None) -> QualityAssessmentResult:
        """
        Perform comprehensive signal quality assessment.
        
        Args:
            ecg_signal: ECG signal array
            lead_name: ECG lead identifier
            r_peaks: Optional R-peak locations for enhanced analysis
            
        Returns:
            QualityAssessmentResult with complete assessment
        """
        try:
            # Basic signal validation
            if len(ecg_signal) == 0:
                return self._create_invalid_result(lead_name, "Empty signal")
            
            if np.all(np.isnan(ecg_signal)) or np.all(np.isinf(ecg_signal)):
                return self._create_invalid_result(lead_name, "Invalid signal values")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(ecg_signal, r_peaks)
            
            # Detect artifacts
            artifacts_detected = self._detect_artifacts(ecg_signal)
            
            # Generate processing recommendations
            recommendations = self._generate_recommendations(quality_metrics, artifacts_detected)
            
            # Generate clinical warnings
            clinical_warnings = self._generate_clinical_warnings(quality_metrics, artifacts_detected)
            
            # Determine clinical usability
            usable_for_diagnosis = self._assess_clinical_usability(quality_metrics, artifacts_detected)
            
            # Calculate confidence adjustment factor
            confidence_adjustment = self._calculate_confidence_adjustment(quality_metrics, artifacts_detected)
            
            # Create assessment result
            result = QualityAssessmentResult(
                lead_name=lead_name,
                signal_length=len(ecg_signal),
                sampling_rate=self.sampling_rate,
                duration_seconds=len(ecg_signal) / self.sampling_rate,
                quality_metrics=quality_metrics,
                artifacts_detected=artifacts_detected,
                processing_recommendations=recommendations,
                clinical_warnings=clinical_warnings,
                usable_for_diagnosis=usable_for_diagnosis,
                confidence_adjustment=confidence_adjustment,
                timestamp="current_time"  # Would use datetime in production
            )
            
            self._update_assessment_stats(result, success=True)
            return result
            
        except Exception as e:
            warnings.warn(f"Quality assessment failed for {lead_name}: {e}")
            failed_result = self._create_invalid_result(lead_name, str(e))
            self._update_assessment_stats(failed_result, success=False)
            return failed_result
    
    def _calculate_quality_metrics(self, 
                                 signal: np.ndarray, 
                                 r_peaks: Optional[np.ndarray]) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        
        # 1. Signal-to-Noise Ratio
        snr_db = self._calculate_snr(signal)
        
        # 2. Baseline Stability
        baseline_stability = self._assess_baseline_stability(signal)
        
        # 3. Morphological Consistency
        morphological_consistency = self._assess_morphological_consistency(signal, r_peaks)
        
        # 4. Temporal Stability
        temporal_stability = self._assess_temporal_stability(signal)
        
        # 5. Dynamic Range Score
        dynamic_range_score = self._assess_dynamic_range(signal)
        
        # 6. Lead Contact Quality
        lead_contact_quality = self._assess_lead_contact_quality(signal)
        
        # 7. Spectral Purity
        spectral_purity = self._assess_spectral_purity(signal)
        
        # 8. Clinical Validity
        clinical_validity = self._assess_clinical_validity(signal, r_peaks)
        
        # 9. Overall Quality Score (weighted combination)
        overall_quality = self._calculate_overall_quality(
            snr_db, baseline_stability, morphological_consistency, 
            temporal_stability, dynamic_range_score, lead_contact_quality,
            spectral_purity, clinical_validity
        )
        
        # 10. Quality Level Classification
        quality_level = self._classify_quality_level(overall_quality)
        
        return QualityMetrics(
            signal_to_noise_ratio=snr_db,
            baseline_stability=baseline_stability,
            morphological_consistency=morphological_consistency,
            temporal_stability=temporal_stability,
            dynamic_range_score=dynamic_range_score,
            lead_contact_quality=lead_contact_quality,
            spectral_purity=spectral_purity,
            clinical_validity=clinical_validity,
            overall_quality_score=overall_quality,
            quality_level=quality_level
        )
    
    def _calculate_snr(self, signal: np.ndarray) -> float:
        """Calculate signal-to-noise ratio in dB."""
        try:
            # Separate signal and noise using filtering approach
            # ECG signal is primarily in 0.5-40 Hz range
            
            # Design bandpass filter for ECG signal
            nyquist = self.sampling_rate / 2
            low_cut = 0.5 / nyquist
            high_cut = 40.0 / nyquist
            
            # Ensure valid cutoff frequencies
            low_cut = max(0.01, min(low_cut, 0.99))
            high_cut = max(low_cut + 0.01, min(high_cut, 0.99))
            
            b, a = scipy.signal.butter(4, [low_cut, high_cut], btype='band')
            signal_filtered = scipy.signal.filtfilt(b, a, signal)
            
            # Estimate noise as high-frequency components
            high_freq_cut = 40.0 / nyquist
            if high_freq_cut < 0.99:
                b_noise, a_noise = scipy.signal.butter(4, high_freq_cut, btype='high')
                noise = scipy.signal.filtfilt(b_noise, a_noise, signal)
            else:
                # Use signal difference as noise proxy
                noise = signal - signal_filtered
            
            # Calculate SNR
            signal_power = np.var(signal_filtered)
            noise_power = np.var(noise)
            
            if noise_power > 0:
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(snr_linear)
            else:
                snr_db = 60.0  # Very high SNR when no noise detected
            
            # Clip to reasonable range
            return float(np.clip(snr_db, 0.0, 60.0))
            
        except Exception:
            return 20.0  # Default moderate SNR
    
    def _assess_baseline_stability(self, signal: np.ndarray) -> float:
        """Assess baseline stability (0.0 = unstable, 1.0 = stable)."""
        try:
            # Remove baseline trend
            detrended = scipy.signal.detrend(signal)
            
            # Calculate baseline drift as low-frequency component
            if len(signal) > 100:
                # Use polynomial detrending for baseline estimation
                baseline_polynomial = np.polyfit(np.arange(len(signal)), signal, deg=3)
                baseline_trend = np.polyval(baseline_polynomial, np.arange(len(signal)))
                
                # Baseline variation relative to signal
                baseline_variation = np.std(signal - baseline_trend)
                signal_amplitude = np.std(signal)
                
                if signal_amplitude > 0:
                    stability_score = 1.0 / (1.0 + baseline_variation / signal_amplitude)
                else:
                    stability_score = 0.0
            else:
                # Short signal - use simple detrending
                baseline_variation = np.std(signal - detrended)
                signal_std = np.std(signal)
                stability_score = 1.0 / (1.0 + baseline_variation / (signal_std + 1e-10))
            
            return float(np.clip(stability_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default moderate stability
    
    def _assess_morphological_consistency(self, 
                                        signal: np.ndarray, 
                                        r_peaks: Optional[np.ndarray]) -> float:
        """Assess consistency of ECG morphology across beats."""
        try:
            if r_peaks is None or len(r_peaks) < 3:
                # Without R-peaks, use correlation-based approach
                return self._assess_morphology_without_peaks(signal)
            
            # Extract beats around R-peaks
            beat_window = int(0.6 * self.sampling_rate)  # 600ms window
            beats = []
            
            for r_peak in r_peaks:
                start = max(0, r_peak - beat_window // 2)
                end = min(len(signal), r_peak + beat_window // 2)
                
                if end - start > beat_window // 2:  # Ensure minimum beat length
                    beat = signal[start:end]
                    # Normalize beat length
                    if len(beat) != beat_window:
                        # Interpolate to standard length
                        beat_interp = np.interp(
                            np.linspace(0, len(beat)-1, beat_window),
                            np.arange(len(beat)),
                            beat
                        )
                        beats.append(beat_interp)
                    else:
                        beats.append(beat)
            
            if len(beats) < 2:
                return 0.5  # Not enough beats for comparison
            
            # Calculate correlation between beats
            beats_array = np.array(beats)
            correlations = []
            
            for i in range(len(beats)):
                for j in range(i+1, len(beats)):
                    corr = np.corrcoef(beats[i], beats[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                consistency_score = np.mean(correlations)
            else:
                consistency_score = 0.5
            
            return float(np.clip(consistency_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default moderate consistency
    
    def _assess_morphology_without_peaks(self, signal: np.ndarray) -> float:
        """Assess morphological consistency without R-peak detection."""
        try:
            # Divide signal into segments and assess similarity
            segment_length = int(2.0 * self.sampling_rate)  # 2-second segments
            
            if len(signal) < 2 * segment_length:
                return 0.5  # Signal too short
            
            segments = []
            for i in range(0, len(signal) - segment_length, segment_length // 2):
                segments.append(signal[i:i + segment_length])
            
            if len(segments) < 2:
                return 0.5
            
            # Calculate cross-correlations between segments
            correlations = []
            for i in range(len(segments)):
                for j in range(i+1, len(segments)):
                    corr = np.corrcoef(segments[i], segments[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                return float(np.mean(correlations))
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _assess_temporal_stability(self, signal: np.ndarray) -> float:
        """Assess temporal stability of signal characteristics."""
        try:
            # Divide signal into temporal windows
            window_duration = 2.0  # seconds
            window_samples = int(window_duration * self.sampling_rate)
            
            if len(signal) < 2 * window_samples:
                return 0.5  # Signal too short for temporal analysis
            
            # Calculate statistical measures in each window
            window_stats = []
            for i in range(0, len(signal) - window_samples, window_samples):
                window = signal[i:i + window_samples]
                stats_dict = {
                    'mean': np.mean(window),
                    'std': np.std(window),
                    'rms': np.sqrt(np.mean(window**2)),
                    'energy': np.sum(window**2)
                }
                window_stats.append(stats_dict)
            
            if len(window_stats) < 2:
                return 0.5
            
            # Assess stability as inverse of coefficient of variation
            stability_scores = []
            for metric in ['mean', 'std', 'rms', 'energy']:
                values = [ws[metric] for ws in window_stats]
                if np.mean(values) > 0:
                    cv = np.std(values) / np.mean(values)
                    stability = 1.0 / (1.0 + cv)
                    stability_scores.append(stability)
            
            if stability_scores:
                temporal_stability = np.mean(stability_scores)
            else:
                temporal_stability = 0.5
            
            return float(np.clip(temporal_stability, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _assess_dynamic_range(self, signal: np.ndarray) -> float:
        """Assess signal dynamic range utilization."""
        try:
            # Calculate signal statistics
            signal_min = np.min(signal)
            signal_max = np.max(signal)
            signal_range = signal_max - signal_min
            signal_std = np.std(signal)
            
            # Assess range utilization
            if signal_std > 0:
                # Good dynamic range when peak-to-peak is several times std dev
                range_ratio = signal_range / (6 * signal_std)  # Expect ~6 std dev range
                range_score = min(1.0, range_ratio)
            else:
                range_score = 0.0  # No dynamic range
            
            # Penalize saturation (values at extremes)
            saturation_threshold = 0.95
            extreme_values = np.sum((np.abs(signal) > saturation_threshold * np.max(np.abs(signal))))
            saturation_penalty = min(1.0, extreme_values / len(signal))
            
            dynamic_range_score = range_score * (1.0 - saturation_penalty)
            
            return float(np.clip(dynamic_range_score, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _assess_lead_contact_quality(self, signal: np.ndarray) -> float:
        """Assess ECG lead contact quality."""
        try:
            # Multiple indicators of poor lead contact
            contact_score = 1.0
            
            # 1. Very low amplitude (poor contact)
            signal_amplitude = np.ptp(signal)  # Peak-to-peak amplitude
            if signal_amplitude < 0.01:  # Very small amplitude
                contact_score *= 0.1
            elif signal_amplitude < 0.1:
                contact_score *= 0.5
            
            # 2. Excessive noise relative to signal
            signal_rms = np.sqrt(np.mean(signal**2))
            # High-frequency noise as proxy for poor contact
            high_freq_b, high_freq_a = scipy.signal.butter(4, 40/(self.sampling_rate/2), btype='high')
            noise = scipy.signal.filtfilt(high_freq_b, high_freq_a, signal)
            noise_rms = np.sqrt(np.mean(noise**2))
            
            if signal_rms > 0:
                noise_ratio = noise_rms / signal_rms
                if noise_ratio > 0.5:  # High noise
                    contact_score *= (1.0 - noise_ratio)
            
            # 3. Sudden amplitude changes (intermittent contact)
            amplitude_changes = np.abs(np.diff(signal))
            extreme_changes = np.sum(amplitude_changes > 5 * np.std(amplitude_changes))
            change_rate = extreme_changes / len(signal)
            contact_score *= (1.0 - min(1.0, change_rate * 10))
            
            return float(np.clip(contact_score, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _assess_spectral_purity(self, signal: np.ndarray) -> float:
        """Assess spectral purity (freedom from interference)."""
        try:
            # Compute power spectral density
            freqs, psd = scipy.signal.periodogram(signal, fs=self.sampling_rate)
            
            # ECG signal should have most energy in 0.5-40 Hz range
            ecg_band_mask = (freqs >= 0.5) & (freqs <= 40.0)
            interference_band_mask = (freqs > 40.0) | (freqs < 0.5)
            
            # Calculate power in each band
            ecg_power = np.sum(psd[ecg_band_mask]) if np.any(ecg_band_mask) else 0
            interference_power = np.sum(psd[interference_band_mask]) if np.any(interference_band_mask) else 0
            total_power = np.sum(psd)
            
            # Check for powerline interference (50/60 Hz)
            powerline_interference = 0.0
            for freq in [50.0, 60.0]:
                if freq <= self.sampling_rate / 2:
                    freq_idx = np.argmin(np.abs(freqs - freq))
                    if freq_idx < len(psd):
                        # Check if power at this frequency is anomalously high
                        local_mean = np.mean(psd[max(0, freq_idx-5):min(len(psd), freq_idx+5)])
                        if psd[freq_idx] > 5 * local_mean:
                            powerline_interference += psd[freq_idx] / total_power
            
            # Calculate spectral purity score
            if total_power > 0:
                signal_purity = ecg_power / total_power
                interference_ratio = interference_power / total_power
                purity_score = signal_purity * (1.0 - interference_ratio - powerline_interference)
            else:
                purity_score = 0.0
            
            return float(np.clip(purity_score, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _assess_clinical_validity(self, 
                                signal: np.ndarray, 
                                r_peaks: Optional[np.ndarray]) -> float:
        """Assess clinical validity of ECG signal."""
        if not self.clinical_validation:
            return 1.0  # Skip clinical validation if disabled
        
        try:
            validity_score = 1.0
            
            # 1. Physiological heart rate range
            if r_peaks is not None and len(r_peaks) > 1:
                rr_intervals = np.diff(r_peaks) / self.sampling_rate
                heart_rates = 60 / rr_intervals
                mean_hr = np.mean(heart_rates)
                
                # Check for physiological range (30-200 BPM for wide clinical range)
                if mean_hr < 30 or mean_hr > 200:
                    validity_score *= 0.3  # Severely abnormal
                elif mean_hr < 40 or mean_hr > 150:
                    validity_score *= 0.7  # Moderately abnormal
            
            # 2. Signal amplitude in physiological range
            signal_amplitude = np.ptp(signal)
            # Typical ECG amplitude range: 0.1-5 mV (normalized units)
            if signal_amplitude < 0.01:  # Too small
                validity_score *= 0.2
            elif signal_amplitude > 10:  # Too large
                validity_score *= 0.5
            
            # 3. Morphological plausibility
            # Check for reasonable ECG-like characteristics
            
            # Signal should have both positive and negative deflections
            has_positive = np.any(signal > np.mean(signal) + np.std(signal))
            has_negative = np.any(signal < np.mean(signal) - np.std(signal))
            
            if not (has_positive and has_negative):
                validity_score *= 0.5  # Unusual for typical ECG
            
            # 4. Frequency content validation
            # ECG signal should have minimal very high frequency content
            freqs, psd = scipy.signal.periodogram(signal, fs=self.sampling_rate)
            high_freq_mask = freqs > 100  # Above typical ECG content
            high_freq_power = np.sum(psd[high_freq_mask]) if np.any(high_freq_mask) else 0
            total_power = np.sum(psd)
            
            if total_power > 0 and high_freq_power / total_power > 0.3:
                validity_score *= 0.6  # Unusual high-frequency content
            
            return float(np.clip(validity_score, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _calculate_overall_quality(self, *quality_components) -> float:
        """Calculate weighted overall quality score."""
        
        # Weights for different quality components
        weights = [0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10]  # Sum = 1.0
        
        if len(quality_components) != len(weights):
            warnings.warn(f"Quality component count mismatch: {len(quality_components)} vs {len(weights)}")
            # Use equal weights if mismatch
            weights = [1.0 / len(quality_components)] * len(quality_components)
        
        # Calculate weighted sum
        overall_quality = sum(w * q for w, q in zip(weights, quality_components))
        
        return float(np.clip(overall_quality, 0.0, 1.0))
    
    def _classify_quality_level(self, overall_quality: float) -> QualityLevel:
        """Classify quality level based on overall score."""
        
        if overall_quality >= self.quality_thresholds['excellent']:
            return QualityLevel.EXCELLENT
        elif overall_quality >= self.quality_thresholds['good']:
            return QualityLevel.GOOD
        elif overall_quality >= self.quality_thresholds['acceptable']:
            return QualityLevel.ACCEPTABLE
        elif overall_quality >= self.quality_thresholds['poor']:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE
    
    def _detect_artifacts(self, signal: np.ndarray) -> List[ArtifactDetection]:
        """Detect various types of artifacts in ECG signal."""
        
        artifacts = []
        
        # 1. Baseline Wander
        baseline_artifact = self._detect_baseline_wander_artifact(signal)
        if baseline_artifact:
            artifacts.append(baseline_artifact)
        
        # 2. Muscle Artifact
        muscle_artifact = self._detect_muscle_artifact(signal)
        if muscle_artifact:
            artifacts.append(muscle_artifact)
        
        # 3. Electrical Interference
        electrical_artifact = self._detect_electrical_interference(signal)
        if electrical_artifact:
            artifacts.append(electrical_artifact)
        
        # 4. Motion Artifact
        motion_artifact = self._detect_motion_artifact(signal)
        if motion_artifact:
            artifacts.append(motion_artifact)
        
        # 5. Lead Disconnection
        disconnection_artifact = self._detect_lead_disconnection_artifact(signal)
        if disconnection_artifact:
            artifacts.append(disconnection_artifact)
        
        # 6. Saturation
        saturation_artifact = self._detect_saturation_artifact(signal)
        if saturation_artifact:
            artifacts.append(saturation_artifact)
        
        # 7. High Frequency Noise
        hf_noise_artifact = self._detect_high_frequency_noise(signal)
        if hf_noise_artifact:
            artifacts.append(hf_noise_artifact)
        
        # 8. Powerline Interference
        powerline_artifact = self._detect_powerline_interference(signal)
        if powerline_artifact:
            artifacts.append(powerline_artifact)
        
        return artifacts
    
    def _detect_baseline_wander_artifact(self, signal: np.ndarray) -> Optional[ArtifactDetection]:
        """Detect baseline wander artifact."""
        try:
            # Low-pass filter to extract baseline
            low_cutoff = 0.5 / (self.sampling_rate / 2)
            if low_cutoff >= 1.0:
                return None
            
            b, a = scipy.signal.butter(4, low_cutoff, btype='low')
            baseline = scipy.signal.filtfilt(b, a, signal)
            
            # Assess baseline variation
            baseline_variation = np.std(baseline)
            signal_amplitude = np.std(signal)
            
            if signal_amplitude > 0:
                wander_severity = baseline_variation / signal_amplitude
                
                if wander_severity > 0.1:  # Threshold for significant baseline wander
                    return ArtifactDetection(
                        artifact_type=ArtifactType.BASELINE_WANDER,
                        severity=min(1.0, wander_severity * 2),
                        confidence=0.8,
                        description=f"Baseline wander detected (severity: {wander_severity:.2f})",
                        recommendation="Apply high-pass filtering to remove baseline drift"
                    )
            
        except Exception:
            pass
        
        return None
    
    def _detect_muscle_artifact(self, signal: np.ndarray) -> Optional[ArtifactDetection]:
        """Detect muscle artifact (EMG interference)."""
        try:
            # High-pass filter for muscle artifact (>20 Hz)
            high_cutoff = 20.0 / (self.sampling_rate / 2)
            if high_cutoff >= 1.0:
                return None
            
            b, a = scipy.signal.butter(4, high_cutoff, btype='high')
            high_freq_content = scipy.signal.filtfilt(b, a, signal)
            
            # Assess high-frequency power
            hf_power = np.var(high_freq_content)
            signal_power = np.var(signal)
            
            if signal_power > 0:
                hf_ratio = hf_power / signal_power
                
                if hf_ratio > 0.15:  # Threshold for muscle artifact
                    return ArtifactDetection(
                        artifact_type=ArtifactType.MUSCLE_ARTIFACT,
                        severity=min(1.0, hf_ratio * 3),
                        confidence=0.7,
                        description=f"Muscle artifact detected (HF ratio: {hf_ratio:.2f})",
                        recommendation="Patient should relax, check electrode placement"
                    )
            
        except Exception:
            pass
        
        return None
    
    def _detect_electrical_interference(self, signal: np.ndarray) -> Optional[ArtifactDetection]:
        """Detect electrical interference."""
        try:
            # Compute power spectrum
            freqs, psd = scipy.signal.periodogram(signal, fs=self.sampling_rate)
            
            # Check for electrical interference patterns
            interference_detected = False
            max_interference = 0.0
            
            # Look for peaks at electrical frequencies and harmonics
            electrical_freqs = [50, 60, 100, 120, 150, 180]  # Common interference frequencies
            
            for freq in electrical_freqs:
                if freq < self.sampling_rate / 2:
                    # Find frequency bin closest to target frequency
                    freq_idx = np.argmin(np.abs(freqs - freq))
                    
                    # Compare power at this frequency to surrounding frequencies
                    if freq_idx > 0 and freq_idx < len(psd) - 1:
                        local_baseline = np.median(psd[max(0, freq_idx-10):min(len(psd), freq_idx+10)])
                        if psd[freq_idx] > 5 * local_baseline:
                            interference_detected = True
                            interference_strength = psd[freq_idx] / np.sum(psd)
                            max_interference = max(max_interference, interference_strength)
            
            if interference_detected:
                return ArtifactDetection(
                    artifact_type=ArtifactType.ELECTRICAL_INTERFERENCE,
                    severity=min(1.0, max_interference * 20),
                    confidence=0.8,
                    description=f"Electrical interference detected (strength: {max_interference:.3f})",
                    recommendation="Check for nearby electrical devices, improve grounding"
                )
        
        except Exception:
            pass
        
        return None
    
    def _detect_motion_artifact(self, signal: np.ndarray) -> Optional[ArtifactDetection]:
        """Detect motion artifacts."""
        try:
            # Motion artifacts typically show as sudden amplitude changes
            signal_diff = np.abs(np.diff(signal))
            diff_threshold = 5 * np.std(signal_diff)
            
            sudden_changes = np.sum(signal_diff > diff_threshold)
            change_rate = sudden_changes / len(signal)
            
            if change_rate > 0.01:  # More than 1% of samples show sudden changes
                # Find locations of motion artifacts
                artifact_locations = np.where(signal_diff > diff_threshold)[0]
                
                return ArtifactDetection(
                    artifact_type=ArtifactType.MOTION_ARTIFACT,
                    severity=min(1.0, change_rate * 50),
                    confidence=0.6,
                    location_indices=artifact_locations,
                    description=f"Motion artifacts detected ({sudden_changes} events)",
                    recommendation="Patient should remain still during recording"
                )
        
        except Exception:
            pass
        
        return None
    
    def _detect_lead_disconnection_artifact(self, signal: np.ndarray) -> Optional[ArtifactDetection]:
        """Detect lead disconnection."""
        try:
            # Lead disconnection indicators:
            # 1. Very low amplitude
            signal_amplitude = np.ptp(signal)
            
            # 2. Near-constant signal
            signal_variation = np.std(signal)
            
            # 3. Unrealistic signal values
            mean_abs_value = np.mean(np.abs(signal))
            
            disconnection_score = 0.0
            
            if signal_amplitude < 0.001:  # Very small amplitude
                disconnection_score += 0.4
            
            if signal_variation < 0.001:  # Very little variation
                disconnection_score += 0.3
            
            if mean_abs_value < 0.001:  # Near-zero signal
                disconnection_score += 0.3
            
            if disconnection_score > 0.5:
                return ArtifactDetection(
                    artifact_type=ArtifactType.LEAD_DISCONNECTION,
                    severity=disconnection_score,
                    confidence=0.9,
                    description="Possible lead disconnection detected",
                    recommendation="Check electrode connections and skin contact"
                )
        
        except Exception:
            pass
        
        return None
    
    def _detect_saturation_artifact(self, signal: np.ndarray) -> Optional[ArtifactDetection]:
        """Detect signal saturation."""
        try:
            # Check for clipping at signal extremes
            signal_range = np.ptp(signal)
            signal_max = np.max(signal)
            signal_min = np.min(signal)
            
            # Count samples at or near extremes
            saturation_threshold = 0.95
            upper_sat = signal_max * saturation_threshold
            lower_sat = signal_min * saturation_threshold
            
            saturated_samples = np.sum((signal >= upper_sat) | (signal <= lower_sat))
            saturation_rate = saturated_samples / len(signal)
            
            if saturation_rate > 0.001:  # More than 0.1% saturated samples
                return ArtifactDetection(
                    artifact_type=ArtifactType.SATURATION,
                    severity=min(1.0, saturation_rate * 100),
                    confidence=0.8,
                    description=f"Signal saturation detected ({saturated_samples} samples)",
                    recommendation="Reduce amplifier gain or check electrode impedance"
                )
        
        except Exception:
            pass
        
        return None
    
    def _detect_high_frequency_noise(self, signal: np.ndarray) -> Optional[ArtifactDetection]:
        """Detect high-frequency noise."""
        try:
            # Filter for high frequencies (>100 Hz)
            high_freq_cutoff = 100.0 / (self.sampling_rate / 2)
            if high_freq_cutoff >= 1.0:
                return None
            
            b, a = scipy.signal.butter(4, high_freq_cutoff, btype='high')
            hf_noise = scipy.signal.filtfilt(b, a, signal)
            
            # Compare HF noise power to signal power
            noise_power = np.var(hf_noise)
            signal_power = np.var(signal)
            
            if signal_power > 0:
                noise_ratio = noise_power / signal_power
                
                if noise_ratio > 0.2:  # High-frequency noise is >20% of signal
                    return ArtifactDetection(
                        artifact_type=ArtifactType.HIGH_FREQUENCY_NOISE,
                        severity=min(1.0, noise_ratio * 2),
                        confidence=0.7,
                        description=f"High-frequency noise detected (ratio: {noise_ratio:.2f})",
                        recommendation="Apply low-pass filtering, check cable shielding"
                    )
        
        except Exception:
            pass
        
        return None
    
    def _detect_powerline_interference(self, signal: np.ndarray) -> Optional[ArtifactDetection]:
        """Detect powerline interference (50/60 Hz)."""
        try:
            # This is a more specific version of electrical interference detection
            freqs, psd = scipy.signal.periodogram(signal, fs=self.sampling_rate)
            
            powerline_interference = 0.0
            detected_frequencies = []
            
            # Check 50 Hz and 60 Hz specifically
            for freq in [50.0, 60.0]:
                if freq < self.sampling_rate / 2:
                    freq_idx = np.argmin(np.abs(freqs - freq))
                    
                    # Local baseline around this frequency
                    local_window = slice(max(0, freq_idx-5), min(len(psd), freq_idx+5))
                    local_baseline = np.median(psd[local_window])
                    
                    if psd[freq_idx] > 10 * local_baseline:  # Strong peak
                        interference_strength = psd[freq_idx] / np.sum(psd)
                        powerline_interference = max(powerline_interference, interference_strength)
                        detected_frequencies.append(freq)
            
            if powerline_interference > 0.01:  # Significant powerline interference
                return ArtifactDetection(
                    artifact_type=ArtifactType.POWERLINE_INTERFERENCE,
                    severity=min(1.0, powerline_interference * 50),
                    confidence=0.9,
                    description=f"Powerline interference at {detected_frequencies} Hz",
                    recommendation="Apply notch filtering, improve electrical isolation"
                )
        
        except Exception:
            pass
        
        return None
    
    def _generate_recommendations(self, 
                                quality_metrics: QualityMetrics,
                                artifacts: List[ArtifactDetection]) -> List[str]:
        """Generate processing recommendations based on quality assessment."""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality_metrics.overall_quality_score < 0.3:
            recommendations.append("Signal quality unacceptable - consider re-acquisition")
        elif quality_metrics.overall_quality_score < 0.5:
            recommendations.append("Poor signal quality - apply preprocessing with caution")
        elif quality_metrics.overall_quality_score < 0.7:
            recommendations.append("Acceptable quality - standard preprocessing recommended")
        
        # SNR-based recommendations
        if quality_metrics.signal_to_noise_ratio < 10:
            recommendations.append("Low SNR - apply noise reduction filtering")
        
        # Baseline stability
        if quality_metrics.baseline_stability < 0.5:
            recommendations.append("Unstable baseline - apply high-pass filtering")
        
        # Lead contact quality
        if quality_metrics.lead_contact_quality < 0.5:
            recommendations.append("Poor lead contact - check electrode placement and impedance")
        
        # Artifact-specific recommendations
        for artifact in artifacts:
            if artifact.recommendation and artifact.recommendation not in recommendations:
                recommendations.append(artifact.recommendation)
        
        # Clinical validity
        if quality_metrics.clinical_validity < 0.5:
            recommendations.append("Questionable clinical validity - expert review recommended")
        
        return recommendations
    
    def _generate_clinical_warnings(self, 
                                  quality_metrics: QualityMetrics,
                                  artifacts: List[ArtifactDetection]) -> List[str]:
        """Generate clinical warnings based on quality assessment."""
        
        warnings = []
        
        # Critical quality issues
        if quality_metrics.overall_quality_score < 0.3:
            warnings.append("CRITICAL: Signal quality insufficient for clinical diagnosis")
        
        # Severe artifacts
        severe_artifacts = [a for a in artifacts if a.severity > 0.7]
        if severe_artifacts:
            warnings.append(f"SEVERE: {len(severe_artifacts)} severe artifacts detected")
        
        # Clinical validity issues
        if quality_metrics.clinical_validity < 0.3:
            warnings.append("WARNING: Signal may not be physiologically valid")
        
        # Lead disconnection
        disconnection_artifacts = [a for a in artifacts if a.artifact_type == ArtifactType.LEAD_DISCONNECTION]
        if disconnection_artifacts:
            warnings.append("WARNING: Possible lead disconnection - verify electrode contact")
        
        # Saturation
        saturation_artifacts = [a for a in artifacts if a.artifact_type == ArtifactType.SATURATION]
        if saturation_artifacts:
            warnings.append("WARNING: Signal saturation detected - may lose diagnostic information")
        
        return warnings
    
    def _assess_clinical_usability(self, 
                                 quality_metrics: QualityMetrics,
                                 artifacts: List[ArtifactDetection]) -> bool:
        """Determine if signal is usable for clinical diagnosis."""
        
        # Minimum quality thresholds for clinical use
        min_overall_quality = 0.4
        min_clinical_validity = 0.3
        max_severe_artifacts = 2
        
        # Check overall quality
        if quality_metrics.overall_quality_score < min_overall_quality:
            return False
        
        # Check clinical validity
        if quality_metrics.clinical_validity < min_clinical_validity:
            return False
        
        # Check for excessive severe artifacts
        severe_artifacts = [a for a in artifacts if a.severity > 0.7]
        if len(severe_artifacts) > max_severe_artifacts:
            return False
        
        # Check for critical artifacts
        critical_artifact_types = [ArtifactType.LEAD_DISCONNECTION, ArtifactType.SATURATION]
        for artifact in artifacts:
            if artifact.artifact_type in critical_artifact_types and artifact.severity > 0.5:
                return False
        
        return True
    
    def _calculate_confidence_adjustment(self, 
                                       quality_metrics: QualityMetrics,
                                       artifacts: List[ArtifactDetection]) -> float:
        """Calculate confidence adjustment factor for AI models."""
        
        # Start with quality-based adjustment
        confidence_factor = quality_metrics.overall_quality_score
        
        # Reduce confidence for artifacts
        for artifact in artifacts:
            severity_penalty = artifact.severity * 0.1  # Max 10% reduction per artifact
            confidence_factor *= (1.0 - severity_penalty)
        
        # Additional penalties for specific issues
        if quality_metrics.clinical_validity < 0.5:
            confidence_factor *= 0.8  # 20% reduction for low clinical validity
        
        if quality_metrics.signal_to_noise_ratio < 10:
            confidence_factor *= 0.9  # 10% reduction for low SNR
        
        return float(np.clip(confidence_factor, 0.0, 1.0))
    
    def _create_invalid_result(self, lead_name: str, error_msg: str) -> QualityAssessmentResult:
        """Create result for invalid/failed assessment."""
        
        # Create minimal quality metrics
        minimal_metrics = QualityMetrics(
            signal_to_noise_ratio=0.0,
            baseline_stability=0.0,
            morphological_consistency=0.0,
            temporal_stability=0.0,
            dynamic_range_score=0.0,
            lead_contact_quality=0.0,
            spectral_purity=0.0,
            clinical_validity=0.0,
            overall_quality_score=0.0,
            quality_level=QualityLevel.UNACCEPTABLE
        )
        
        return QualityAssessmentResult(
            lead_name=lead_name,
            signal_length=0,
            sampling_rate=self.sampling_rate,
            duration_seconds=0.0,
            quality_metrics=minimal_metrics,
            artifacts_detected=[],
            processing_recommendations=[f"Assessment failed: {error_msg}"],
            clinical_warnings=[f"CRITICAL: Quality assessment failed - {error_msg}"],
            usable_for_diagnosis=False,
            confidence_adjustment=0.0,
            timestamp="current_time"
        )
    
    def _update_assessment_stats(self, result: QualityAssessmentResult, success: bool):
        """Update assessment statistics."""
        
        if not hasattr(self, 'assessment_stats'):
            self.assessment_stats = {
                'assessments_performed': 0,
                'successful_assessments': 0,
                'failed_assessments': 0,
                'total_artifacts_detected': 0,
                'clinically_usable_signals': 0
            }
        
        self.assessment_stats['assessments_performed'] += 1
        
        if success:
            self.assessment_stats['successful_assessments'] += 1
            self.assessment_stats['total_artifacts_detected'] += len(result.artifacts_detected)
            
            if result.usable_for_diagnosis:
                self.assessment_stats['clinically_usable_signals'] += 1
        else:
            self.assessment_stats['failed_assessments'] += 1
    
    def get_assessment_stats(self) -> Dict:
        """Get quality assessment statistics."""
        if not hasattr(self, 'assessment_stats'):
            return {}
        
        stats = self.assessment_stats.copy()
        if stats['assessments_performed'] > 0:
            stats['success_rate'] = stats['successful_assessments'] / stats['assessments_performed']
            stats['clinical_usability_rate'] = stats['clinically_usable_signals'] / stats['assessments_performed']
            
            if stats['successful_assessments'] > 0:
                stats['avg_artifacts_per_signal'] = (stats['total_artifacts_detected'] / 
                                                   stats['successful_assessments'])
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("ECG Signal Quality Assessment Module Test")
    print("=" * 50)
    
    # Import signal generation
    from signal_preprocessing import create_test_ecg_signal, ECGSignalProcessor
    
    # Generate test signals with different quality levels
    
    # 1. High quality signal
    clean_ecg = create_test_ecg_signal(duration=10.0, sampling_rate=500.0, leads=["II"])["II"]
    
    # 2. Noisy signal
    noisy_ecg = clean_ecg + 0.1 * np.random.randn(len(clean_ecg))
    
    # 3. Signal with baseline wander
    t = np.linspace(0, len(clean_ecg)/500.0, len(clean_ecg))
    baseline_wander = 0.2 * np.sin(2 * np.pi * 0.1 * t)
    wandering_ecg = clean_ecg + baseline_wander
    
    # 4. Signal with powerline interference
    powerline_interference = 0.05 * np.sin(2 * np.pi * 50 * t)
    interference_ecg = clean_ecg + powerline_interference
    
    # Initialize quality assessor
    assessor = SignalQualityAssessor(sampling_rate=500.0)
    
    # Test different signal qualities
    test_signals = {
        "Clean ECG": clean_ecg,
        "Noisy ECG": noisy_ecg,
        "Baseline Wander": wandering_ecg,
        "Powerline Interference": interference_ecg
    }
    
    print(f"Testing {len(test_signals)} different signal types...\n")
    
    for signal_name, signal in test_signals.items():
        print(f"Assessing: {signal_name}")
        print("-" * 30)
        
        # Perform quality assessment
        assessment = assessor.assess_signal_quality(signal, lead_name=signal_name)
        
        # Display key results
        print(f"Overall Quality: {assessment.quality_metrics.quality_level.value.upper()} "
              f"({assessment.quality_metrics.overall_quality_score:.3f})")
        print(f"SNR: {assessment.quality_metrics.signal_to_noise_ratio:.1f} dB")
        print(f"Artifacts Detected: {len(assessment.artifacts_detected)}")
        
        if assessment.artifacts_detected:
            for artifact in assessment.artifacts_detected:
                print(f"  - {artifact.artifact_type.value}: Severity {artifact.severity:.2f}")
        
        print(f"Clinically Usable: {'Yes' if assessment.usable_for_diagnosis else 'No'}")
        print(f"Confidence Adjustment: {assessment.confidence_adjustment:.3f}")
        print()
    
    # Display assessment statistics
    stats = assessor.get_assessment_stats()
    print("Assessment Statistics:")
    print(f"  Total assessments: {stats.get('assessments_performed', 0)}")
    print(f"  Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    print(f"  Clinical usability rate: {stats.get('clinical_usability_rate', 0)*100:.1f}%")
    print(f"  Average artifacts per signal: {stats.get('avg_artifacts_per_signal', 0):.1f}")
    
    # Generate detailed report for one signal
    print("\nDetailed Quality Report Example:")
    print("=" * 50)
    detailed_assessment = assessor.assess_signal_quality(noisy_ecg, lead_name="Noisy_Test")
    print(detailed_assessment.get_quality_report())
    
    print("ECG Signal Quality Assessment Module Test Complete!")