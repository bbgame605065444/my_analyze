#!/usr/bin/env python3
"""
ECG Feature Extraction Module
=============================

Advanced feature extraction for ECG signals in CoT-RAG Stage 4.
Extracts comprehensive time-domain, frequency-domain, and morphological
features for SE-ResNet and HAN classifier input.

Features Extracted:
- Time-domain: RR intervals, HRV metrics, QRS width, PR interval
- Frequency-domain: Spectral power, frequency ratios, wavelet coefficients
- Morphological: P-wave, QRS complex, T-wave characteristics
- Statistical: Signal statistics, entropy measures
- Clinical: Heart rate variability, arrhythmia indicators

Clinical Relevance:
- Compatible with cardiology diagnostic criteria
- Supports arrhythmia classification
- Enables ischemia detection
- Provides morphology analysis
"""

import numpy as np
import scipy.signal
import scipy.stats
from scipy import integrate
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    import pywt  # PyWavelets for wavelet analysis
    WAVELETS_AVAILABLE = True
except ImportError:
    WAVELETS_AVAILABLE = False
    warnings.warn("PyWavelets not available. Wavelet features will be skipped.")

class FeatureCategory(Enum):
    """ECG feature categories."""
    TIME_DOMAIN = "time_domain"
    FREQUENCY_DOMAIN = "frequency_domain"
    MORPHOLOGICAL = "morphological"
    STATISTICAL = "statistical"
    CLINICAL = "clinical"
    WAVELET = "wavelet"

@dataclass
class ECGFeatures:
    """Container for extracted ECG features."""
    time_domain: Dict[str, float]
    frequency_domain: Dict[str, float]
    morphological: Dict[str, float]
    statistical: Dict[str, float]
    clinical: Dict[str, float]
    wavelet: Dict[str, float]
    metadata: Dict[str, any]
    
    def to_vector(self, feature_categories: List[FeatureCategory] = None) -> np.ndarray:
        """Convert features to numerical vector for ML models."""
        if feature_categories is None:
            feature_categories = list(FeatureCategory)
        
        feature_vector = []
        
        for category in feature_categories:
            if category == FeatureCategory.TIME_DOMAIN:
                feature_vector.extend(self.time_domain.values())
            elif category == FeatureCategory.FREQUENCY_DOMAIN:
                feature_vector.extend(self.frequency_domain.values())
            elif category == FeatureCategory.MORPHOLOGICAL:
                feature_vector.extend(self.morphological.values())
            elif category == FeatureCategory.STATISTICAL:
                feature_vector.extend(self.statistical.values())
            elif category == FeatureCategory.CLINICAL:
                feature_vector.extend(self.clinical.values())
            elif category == FeatureCategory.WAVELET:
                feature_vector.extend(self.wavelet.values())
        
        return np.array(feature_vector, dtype=np.float32)
    
    def get_feature_names(self, feature_categories: List[FeatureCategory] = None) -> List[str]:
        """Get feature names for vector interpretation."""
        if feature_categories is None:
            feature_categories = list(FeatureCategory)
        
        feature_names = []
        
        for category in feature_categories:
            if category == FeatureCategory.TIME_DOMAIN:
                feature_names.extend([f"time_{k}" for k in self.time_domain.keys()])
            elif category == FeatureCategory.FREQUENCY_DOMAIN:
                feature_names.extend([f"freq_{k}" for k in self.frequency_domain.keys()])
            elif category == FeatureCategory.MORPHOLOGICAL:
                feature_names.extend([f"morph_{k}" for k in self.morphological.keys()])
            elif category == FeatureCategory.STATISTICAL:
                feature_names.extend([f"stat_{k}" for k in self.statistical.keys()])
            elif category == FeatureCategory.CLINICAL:
                feature_names.extend([f"clin_{k}" for k in self.clinical.keys()])
            elif category == FeatureCategory.WAVELET:
                feature_names.extend([f"wavelet_{k}" for k in self.wavelet.keys()])
        
        return feature_names

class ECGFeatureExtractor:
    """
    Comprehensive ECG feature extraction system.
    
    Extracts clinically relevant features from preprocessed ECG signals
    for deep learning classification and medical analysis.
    """
    
    def __init__(self, 
                 sampling_rate: float = 500.0,
                 extract_wavelets: bool = True):
        """
        Initialize ECG feature extractor.
        
        Args:
            sampling_rate: ECG sampling rate (Hz)
            extract_wavelets: Whether to extract wavelet features
        """
        self.sampling_rate = sampling_rate
        self.extract_wavelets = extract_wavelets and WAVELETS_AVAILABLE
        self.feature_extraction_stats = {}
        
    def extract_features(self, 
                        ecg_signal: np.ndarray,
                        r_peaks: Optional[np.ndarray] = None,
                        lead_name: str = "unknown") -> ECGFeatures:
        """
        Extract comprehensive features from ECG signal.
        
        Args:
            ecg_signal: Preprocessed ECG signal
            r_peaks: Optional R-peak locations (sample indices)
            lead_name: ECG lead identifier
            
        Returns:
            ECGFeatures object containing all extracted features
        """
        try:
            # Initialize feature dictionaries
            time_features = {}
            freq_features = {}
            morph_features = {}
            stat_features = {}
            clin_features = {}
            wavelet_features = {}
            
            # Extract features by category
            time_features = self._extract_time_domain_features(ecg_signal, r_peaks)
            freq_features = self._extract_frequency_domain_features(ecg_signal)
            morph_features = self._extract_morphological_features(ecg_signal, r_peaks)
            stat_features = self._extract_statistical_features(ecg_signal)
            clin_features = self._extract_clinical_features(ecg_signal, r_peaks)
            
            if self.extract_wavelets:
                wavelet_features = self._extract_wavelet_features(ecg_signal)
            
            # Create metadata
            metadata = {
                'lead_name': lead_name,
                'signal_length': len(ecg_signal),
                'sampling_rate': self.sampling_rate,
                'duration_sec': len(ecg_signal) / self.sampling_rate,
                'r_peaks_detected': len(r_peaks) if r_peaks is not None else 0,
                'feature_extraction_success': True
            }
            
            # Create ECGFeatures object
            features = ECGFeatures(
                time_domain=time_features,
                frequency_domain=freq_features,
                morphological=morph_features,
                statistical=stat_features,
                clinical=clin_features,
                wavelet=wavelet_features,
                metadata=metadata
            )
            
            self._update_extraction_stats(features, success=True)
            return features
            
        except Exception as e:
            warnings.warn(f"Feature extraction failed for {lead_name}: {e}")
            
            # Return empty features with error metadata
            empty_features = ECGFeatures(
                time_domain={}, frequency_domain={}, morphological={},
                statistical={}, clinical={}, wavelet={},
                metadata={'lead_name': lead_name, 'error': str(e), 
                         'feature_extraction_success': False}
            )
            
            self._update_extraction_stats(empty_features, success=False)
            return empty_features
    
    def _extract_time_domain_features(self, 
                                    signal: np.ndarray, 
                                    r_peaks: Optional[np.ndarray]) -> Dict[str, float]:
        """Extract time-domain features."""
        features = {}
        
        # Basic signal characteristics
        features['signal_mean'] = float(np.mean(signal))
        features['signal_std'] = float(np.std(signal))
        features['signal_var'] = float(np.var(signal))
        features['signal_range'] = float(np.ptp(signal))
        features['signal_energy'] = float(np.sum(signal**2))
        
        # RR interval analysis (if R-peaks available)
        if r_peaks is not None and len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / self.sampling_rate  # Convert to seconds
            
            if len(rr_intervals) > 0:
                features['mean_rr'] = float(np.mean(rr_intervals))
                features['std_rr'] = float(np.std(rr_intervals))
                features['min_rr'] = float(np.min(rr_intervals))
                features['max_rr'] = float(np.max(rr_intervals))
                features['rr_range'] = float(np.ptp(rr_intervals))
                
                # Heart rate from RR intervals
                features['mean_hr'] = float(60 / np.mean(rr_intervals))
                features['hr_variability'] = float(np.std(60 / rr_intervals))
                
                # RR interval ratios
                if len(rr_intervals) > 1:
                    rr_ratios = rr_intervals[1:] / rr_intervals[:-1]
                    features['rr_ratio_mean'] = float(np.mean(rr_ratios))
                    features['rr_ratio_std'] = float(np.std(rr_ratios))
            else:
                # Default values when no valid RR intervals
                features.update({
                    'mean_rr': 0.8, 'std_rr': 0.0, 'min_rr': 0.8, 'max_rr': 0.8,
                    'rr_range': 0.0, 'mean_hr': 75.0, 'hr_variability': 0.0,
                    'rr_ratio_mean': 1.0, 'rr_ratio_std': 0.0
                })
        else:
            # Default values when no R-peaks detected
            features.update({
                'mean_rr': 0.8, 'std_rr': 0.0, 'min_rr': 0.8, 'max_rr': 0.8,
                'rr_range': 0.0, 'mean_hr': 75.0, 'hr_variability': 0.0,
                'rr_ratio_mean': 1.0, 'rr_ratio_std': 0.0
            })
        
        # Signal derivative features
        first_diff = np.diff(signal)
        features['first_diff_mean'] = float(np.mean(first_diff))
        features['first_diff_std'] = float(np.std(first_diff))
        
        second_diff = np.diff(first_diff)
        features['second_diff_mean'] = float(np.mean(second_diff))
        features['second_diff_std'] = float(np.std(second_diff))
        
        return features
    
    def _extract_frequency_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features using FFT analysis."""
        features = {}
        
        # Compute power spectral density
        freqs, psd = scipy.signal.periodogram(signal, fs=self.sampling_rate)
        
        # Define frequency bands (clinical ECG bands)
        bands = {
            'ultra_low': (0, 0.04),      # ULF: 0-0.04 Hz
            'very_low': (0.04, 0.15),    # VLF: 0.04-0.15 Hz  
            'low': (0.15, 0.4),          # LF: 0.15-0.4 Hz
            'high': (0.4, 0.5),          # HF: 0.4-0.5 Hz (limited by Nyquist)
            'p_wave': (0.5, 3),          # P-wave band
            'qrs': (3, 40),              # QRS complex band
            't_wave': (1, 5)             # T-wave band
        }
        
        # Calculate power in each frequency band
        total_power = integrate.trapz(psd, freqs)
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequency indices for this band
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = integrate.trapz(psd[band_mask], freqs[band_mask])
                features[f'{band_name}_power'] = float(band_power)
                features[f'{band_name}_power_rel'] = float(band_power / (total_power + 1e-10))
            else:
                features[f'{band_name}_power'] = 0.0
                features[f'{band_name}_power_rel'] = 0.0
        
        # Spectral characteristics
        features['total_power'] = float(total_power)
        features['spectral_centroid'] = float(np.sum(freqs * psd) / (total_power + 1e-10))
        
        # Peak frequency
        peak_idx = np.argmax(psd)
        features['peak_frequency'] = float(freqs[peak_idx])
        features['peak_power'] = float(psd[peak_idx])
        
        # Frequency band ratios (HRV analysis)
        lf_power = features['low_power']
        hf_power = features['high_power']
        features['lf_hf_ratio'] = float(lf_power / (hf_power + 1e-10))
        
        # Spectral entropy
        normalized_psd = psd / (np.sum(psd) + 1e-10)
        spectral_entropy = -np.sum(normalized_psd * np.log(normalized_psd + 1e-10))
        features['spectral_entropy'] = float(spectral_entropy)
        
        return features
    
    def _extract_morphological_features(self, 
                                      signal: np.ndarray, 
                                      r_peaks: Optional[np.ndarray]) -> Dict[str, float]:
        """Extract morphological features from ECG waveforms."""
        features = {}
        
        # Basic morphological statistics
        features['amplitude_mean'] = float(np.mean(signal))
        features['amplitude_std'] = float(np.std(signal))
        features['amplitude_skewness'] = float(scipy.stats.skew(signal))
        features['amplitude_kurtosis'] = float(scipy.stats.kurtosis(signal))
        
        # Peak and valley analysis
        peaks, _ = scipy.signal.find_peaks(signal, height=np.std(signal))
        valleys, _ = scipy.signal.find_peaks(-signal, height=np.std(signal))
        
        features['num_peaks'] = float(len(peaks))
        features['num_valleys'] = float(len(valleys))
        features['peak_valley_ratio'] = float(len(peaks) / (len(valleys) + 1))
        
        # Peak characteristics
        if len(peaks) > 0:
            peak_heights = signal[peaks]
            features['peak_height_mean'] = float(np.mean(peak_heights))
            features['peak_height_std'] = float(np.std(peak_heights))
            features['max_peak_height'] = float(np.max(peak_heights))
        else:
            features.update({
                'peak_height_mean': 0.0, 'peak_height_std': 0.0, 'max_peak_height': 0.0
            })
        
        # QRS complex analysis (if R-peaks available)
        if r_peaks is not None and len(r_peaks) > 0:
            qrs_features = self._analyze_qrs_complexes(signal, r_peaks)
            features.update(qrs_features)
        else:
            # Default QRS features
            features.update({
                'qrs_width_mean': 0.08, 'qrs_width_std': 0.01,
                'qrs_amplitude_mean': 1.0, 'qrs_amplitude_std': 0.1,
                'r_peak_prominence_mean': 0.5, 'r_peak_prominence_std': 0.1
            })
        
        # Waveform complexity
        # Hjorth parameters
        hjorth_activity, hjorth_mobility, hjorth_complexity = self._compute_hjorth_parameters(signal)
        features['hjorth_activity'] = float(hjorth_activity)
        features['hjorth_mobility'] = float(hjorth_mobility)
        features['hjorth_complexity'] = float(hjorth_complexity)
        
        return features
    
    def _extract_statistical_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from ECG signal."""
        features = {}
        
        # Basic statistics
        features['mean'] = float(np.mean(signal))
        features['median'] = float(np.median(signal))
        features['std'] = float(np.std(signal))
        features['variance'] = float(np.var(signal))
        features['min'] = float(np.min(signal))
        features['max'] = float(np.max(signal))
        features['range'] = float(np.ptp(signal))
        features['rms'] = float(np.sqrt(np.mean(signal**2)))
        
        # Distribution shape
        features['skewness'] = float(scipy.stats.skew(signal))
        features['kurtosis'] = float(scipy.stats.kurtosis(signal))
        
        # Percentiles
        percentiles = [5, 10, 25, 75, 90, 95]
        for p in percentiles:
            features[f'percentile_{p}'] = float(np.percentile(signal, p))
        
        # Inter-quartile range
        features['iqr'] = float(np.percentile(signal, 75) - np.percentile(signal, 25))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features['zero_crossing_rate'] = float(zero_crossings / len(signal))
        
        # Signal entropy (approximate)
        hist, _ = np.histogram(signal, bins=50)
        hist_norm = hist / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
        features['signal_entropy'] = float(entropy)
        
        return features
    
    def _extract_clinical_features(self, 
                                 signal: np.ndarray, 
                                 r_peaks: Optional[np.ndarray]) -> Dict[str, float]:
        """Extract clinically relevant features."""
        features = {}
        
        # Heart rate variability (HRV) features
        if r_peaks is not None and len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # Convert to ms
            
            if len(rr_intervals) > 0:
                # Time-domain HRV
                features['sdnn'] = float(np.std(rr_intervals))  # Standard deviation of NN intervals
                features['rmssd'] = float(np.sqrt(np.mean(np.diff(rr_intervals)**2)))  # RMSSD
                
                # pNN50: percentage of successive RR intervals > 50ms
                nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
                features['pnn50'] = float(nn50 / len(rr_intervals) * 100)
                
                # Triangular index approximation
                hist, bins = np.histogram(rr_intervals, bins=50)
                if np.max(hist) > 0:
                    features['tri_index'] = float(len(rr_intervals) / np.max(hist))
                else:
                    features['tri_index'] = 0.0
            else:
                features.update({
                    'sdnn': 50.0, 'rmssd': 30.0, 'pnn50': 10.0, 'tri_index': 20.0
                })
        else:
            # Default HRV values
            features.update({
                'sdnn': 50.0, 'rmssd': 30.0, 'pnn50': 10.0, 'tri_index': 20.0
            })
        
        # Arrhythmia indicators
        features['rhythm_regularity'] = self._assess_rhythm_regularity(signal, r_peaks)
        features['ectopic_beats'] = self._detect_ectopic_beats(signal, r_peaks)
        
        # ST segment analysis (simplified)
        features['st_deviation'] = self._analyze_st_segment(signal, r_peaks)
        
        # QT interval estimation (simplified)
        features['qt_interval'] = self._estimate_qt_interval(signal, r_peaks)
        
        return features
    
    def _extract_wavelet_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract wavelet-based features."""
        if not WAVELETS_AVAILABLE:
            return {}
        
        features = {}
        
        try:
            # Multi-resolution analysis using different wavelets
            wavelets = ['db4', 'db8', 'haar', 'coif2']
            
            for wavelet_name in wavelets:
                # Discrete wavelet transform
                coeffs = pywt.wavedec(signal, wavelet_name, level=5)
                
                # Energy in each level
                for i, coeff in enumerate(coeffs):
                    energy = np.sum(coeff**2)
                    features[f'{wavelet_name}_level{i}_energy'] = float(energy)
                    
                    # Statistical features of coefficients
                    features[f'{wavelet_name}_level{i}_mean'] = float(np.mean(coeff))
                    features[f'{wavelet_name}_level{i}_std'] = float(np.std(coeff))
                
                # Total wavelet energy
                total_energy = sum(np.sum(c**2) for c in coeffs)
                features[f'{wavelet_name}_total_energy'] = float(total_energy)
            
            # Continuous wavelet transform features (using Morlet wavelet)
            scales = np.arange(1, 32)
            coeffs_cwt, freqs_cwt = pywt.cwt(signal[:min(1000, len(signal))], 
                                           scales, 'morl', sampling_period=1/self.sampling_rate)
            
            # Energy distribution across scales
            cwt_energy = np.sum(np.abs(coeffs_cwt)**2, axis=1)
            features['cwt_energy_mean'] = float(np.mean(cwt_energy))
            features['cwt_energy_std'] = float(np.std(cwt_energy))
            features['cwt_peak_scale'] = float(scales[np.argmax(cwt_energy)])
            
        except Exception as e:
            warnings.warn(f"Wavelet feature extraction failed: {e}")
        
        return features
    
    def _analyze_qrs_complexes(self, signal: np.ndarray, r_peaks: np.ndarray) -> Dict[str, float]:
        """Analyze QRS complex characteristics."""
        features = {}
        
        if len(r_peaks) == 0:
            return {'qrs_width_mean': 0.08, 'qrs_width_std': 0.01,
                   'qrs_amplitude_mean': 1.0, 'qrs_amplitude_std': 0.1,
                   'r_peak_prominence_mean': 0.5, 'r_peak_prominence_std': 0.1}
        
        qrs_widths = []
        qrs_amplitudes = []
        r_prominences = []
        
        # Define QRS window (typically 40-120 ms)
        qrs_window_samples = int(0.12 * self.sampling_rate)  # 120 ms window
        
        for r_peak in r_peaks:
            # Define QRS complex window around R-peak
            start = max(0, r_peak - qrs_window_samples // 2)
            end = min(len(signal), r_peak + qrs_window_samples // 2)
            
            if end - start > 10:  # Minimum window size
                qrs_segment = signal[start:end]
                
                # QRS width estimation (zero-crossing based)
                # Find significant deflections from baseline
                baseline = np.mean(qrs_segment)
                significant_deflection = np.std(qrs_segment)
                
                above_threshold = np.abs(qrs_segment - baseline) > significant_deflection
                if np.any(above_threshold):
                    first_deflection = np.argmax(above_threshold)
                    last_deflection = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])
                    width_samples = last_deflection - first_deflection
                    width_sec = width_samples / self.sampling_rate
                    qrs_widths.append(width_sec)
                
                # QRS amplitude (peak-to-peak)
                amplitude = np.ptp(qrs_segment)
                qrs_amplitudes.append(amplitude)
                
                # R-peak prominence
                prominence = signal[r_peak] - baseline
                r_prominences.append(prominence)
        
        # Calculate statistics
        if qrs_widths:
            features['qrs_width_mean'] = float(np.mean(qrs_widths))
            features['qrs_width_std'] = float(np.std(qrs_widths))
        else:
            features['qrs_width_mean'] = 0.08
            features['qrs_width_std'] = 0.01
        
        if qrs_amplitudes:
            features['qrs_amplitude_mean'] = float(np.mean(qrs_amplitudes))
            features['qrs_amplitude_std'] = float(np.std(qrs_amplitudes))
        else:
            features['qrs_amplitude_mean'] = 1.0
            features['qrs_amplitude_std'] = 0.1
        
        if r_prominences:
            features['r_peak_prominence_mean'] = float(np.mean(r_prominences))
            features['r_peak_prominence_std'] = float(np.std(r_prominences))
        else:
            features['r_peak_prominence_mean'] = 0.5
            features['r_peak_prominence_std'] = 0.1
        
        return features
    
    def _compute_hjorth_parameters(self, signal: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute Hjorth parameters (Activity, Mobility, Complexity).
        
        Activity: Variance of the signal
        Mobility: Standard deviation of the derivative / standard deviation of signal
        Complexity: Mobility of derivative / Mobility of signal
        """
        # First derivative
        first_deriv = np.diff(signal)
        # Second derivative  
        second_deriv = np.diff(first_deriv)
        
        # Activity
        activity = np.var(signal)
        
        # Mobility
        mobility = np.std(first_deriv) / (np.std(signal) + 1e-10)
        
        # Complexity
        mobility_deriv = np.std(second_deriv) / (np.std(first_deriv) + 1e-10)
        complexity = mobility_deriv / (mobility + 1e-10)
        
        return activity, mobility, complexity
    
    def _assess_rhythm_regularity(self, 
                                signal: np.ndarray, 
                                r_peaks: Optional[np.ndarray]) -> float:
        """Assess rhythm regularity (0=irregular, 1=regular)."""
        if r_peaks is None or len(r_peaks) < 3:
            return 0.5  # Unknown
        
        rr_intervals = np.diff(r_peaks)
        if len(rr_intervals) < 2:
            return 0.5
        
        # Calculate coefficient of variation of RR intervals
        cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-10)
        
        # Regular rhythm has CV < 0.1, irregular has CV > 0.3
        regularity = max(0.0, min(1.0, (0.3 - cv) / 0.2))
        
        return float(regularity)
    
    def _detect_ectopic_beats(self, 
                            signal: np.ndarray, 
                            r_peaks: Optional[np.ndarray]) -> float:
        """Detect potential ectopic beats (simplified)."""
        if r_peaks is None or len(r_peaks) < 3:
            return 0.0
        
        rr_intervals = np.diff(r_peaks)
        if len(rr_intervals) < 2:
            return 0.0
        
        # Look for sudden changes in RR intervals (simple ectopic detection)
        mean_rr = np.mean(rr_intervals)
        ectopic_threshold = 0.3 * mean_rr  # 30% deviation
        
        ectopic_beats = 0
        for i in range(1, len(rr_intervals)):
            if abs(rr_intervals[i] - rr_intervals[i-1]) > ectopic_threshold:
                ectopic_beats += 1
        
        ectopic_rate = ectopic_beats / len(r_peaks)
        return float(ectopic_rate)
    
    def _analyze_st_segment(self, 
                          signal: np.ndarray, 
                          r_peaks: Optional[np.ndarray]) -> float:
        """Analyze ST segment deviation (simplified)."""
        if r_peaks is None or len(r_peaks) == 0:
            return 0.0
        
        st_deviations = []
        
        # ST segment is approximately 80-120 ms after R-peak
        st_offset_samples = int(0.08 * self.sampling_rate)  # 80 ms
        st_duration_samples = int(0.08 * self.sampling_rate)  # 80 ms window
        
        for r_peak in r_peaks:
            st_start = r_peak + st_offset_samples
            st_end = st_start + st_duration_samples
            
            if st_end < len(signal):
                # Baseline reference (before QRS)
                baseline_start = max(0, r_peak - int(0.2 * self.sampling_rate))
                baseline_end = max(0, r_peak - int(0.1 * self.sampling_rate))
                
                if baseline_end > baseline_start:
                    baseline = np.mean(signal[baseline_start:baseline_end])
                    st_level = np.mean(signal[st_start:st_end])
                    st_deviation = st_level - baseline
                    st_deviations.append(st_deviation)
        
        if st_deviations:
            return float(np.mean(np.abs(st_deviations)))
        else:
            return 0.0
    
    def _estimate_qt_interval(self, 
                            signal: np.ndarray, 
                            r_peaks: Optional[np.ndarray]) -> float:
        """Estimate QT interval (simplified)."""
        if r_peaks is None or len(r_peaks) < 2:
            return 0.4  # Default QT interval (400 ms)
        
        # QT interval is typically 300-450 ms
        # Simplified estimation: look for T-wave end after R-peak
        qt_intervals = []
        
        for i, r_peak in enumerate(r_peaks[:-1]):
            # Search for T-wave end between current and next R-peak
            next_r_peak = r_peaks[i+1]
            search_start = r_peak + int(0.2 * self.sampling_rate)  # Start 200ms after R
            search_end = min(next_r_peak, r_peak + int(0.5 * self.sampling_rate))  # Max 500ms
            
            if search_end > search_start:
                # Find minimum in T-wave region (T-wave end approximation)
                t_wave_region = signal[search_start:search_end]
                t_end_relative = np.argmin(np.abs(np.diff(t_wave_region))) + search_start
                
                qt_duration = (t_end_relative - r_peak) / self.sampling_rate
                if 0.25 < qt_duration < 0.6:  # Reasonable QT range
                    qt_intervals.append(qt_duration)
        
        if qt_intervals:
            return float(np.mean(qt_intervals))
        else:
            return 0.4  # Default
    
    def _update_extraction_stats(self, features: ECGFeatures, success: bool):
        """Update feature extraction statistics."""
        if not hasattr(self, 'feature_extraction_stats'):
            self.feature_extraction_stats = {
                'extractions_attempted': 0,
                'extractions_successful': 0,
                'extractions_failed': 0,
                'total_features_extracted': 0
            }
        
        self.feature_extraction_stats['extractions_attempted'] += 1
        
        if success:
            self.feature_extraction_stats['extractions_successful'] += 1
            # Count total features
            feature_count = (len(features.time_domain) + len(features.frequency_domain) +
                           len(features.morphological) + len(features.statistical) +
                           len(features.clinical) + len(features.wavelet))
            self.feature_extraction_stats['total_features_extracted'] += feature_count
        else:
            self.feature_extraction_stats['extractions_failed'] += 1
    
    def get_extraction_stats(self) -> Dict:
        """Get feature extraction statistics."""
        if not hasattr(self, 'feature_extraction_stats'):
            return {}
        
        stats = self.feature_extraction_stats.copy()
        if stats['extractions_attempted'] > 0:
            stats['success_rate'] = stats['extractions_successful'] / stats['extractions_attempted']
            if stats['extractions_successful'] > 0:
                stats['avg_features_per_extraction'] = (stats['total_features_extracted'] / 
                                                       stats['extractions_successful'])
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("ECG Feature Extraction Module Test")
    print("=" * 50)
    
    # Create test ECG signal
    from signal_preprocessing import create_test_ecg_signal, ECGSignalProcessor
    
    # Generate test signal
    test_ecg = create_test_ecg_signal(duration=10.0, sampling_rate=500.0, leads=["II"])
    test_signal = test_ecg["II"]
    
    print(f"Generated test ECG signal: {len(test_signal)} samples")
    
    # Preprocess signal to get R-peaks
    processor = ECGSignalProcessor()
    processed_result = processor.preprocess_signal({"II": test_signal}, sampling_rate=500.0)
    
    r_peaks = processed_result['metadata']['r_peaks'].get('II', None)
    processed_signal = processed_result['processed_ecg']['II']
    
    print(f"R-peaks detected: {len(r_peaks) if r_peaks is not None else 0}")
    
    # Extract features
    extractor = ECGFeatureExtractor(sampling_rate=500.0)
    features = extractor.extract_features(processed_signal, r_peaks, "II")
    
    # Display feature extraction results
    print(f"\nFeature Extraction Results:")
    print(f"  Success: {features.metadata['feature_extraction_success']}")
    print(f"  Time-domain features: {len(features.time_domain)}")
    print(f"  Frequency-domain features: {len(features.frequency_domain)}")
    print(f"  Morphological features: {len(features.morphological)}")
    print(f"  Statistical features: {len(features.statistical)}")
    print(f"  Clinical features: {len(features.clinical)}")
    print(f"  Wavelet features: {len(features.wavelet)}")
    
    # Show some key features
    print(f"\nKey Features:")
    print(f"  Mean HR: {features.time_domain.get('mean_hr', 'N/A'):.1f} BPM")
    print(f"  HRV (SDNN): {features.clinical.get('sdnn', 'N/A'):.1f} ms")
    print(f"  QRS width: {features.morphological.get('qrs_width_mean', 'N/A'):.3f} s")
    print(f"  Rhythm regularity: {features.clinical.get('rhythm_regularity', 'N/A'):.2f}")
    
    # Convert to feature vector
    feature_vector = features.to_vector()
    feature_names = features.get_feature_names()
    
    print(f"\nFeature Vector:")
    print(f"  Total features: {len(feature_vector)}")
    print(f"  Feature vector shape: {feature_vector.shape}")
    print(f"  Sample features: {feature_vector[:5]}")
    
    # Show extraction statistics
    stats = extractor.get_extraction_stats()
    print(f"\nExtraction Statistics:")
    print(f"  Extractions attempted: {stats.get('extractions_attempted', 0)}")
    print(f"  Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    print(f"  Features per extraction: {stats.get('avg_features_per_extraction', 0):.1f}")
    
    print("\nECG Feature Extraction Module Test Complete!")