#!/usr/bin/env python3
"""
ECG Signal Preprocessing Module
==============================

Advanced ECG signal preprocessing for CoT-RAG Stage 4 Medical Domain Integration.
Provides comprehensive signal cleaning, noise reduction, and normalization for
multi-lead ECG data compatible with SE-ResNet and HAN classifiers.

Features:
- Digital filtering (bandpass, notch, baseline drift removal)
- Noise reduction and artifact removal
- Signal normalization and standardization
- Multi-lead ECG processing
- R-peak detection and rhythm analysis
- Signal quality validation

Clinical Standards:
- Supports AHA/ESC guidelines for ECG processing
- Compatible with PTB-XL and MIMIC-IV data formats
- Maintains clinical-grade signal fidelity
"""

import numpy as np
import scipy.signal
from scipy.ndimage import median_filter
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum

class ECGLead(Enum):
    """Standard 12-lead ECG nomenclature."""
    I = "I"
    II = "II" 
    III = "III"
    aVR = "aVR"
    aVL = "aVL"
    aVF = "aVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"

@dataclass
class ECGSignalMetadata:
    """Metadata for ECG signal processing."""
    sampling_rate: float
    duration_sec: float
    leads: List[str]
    patient_id: Optional[str] = None
    acquisition_date: Optional[str] = None
    signal_quality: Optional[float] = None
    preprocessing_applied: List[str] = None

@dataclass
class ProcessingParameters:
    """ECG signal processing configuration parameters."""
    # Filtering parameters
    lowpass_cutoff: float = 40.0  # Hz
    highpass_cutoff: float = 0.5  # Hz
    notch_freq: float = 50.0      # Hz (60.0 for US)
    notch_quality: float = 30.0   # Q factor
    
    # Noise reduction
    median_filter_size: int = 3
    baseline_window: float = 0.2  # seconds
    
    # Normalization
    normalize_amplitude: bool = True
    z_score_normalize: bool = True
    
    # R-peak detection
    r_peak_detection: bool = True
    min_rr_interval: float = 0.3  # seconds
    max_rr_interval: float = 2.0  # seconds
    
    # Signal validation
    min_signal_quality: float = 0.7
    max_noise_level: float = 0.3

class ECGSignalProcessor:
    """
    Comprehensive ECG signal preprocessing system.
    
    Implements clinical-grade ECG signal processing pipeline compatible with
    deep learning models (SE-ResNet, HAN) and medical standards.
    """
    
    def __init__(self, 
                 params: Optional[ProcessingParameters] = None,
                 sampling_rate: float = 500.0):
        """
        Initialize ECG signal processor.
        
        Args:
            params: Processing parameters configuration
            sampling_rate: Default sampling rate (Hz)
        """
        self.params = params or ProcessingParameters()
        self.default_sampling_rate = sampling_rate
        self.processing_stats = {}
        
    def preprocess_signal(self, 
                         ecg_data: Union[np.ndarray, Dict[str, np.ndarray]],
                         sampling_rate: Optional[float] = None,
                         metadata: Optional[ECGSignalMetadata] = None) -> Dict:
        """
        Complete ECG signal preprocessing pipeline.
        
        Args:
            ecg_data: Raw ECG data (1D array or dict of lead arrays)
            sampling_rate: Signal sampling rate (Hz)
            metadata: Optional signal metadata
            
        Returns:
            Dict containing processed ECG data and metadata
        """
        fs = sampling_rate or self.default_sampling_rate
        
        # Handle input format
        if isinstance(ecg_data, np.ndarray):
            if ecg_data.ndim == 1:
                # Single lead
                leads_data = {"II": ecg_data}  # Default to lead II
            elif ecg_data.ndim == 2:
                # Multiple leads (assume standard 12-lead order)
                lead_names = [lead.value for lead in ECGLead][:ecg_data.shape[0]]
                leads_data = {name: ecg_data[i] for i, name in enumerate(lead_names)}
            else:
                raise ValueError(f"Unsupported ECG data shape: {ecg_data.shape}")
        elif isinstance(ecg_data, dict):
            leads_data = ecg_data.copy()
        else:
            raise ValueError(f"Unsupported ECG data type: {type(ecg_data)}")
        
        # Initialize processing results
        processed_leads = {}
        processing_metadata = {
            'sampling_rate': fs,
            'original_leads': list(leads_data.keys()),
            'processing_steps': [],
            'signal_quality': {},
            'r_peaks': {}
        }
        
        # Process each lead
        for lead_name, signal in leads_data.items():
            try:
                processed_signal, lead_metadata = self._process_single_lead(
                    signal, lead_name, fs
                )
                processed_leads[lead_name] = processed_signal
                processing_metadata['signal_quality'][lead_name] = lead_metadata['quality_score']
                
                if lead_metadata.get('r_peaks') is not None:
                    processing_metadata['r_peaks'][lead_name] = lead_metadata['r_peaks']
                    
            except Exception as e:
                warnings.warn(f"Failed to process lead {lead_name}: {e}")
                # Use original signal as fallback
                processed_leads[lead_name] = signal
                processing_metadata['signal_quality'][lead_name] = 0.0
        
        # Calculate overall signal quality
        quality_scores = list(processing_metadata['signal_quality'].values())
        overall_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Create result
        result = {
            'processed_ecg': processed_leads,
            'metadata': processing_metadata,
            'original_ecg': leads_data,
            'sampling_rate': fs,
            'overall_quality': overall_quality,
            'processing_successful': len(processed_leads) > 0
        }
        
        # Update processing stats
        self._update_processing_stats(result)
        
        return result
    
    def _process_single_lead(self, 
                           signal: np.ndarray, 
                           lead_name: str, 
                           fs: float) -> Tuple[np.ndarray, Dict]:
        """
        Process a single ECG lead through complete preprocessing pipeline.
        
        Args:
            signal: Raw ECG signal
            lead_name: ECG lead identifier
            fs: Sampling frequency
            
        Returns:
            Tuple of (processed_signal, metadata)
        """
        # Initialize processing steps
        processed_signal = signal.copy().astype(np.float64)
        steps = []
        
        # 1. Basic signal validation
        if len(processed_signal) < fs:  # Less than 1 second
            raise ValueError(f"Signal too short: {len(processed_signal)} samples")
        
        # 2. Baseline drift removal
        if self.params.baseline_window > 0:
            processed_signal = self._remove_baseline_drift(processed_signal, fs)
            steps.append("baseline_drift_removal")
        
        # 3. Bandpass filtering
        processed_signal = self._apply_bandpass_filter(processed_signal, fs)
        steps.append("bandpass_filter")
        
        # 4. Notch filtering (powerline interference)
        processed_signal = self._apply_notch_filter(processed_signal, fs)
        steps.append("notch_filter")
        
        # 5. Noise reduction
        if self.params.median_filter_size > 1:
            processed_signal = self._apply_noise_reduction(processed_signal)
            steps.append("noise_reduction")
        
        # 6. Signal quality assessment
        quality_score = self._assess_signal_quality(processed_signal, fs)
        
        # 7. R-peak detection (if enabled and quality sufficient)
        r_peaks = None
        if (self.params.r_peak_detection and 
            quality_score >= self.params.min_signal_quality):
            try:
                r_peaks = self._detect_r_peaks(processed_signal, fs)
                steps.append("r_peak_detection")
            except Exception as e:
                warnings.warn(f"R-peak detection failed for {lead_name}: {e}")
        
        # 8. Normalization
        if self.params.normalize_amplitude:
            processed_signal = self._normalize_amplitude(processed_signal)
            steps.append("amplitude_normalization")
            
        if self.params.z_score_normalize:
            processed_signal = self._z_score_normalize(processed_signal)
            steps.append("z_score_normalization")
        
        # Create metadata
        metadata = {
            'lead_name': lead_name,
            'processing_steps': steps,
            'quality_score': quality_score,
            'r_peaks': r_peaks,
            'duration_sec': len(signal) / fs,
            'samples_processed': len(processed_signal)
        }
        
        return processed_signal, metadata
    
    def _remove_baseline_drift(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Remove baseline drift using high-pass filtering."""
        # High-pass filter to remove baseline drift
        nyquist = fs / 2
        normalized_cutoff = self.params.highpass_cutoff / nyquist
        
        # Ensure cutoff frequency is valid
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
        
        b, a = scipy.signal.butter(4, normalized_cutoff, btype='high')
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        return filtered_signal
    
    def _apply_bandpass_filter(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Apply bandpass filter to ECG signal."""
        nyquist = fs / 2
        low_normalized = self.params.highpass_cutoff / nyquist
        high_normalized = self.params.lowpass_cutoff / nyquist
        
        # Ensure cutoff frequencies are valid
        low_normalized = max(0.01, min(low_normalized, 0.99))
        high_normalized = max(low_normalized + 0.01, min(high_normalized, 0.99))
        
        # Design bandpass filter
        b, a = scipy.signal.butter(4, [low_normalized, high_normalized], btype='band')
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        return filtered_signal
    
    def _apply_notch_filter(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Apply notch filter to remove powerline interference."""
        nyquist = fs / 2
        notch_normalized = self.params.notch_freq / nyquist
        
        if notch_normalized >= 1.0:
            # Skip notch filtering if frequency is too high
            return signal
        
        # Design notch filter
        b, a = scipy.signal.iirnotch(self.params.notch_freq, 
                                   self.params.notch_quality, fs)
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        return filtered_signal
    
    def _apply_noise_reduction(self, signal: np.ndarray) -> np.ndarray:
        """Apply noise reduction using median filtering."""
        # Median filter for impulse noise reduction
        filtered_signal = median_filter(signal, size=self.params.median_filter_size)
        return filtered_signal
    
    def _assess_signal_quality(self, signal: np.ndarray, fs: float) -> float:
        """
        Assess ECG signal quality.
        
        Returns quality score between 0.0 (poor) and 1.0 (excellent).
        """
        try:
            # Calculate various quality metrics
            
            # 1. Signal-to-noise ratio estimation
            # Use high-frequency content as noise proxy
            b, a = scipy.signal.butter(4, 40/(fs/2), btype='high')
            noise = scipy.signal.filtfilt(b, a, signal)
            snr = 20 * np.log10(np.std(signal) / (np.std(noise) + 1e-10))
            snr_score = min(1.0, max(0.0, (snr - 10) / 20))  # Normalize 10-30 dB to 0-1
            
            # 2. Baseline stability
            baseline_std = np.std(signal - scipy.signal.detrend(signal))
            baseline_score = 1.0 / (1.0 + baseline_std / np.std(signal))
            
            # 3. Dynamic range utilization
            signal_range = np.ptp(signal)  # Peak-to-peak
            range_score = min(1.0, signal_range / (4 * np.std(signal)))
            
            # 4. Artifact detection (simple)
            # Look for sudden amplitude changes
            diff_signal = np.diff(signal)
            artifact_threshold = 5 * np.std(diff_signal)
            artifacts = np.sum(np.abs(diff_signal) > artifact_threshold)
            artifact_score = max(0.0, 1.0 - artifacts / (len(signal) * 0.01))
            
            # Combine scores
            overall_quality = (snr_score * 0.4 + 
                             baseline_score * 0.3 + 
                             range_score * 0.2 + 
                             artifact_score * 0.1)
            
            return float(np.clip(overall_quality, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default quality score
    
    def _detect_r_peaks(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """
        Detect R-peaks in ECG signal using adaptive thresholding.
        
        Returns array of R-peak indices.
        """
        # Preprocessing for R-peak detection
        # Apply additional filtering for better peak detection
        b, a = scipy.signal.butter(4, [5/(fs/2), 20/(fs/2)], btype='band')
        filtered_signal = scipy.signal.filtfilt(b, a, signal)
        
        # Square the signal to emphasize peaks
        squared_signal = filtered_signal ** 2
        
        # Moving average integration
        window_size = int(0.1 * fs)  # 100ms window
        integrated_signal = np.convolve(squared_signal, 
                                      np.ones(window_size)/window_size, 
                                      mode='same')
        
        # Adaptive thresholding
        threshold = 0.4 * np.max(integrated_signal)
        
        # Find peaks above threshold with minimum distance
        min_distance = int(self.params.min_rr_interval * fs)
        peaks, _ = scipy.signal.find_peaks(integrated_signal,
                                         height=threshold,
                                         distance=min_distance)
        
        # Validate R-R intervals
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs
            valid_mask = ((rr_intervals >= self.params.min_rr_interval) &
                         (rr_intervals <= self.params.max_rr_interval))
            
            # Keep peaks with valid intervals
            valid_peaks = [peaks[0]]  # Always keep first peak
            for i, is_valid in enumerate(valid_mask):
                if is_valid:
                    valid_peaks.append(peaks[i+1])
            
            peaks = np.array(valid_peaks)
        
        return peaks
    
    def _normalize_amplitude(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal amplitude to [-1, 1] range."""
        signal_range = np.ptp(signal)
        if signal_range > 0:
            normalized = 2 * (signal - np.min(signal)) / signal_range - 1
        else:
            normalized = signal
        return normalized
    
    def _z_score_normalize(self, signal: np.ndarray) -> np.ndarray:
        """Apply z-score normalization (zero mean, unit variance)."""
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        if signal_std > 0:
            normalized = (signal - signal_mean) / signal_std
        else:
            normalized = signal - signal_mean
        return normalized
    
    def _update_processing_stats(self, result: Dict):
        """Update internal processing statistics."""
        if not hasattr(self, 'processing_stats'):
            self.processing_stats = {
                'signals_processed': 0,
                'total_quality_score': 0.0,
                'successful_processing': 0,
                'failed_processing': 0
            }
        
        self.processing_stats['signals_processed'] += 1
        self.processing_stats['total_quality_score'] += result['overall_quality']
        
        if result['processing_successful']:
            self.processing_stats['successful_processing'] += 1
        else:
            self.processing_stats['failed_processing'] += 1
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics."""
        if not hasattr(self, 'processing_stats'):
            return {}
        
        stats = self.processing_stats.copy()
        if stats['signals_processed'] > 0:
            stats['average_quality'] = (stats['total_quality_score'] / 
                                       stats['signals_processed'])
            stats['success_rate'] = (stats['successful_processing'] / 
                                   stats['signals_processed'])
        
        return stats
    
    def validate_processed_signal(self, processed_result: Dict) -> Dict:
        """
        Validate processed ECG signal for clinical use.
        
        Args:
            processed_result: Result from preprocess_signal()
            
        Returns:
            Validation report with recommendations
        """
        validation_report = {
            'overall_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check overall signal quality
        if processed_result['overall_quality'] < self.params.min_signal_quality:
            validation_report['warnings'].append(
                f"Signal quality ({processed_result['overall_quality']:.2f}) "
                f"below recommended threshold ({self.params.min_signal_quality})"
            )
        
        # Check individual leads
        for lead_name, quality in processed_result['metadata']['signal_quality'].items():
            if quality < 0.3:
                validation_report['errors'].append(
                    f"Lead {lead_name} has very poor quality ({quality:.2f})"
                )
                validation_report['overall_valid'] = False
        
        # Check sampling rate
        fs = processed_result['sampling_rate']
        if fs < 250:
            validation_report['warnings'].append(
                f"Low sampling rate ({fs} Hz) may limit high-frequency analysis"
            )
        
        # Provide recommendations
        if validation_report['warnings'] or validation_report['errors']:
            validation_report['recommendations'].extend([
                "Consider signal reacquisition if quality is insufficient",
                "Verify electrode placement and skin preparation",
                "Check for electromagnetic interference sources"
            ])
        
        return validation_report


def create_test_ecg_signal(duration: float = 10.0, 
                          sampling_rate: float = 500.0,
                          leads: List[str] = None) -> Dict:
    """
    Generate synthetic ECG signal for testing purposes.
    
    Args:
        duration: Signal duration in seconds
        sampling_rate: Sampling frequency in Hz
        leads: List of ECG leads to generate
        
    Returns:
        Dictionary with synthetic ECG data
    """
    if leads is None:
        leads = ["II"]  # Default to lead II
    
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Generate synthetic ECG components
    heart_rate = 70  # BPM
    rr_interval = 60 / heart_rate  # seconds
    
    ecg_data = {}
    
    for lead_name in leads:
        # Basic ECG waveform synthesis
        ecg_signal = np.zeros_like(t)
        
        # Add R-peaks and surrounding complexes
        peak_times = np.arange(0, duration, rr_interval)
        
        for peak_time in peak_times:
            if peak_time < duration:
                # Gaussian R-peak
                r_peak = 1.0 * np.exp(-((t - peak_time) / 0.02) ** 2)
                
                # P-wave (before R)
                p_time = peak_time - 0.15
                if p_time > 0:
                    p_wave = 0.2 * np.exp(-((t - p_time) / 0.04) ** 2)
                    ecg_signal += p_wave
                
                # QRS complex
                ecg_signal += r_peak
                
                # T-wave (after R)
                t_time = peak_time + 0.3
                if t_time < duration:
                    t_wave = 0.3 * np.exp(-((t - t_time) / 0.08) ** 2)
                    ecg_signal += t_wave
        
        # Add noise
        noise_level = 0.05
        noise = noise_level * np.random.randn(len(t))
        ecg_signal += noise
        
        # Add baseline drift
        baseline_drift = 0.1 * np.sin(2 * np.pi * 0.1 * t)
        ecg_signal += baseline_drift
        
        ecg_data[lead_name] = ecg_signal
    
    return ecg_data


# Example usage and testing
if __name__ == "__main__":
    print("ECG Signal Preprocessing Module Test")
    print("=" * 50)
    
    # Create test ECG signal
    test_ecg = create_test_ecg_signal(
        duration=10.0, 
        sampling_rate=500.0, 
        leads=["I", "II", "V1", "V5"]
    )
    
    print(f"Generated test ECG with {len(test_ecg)} leads")
    for lead, signal in test_ecg.items():
        print(f"  Lead {lead}: {len(signal)} samples")
    
    # Initialize processor
    processor = ECGSignalProcessor()
    
    # Process the test signal
    print("\nProcessing ECG signal...")
    result = processor.preprocess_signal(test_ecg, sampling_rate=500.0)
    
    # Display results
    print(f"\nProcessing Results:")
    print(f"  Overall quality: {result['overall_quality']:.3f}")
    print(f"  Leads processed: {len(result['processed_ecg'])}")
    
    for lead_name, quality in result['metadata']['signal_quality'].items():
        print(f"    {lead_name}: quality={quality:.3f}")
    
    # Validate processed signal
    validation = processor.validate_processed_signal(result)
    print(f"\nValidation: {'PASS' if validation['overall_valid'] else 'FAIL'}")
    if validation['warnings']:
        print(f"  Warnings: {len(validation['warnings'])}")
    if validation['errors']:
        print(f"  Errors: {len(validation['errors'])}")
    
    # Show processing stats
    stats = processor.get_processing_stats()
    print(f"\nProcessor Statistics:")
    print(f"  Signals processed: {stats.get('signals_processed', 0)}")
    print(f"  Average quality: {stats.get('average_quality', 0):.3f}")
    print(f"  Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    
    print("\nECG Signal Preprocessing Module Test Complete!")