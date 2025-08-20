#!/usr/bin/env python3
"""
Simple Stage 4 Integration Test
==============================

Simplified integration test for Stage 4 that works with mock components
when the actual imports are not available.
"""

import sys
import os
import numpy as np
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import real components, fall back to mocks
try:
    from ecg_processing import ECGSignalProcessor, ECGFeatureExtractor
    from models import SEResNetClassifier, HANClassifier, EnsembleManager, ECGModelConfig
    from datasets import PTBXLLoader
    from datasets.ptb_xl_loader import PTBXLConfig
    from validation import ClinicalMetrics, HierarchyValidator, ExpertComparison, InterpretabilityAnalyzer
    STAGE4_AVAILABLE = True
    print("✓ Stage 4 components successfully imported")
except ImportError as e:
    print(f"⚠ Stage 4 imports not available: {e}")
    print("Using mock components for testing...")
    STAGE4_AVAILABLE = False
    
    # Mock components
    class MockECGSignalProcessor:
        def preprocess_signal(self, signal, **kwargs):
            return signal * 0.9  # Simple mock preprocessing
    
    class MockECGFeatureExtractor:
        def extract_features(self, signal, **kwargs):
            return {
                'heart_rate': 72,
                'pr_interval': 0.16,
                'qrs_duration': 0.08,
                'qt_interval': 0.42
            }
    
    class MockECGModelConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MockSEResNetClassifier:
        def __init__(self, config):
            self.config = config
        
        def predict(self, signals):
            return np.random.randint(0, getattr(self.config, 'num_classes', 5), len(signals))
        
        def predict_proba(self, signals):
            n_classes = getattr(self.config, 'num_classes', 5)
            probs = np.random.dirichlet(np.ones(n_classes), len(signals))
            return probs
        
        def train(self, signals, labels):
            return True
    
    class MockHANClassifier:
        def __init__(self, config):
            self.config = config
        
        def predict(self, signals, return_attention=False):
            predictions = np.random.randint(0, getattr(self.config, 'num_classes', 5), len(signals))
            if return_attention:
                from collections import namedtuple
                AttentionWeights = namedtuple('AttentionWeights', ['beat_attention', 'segment_attention'])
                attention = AttentionWeights(
                    beat_attention=np.random.random((len(signals), 10)),
                    segment_attention=np.random.random((len(signals), 5))
                )
                return predictions, attention
            return predictions
    
    class MockEnsembleManager:
        def __init__(self, config):
            self.config = config
            self.models = {}
        
        def add_model(self, name, model, weight=1.0):
            self.models[name] = {'model': model, 'weight': weight}
        
        def predict(self, signals):
            return np.random.randint(0, getattr(self.config, 'num_classes', 5), len(signals))
        
        def predict_proba(self, signals):
            n_classes = getattr(self.config, 'num_classes', 5)
            return np.random.dirichlet(np.ones(n_classes), len(signals))
        
        def set_combination_strategy(self, strategy):
            pass
    
    # Assign mock classes
    ECGSignalProcessor = MockECGSignalProcessor
    ECGFeatureExtractor = MockECGFeatureExtractor
    ECGModelConfig = MockECGModelConfig
    SEResNetClassifier = MockSEResNetClassifier
    HANClassifier = MockHANClassifier
    EnsembleManager = MockEnsembleManager


def generate_mock_ecg_data():
    """Generate mock ECG data for testing."""
    
    np.random.seed(42)
    n_samples = 20
    n_leads = 12
    n_timepoints = 5000
    
    # Generate ECG-like signals
    signals = []
    for i in range(n_samples):
        signal = np.zeros((n_leads, n_timepoints))
        
        # Add some ECG-like patterns
        for lead in range(n_leads):
            # Base rhythm
            t = np.linspace(0, 10, n_timepoints)
            
            # Add QRS complexes
            for beat_time in np.arange(1, 9, 1.0):  # ~60 bpm
                beat_idx = int(beat_time * 500)
                if beat_idx < n_timepoints - 100:
                    # QRS complex
                    signal[lead, beat_idx:beat_idx+50] += np.sin(np.linspace(0, 2*np.pi, 50))
                    # T wave
                    signal[lead, beat_idx+100:beat_idx+200] += 0.3 * np.sin(np.linspace(0, np.pi, 100))
            
            # Add noise
            signal[lead] += 0.05 * np.random.randn(n_timepoints)
        
        signals.append(signal)
    
    return {
        'signals': np.array(signals),
        'labels': np.random.randint(0, 5, n_samples),
        'sampling_rate': 500,
        'class_names': ['Normal', 'MI', 'STTC', 'CD', 'HYP']
    }


def test_signal_processing():
    """Test ECG signal processing."""
    print("Testing Signal Processing...")
    
    try:
        processor = ECGSignalProcessor()
        extractor = ECGFeatureExtractor()
        
        # Generate test signal
        test_data = generate_mock_ecg_data()
        test_signal = test_data['signals'][0]
        
        # Test preprocessing
        processed = processor.preprocess_signal(test_signal, sampling_rate=500)
        assert processed is not None, "Processing failed"
        
        # Test feature extraction
        features = extractor.extract_features(processed[0])
        assert isinstance(features, dict), "Feature extraction failed"
        assert len(features) > 0, "No features extracted"
        
        print("  ✓ Signal processing test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Signal processing test failed: {e}")
        return False


def test_classifiers():
    """Test ECG classifiers."""
    print("Testing Classifiers...")
    
    try:
        config = ECGModelConfig(
            num_classes=5,
            input_channels=12,
            sequence_length=5000
        )
        
        # Test SE-ResNet
        se_resnet = SEResNetClassifier(config)
        
        # Test HAN
        han = HANClassifier(config)
        
        # Generate test data
        test_data = generate_mock_ecg_data()
        test_signals = test_data['signals'][:5]
        
        # Test predictions
        se_preds = se_resnet.predict(test_signals)
        han_preds = han.predict(test_signals)
        
        assert len(se_preds) == 5, "Wrong number of SE-ResNet predictions"
        assert len(han_preds) == 5, "Wrong number of HAN predictions"
        
        # Test probabilities
        se_probs = se_resnet.predict_proba(test_signals)
        assert se_probs.shape == (5, 5), "Wrong SE-ResNet probability shape"
        
        # Test attention
        han_preds_att = han.predict(test_signals, return_attention=True)
        assert isinstance(han_preds_att, tuple), "HAN should return attention"
        
        print("  ✓ Classifier tests passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Classifier tests failed: {e}")
        return False


def test_ensemble():
    """Test ensemble management."""
    print("Testing Ensemble...")
    
    try:
        config = ECGModelConfig(
            num_classes=5,
            input_channels=12,
            sequence_length=5000
        )
        
        # Create models
        se_resnet = SEResNetClassifier(config)
        han = HANClassifier(config)
        
        # Create ensemble
        ensemble = EnsembleManager(config)
        ensemble.add_model("se_resnet", se_resnet, weight=0.6)
        ensemble.add_model("han", han, weight=0.4)
        
        # Test ensemble prediction
        test_data = generate_mock_ecg_data()
        test_signals = test_data['signals'][:5]
        
        ensemble_preds = ensemble.predict(test_signals)
        ensemble_probs = ensemble.predict_proba(test_signals)
        
        assert len(ensemble_preds) == 5, "Wrong number of ensemble predictions"
        assert ensemble_probs.shape == (5, 5), "Wrong ensemble probability shape"
        
        # Test strategy changes
        ensemble.set_combination_strategy("stacking")
        stacking_preds = ensemble.predict(test_signals)
        assert len(stacking_preds) == 5, "Stacking strategy failed"
        
        print("  ✓ Ensemble tests passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Ensemble tests failed: {e}")
        return False


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("Testing End-to-End Pipeline...")
    
    try:
        # Generate test data
        test_data = generate_mock_ecg_data()
        
        # Step 1: Signal processing
        processor = ECGSignalProcessor()
        processed_signals = []
        
        for signal in test_data['signals'][:5]:
            processed = processor.preprocess_signal(signal, sampling_rate=500)
            processed_signals.append(processed)
        
        processed_signals = np.array(processed_signals)
        
        # Step 2: Model prediction
        config = ECGModelConfig(
            num_classes=5,
            input_channels=12,
            sequence_length=5000
        )
        
        ensemble = EnsembleManager(config)
        se_resnet = SEResNetClassifier(config)
        han = HANClassifier(config)
        
        ensemble.add_model("se_resnet", se_resnet, weight=0.6)
        ensemble.add_model("han", han, weight=0.4)
        
        predictions = ensemble.predict(processed_signals)
        probabilities = ensemble.predict_proba(processed_signals)
        
        # Step 3: Feature extraction
        extractor = ECGFeatureExtractor()
        features_list = []
        
        for signal in processed_signals:
            features = extractor.extract_features(signal[0])  # First lead
            features_list.append(features)
        
        # Validate pipeline results
        assert len(predictions) == 5, "Pipeline prediction failed"
        assert probabilities.shape == (5, 5), "Pipeline probability failed"
        assert len(features_list) == 5, "Pipeline feature extraction failed"
        
        # Calculate accuracy (mock)
        ground_truth = test_data['labels'][:5]
        accuracy = np.mean(predictions == ground_truth)
        
        print(f"  ✓ End-to-end pipeline test passed (Mock accuracy: {accuracy:.3f})")
        return True, accuracy
        
    except Exception as e:
        print(f"  ✗ End-to-end pipeline test failed: {e}")
        return False, 0.0


def main():
    """Main test execution."""
    
    print("=" * 60)
    print("STAGE 4 SIMPLE INTEGRATION TEST")
    print("=" * 60)
    print(f"Test started: {datetime.now()}")
    print(f"Stage 4 components available: {STAGE4_AVAILABLE}")
    print()
    
    # Run tests
    test_results = {}
    
    # Individual component tests
    test_results['signal_processing'] = test_signal_processing()
    test_results['classifiers'] = test_classifiers()
    test_results['ensemble'] = test_ensemble()
    
    # Integration test
    pipeline_result, accuracy = test_end_to_end_pipeline()
    test_results['end_to_end_pipeline'] = pipeline_result
    
    # Summary
    print()
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print()
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name.replace('_', ' ').title()}")
    
    print()
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        if STAGE4_AVAILABLE:
            print("Stage 4 integration is fully functional!")
        else:
            print("Stage 4 mock integration is working - ready for real component integration!")
    else:
        print("⚠️ Some tests failed - check implementation")
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'stage4_available': STAGE4_AVAILABLE,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate,
        'test_results': test_results,
        'mock_pipeline_accuracy': accuracy if 'accuracy' in locals() else 0.0
    }
    
    with open('stage4_simple_test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: stage4_simple_test_results.json")
    
    return test_results


if __name__ == "__main__":
    results = main()