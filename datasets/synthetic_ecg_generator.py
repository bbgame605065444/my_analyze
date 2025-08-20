#!/usr/bin/env python3
"""
Synthetic ECG Generator
======================

Generates synthetic ECG signals for testing, validation, and data augmentation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SyntheticConfig:
    """Configuration for synthetic ECG generation."""
    sampling_rate: int = 500
    duration: float = 10.0
    num_leads: int = 12
    
class SyntheticECGGenerator:
    """Synthetic ECG generator (placeholder)."""
    
    def __init__(self, config: SyntheticConfig = None, **kwargs):
        if config is None:
            config = SyntheticConfig(**kwargs)
        self.config = config
    
    def generate_normal_ecg(self) -> np.ndarray:
        """Generate synthetic normal ECG."""
        n_samples = int(self.config.duration * self.config.sampling_rate)
        return np.random.randn(self.config.num_leads, n_samples) * 0.1