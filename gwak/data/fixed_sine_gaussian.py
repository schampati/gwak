"""Fixed-frequency/Q sine-Gaussian prior used by the directional studies.

The 2048 Hz upper bound assumes the study's 4096 Hz sampling rate.
Other sampled parameters and random-number consumption match SineGaussianBBC.
"""
import numpy as np
import torch
from gwak.data.prior import SineGaussianBBC


class FixedSineGaussianPrior(SineGaussianBBC):
    def __init__(self, frequency: float = 100.0, quality: float = 10.0):
        super().__init__()
        if not np.isfinite(frequency) or not 0 < frequency < 2048:
            raise ValueError('frequency must be between 0 and 2048 Hz')
        if not np.isfinite(quality) or quality <= 0:
            raise ValueError('quality must be positive and finite')
        self.frequency = float(frequency)
        self.quality = float(quality)

    def sample(self, batch_size):
        params = super().sample(batch_size)
        # BasePrior's Constant handling returns scalars; the generator needs vectors.
        params['frequency'] = torch.full((batch_size,), self.frequency)
        params['quality'] = torch.full((batch_size,), self.quality)
        return params
