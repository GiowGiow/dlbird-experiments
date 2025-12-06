"""
Augmentation subpackage for audio spectrograms.
"""

from .spec_augment import (
    FrequencyMasking,
    TimeMasking,
    SpecAugment,
    mixup_batch,
    BackgroundNoiseInjection,
)

__all__ = [
    'FrequencyMasking',
    'TimeMasking',
    'SpecAugment',
    'mixup_batch',
    'BackgroundNoiseInjection',
]
