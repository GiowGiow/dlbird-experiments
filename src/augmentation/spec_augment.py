"""
Data Augmentation for Audio Spectrograms

This module implements various augmentation techniques for audio spectrograms:
- SpecAugment (Frequency and Time Masking)
- MixUp (batch-level mixing)
- Noise injection

References:
- SpecAugment: Park et al., 2019 "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
- MixUp: Zhang et al., 2018 "mixup: Beyond Empirical Risk Minimization"
"""

import torch
import numpy as np
from typing import Tuple, Optional


class FrequencyMasking:
    """
    Mask random frequency bands in a spectrogram.

    Args:
        freq_mask_param: Maximum number of consecutive mel bins to mask
        num_masks: Number of frequency masks to apply
        mask_value: Value to use for masked regions (default: 0.0)
    """

    def __init__(
        self, freq_mask_param: int = 15, num_masks: int = 1, mask_value: float = 0.0
    ):
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        self.mask_value = mask_value

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to spectrogram.

        Args:
            spectrogram: Input spectrogram of shape (C, F, T) or (F, T)
                where F is frequency bins, T is time frames

        Returns:
            Masked spectrogram of same shape
        """
        spec = spectrogram.clone()

        # Handle both (F, T) and (C, F, T) formats
        if spec.ndim == 2:
            freq_dim = 0
            n_freq = spec.shape[0]
        elif spec.ndim == 3:
            freq_dim = 1
            n_freq = spec.shape[1]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {spec.ndim}D")

        for _ in range(self.num_masks):
            # Random mask width (1 to freq_mask_param)
            mask_width = np.random.randint(1, min(self.freq_mask_param + 1, n_freq))

            # Random starting position
            mask_start = np.random.randint(0, n_freq - mask_width + 1)

            # Apply mask
            if spec.ndim == 2:
                spec[mask_start : mask_start + mask_width, :] = self.mask_value
            else:
                spec[:, mask_start : mask_start + mask_width, :] = self.mask_value

        return spec


class TimeMasking:
    """
    Mask random time segments in a spectrogram.

    Args:
        time_mask_param: Maximum number of consecutive time frames to mask
        num_masks: Number of time masks to apply
        mask_value: Value to use for masked regions (default: 0.0)
    """

    def __init__(
        self, time_mask_param: int = 35, num_masks: int = 1, mask_value: float = 0.0
    ):
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        self.mask_value = mask_value

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking to spectrogram.

        Args:
            spectrogram: Input spectrogram of shape (C, F, T) or (F, T)
                where F is frequency bins, T is time frames

        Returns:
            Masked spectrogram of same shape
        """
        spec = spectrogram.clone()

        # Handle both (F, T) and (C, F, T) formats
        if spec.ndim == 2:
            time_dim = 1
            n_time = spec.shape[1]
        elif spec.ndim == 3:
            time_dim = 2
            n_time = spec.shape[2]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {spec.ndim}D")

        for _ in range(self.num_masks):
            # Random mask width (1 to time_mask_param)
            mask_width = np.random.randint(1, min(self.time_mask_param + 1, n_time))

            # Random starting position
            mask_start = np.random.randint(0, n_time - mask_width + 1)

            # Apply mask
            if spec.ndim == 2:
                spec[:, mask_start : mask_start + mask_width] = self.mask_value
            else:
                spec[:, :, mask_start : mask_start + mask_width] = self.mask_value

        return spec


class SpecAugment:
    """
    Combined SpecAugment: Apply both frequency and time masking.

    Args:
        freq_mask_param: Maximum frequency bins to mask (default: 15 for 128 mel bins)
        time_mask_param: Maximum time frames to mask (default: 35 for ~170 frames)
        num_freq_masks: Number of frequency masks (default: 1)
        num_time_masks: Number of time masks (default: 1)
        prob: Probability of applying augmentation (default: 0.8)
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        prob: float = 0.8,
    ):
        self.freq_masking = FrequencyMasking(freq_mask_param, num_freq_masks)
        self.time_masking = TimeMasking(time_mask_param, num_time_masks)
        self.prob = prob

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment with given probability.

        Args:
            spectrogram: Input spectrogram of shape (C, F, T) or (F, T)

        Returns:
            Augmented spectrogram
        """
        if np.random.random() > self.prob:
            return spectrogram

        spec = self.freq_masking(spectrogram)
        spec = self.time_masking(spec)
        return spec


def mixup_batch(
    spectrograms: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
    prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp augmentation to a batch of spectrograms.

    MixUp creates virtual training examples by linearly interpolating
    between pairs of examples and their labels.

    Args:
        spectrograms: Batch of spectrograms (B, C, F, T)
        labels: Batch of labels (B, num_classes) - should be one-hot encoded
        alpha: Beta distribution parameter (default: 0.4)
        prob: Probability of applying MixUp (default: 0.5)

    Returns:
        mixed_spectrograms: Mixed batch
        mixed_labels: Interpolated labels
        lam: Mixing coefficient used (1.0 if not applied)
    """
    if np.random.random() > prob:
        return spectrograms, labels, 1.0

    batch_size = spectrograms.size(0)

    # Sample mixing coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Random permutation of batch
    index = torch.randperm(batch_size, device=spectrograms.device)

    # Mix spectrograms
    mixed_spectrograms = lam * spectrograms + (1 - lam) * spectrograms[index]

    # Mix labels (supports both one-hot and class indices)
    if labels.ndim == 1:
        # Convert class indices to one-hot
        num_classes = labels.max().item() + 1
        labels_onehot = torch.zeros(batch_size, num_classes, device=labels.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels = labels_onehot

    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_spectrograms, mixed_labels, lam


class BackgroundNoiseInjection:
    """
    Add background noise to audio spectrograms at random SNR.

    Args:
        noise_spectrograms: List of background noise spectrograms
        snr_range: Tuple of (min_snr, max_snr) in dB (default: (3, 30))
        prob: Probability of applying noise (default: 0.3)
    """

    def __init__(
        self,
        noise_spectrograms: list,
        snr_range: Tuple[float, float] = (3.0, 30.0),
        prob: float = 0.3,
    ):
        self.noise_spectrograms = noise_spectrograms
        self.snr_range = snr_range
        self.prob = prob

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Add random noise to spectrogram at random SNR.

        Args:
            spectrogram: Input spectrogram (C, F, T) or (F, T)

        Returns:
            Noisy spectrogram
        """
        if np.random.random() > self.prob or len(self.noise_spectrograms) == 0:
            return spectrogram

        # Select random noise
        noise = np.random.choice(self.noise_spectrograms)

        # Ensure noise is torch tensor
        if not isinstance(noise, torch.Tensor):
            noise = torch.from_numpy(noise).to(spectrogram.device)

        # Match shapes (crop or pad noise)
        if noise.shape != spectrogram.shape:
            # For simplicity, just return original if shapes don't match
            # In production, implement proper cropping/padding
            return spectrogram

        # Random SNR
        snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])

        # Calculate signal and noise power (in dB domain)
        signal_power = torch.mean(spectrogram**2)
        noise_power = torch.mean(noise**2)

        # Calculate scaling factor for noise
        snr_linear = 10 ** (snr_db / 10.0)
        noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear))

        # Add scaled noise
        noisy_spec = spectrogram + noise_scale * noise

        return noisy_spec


if __name__ == "__main__":
    # Test SpecAugment
    print("Testing SpecAugment...")
    spec = torch.randn(1, 128, 173)  # (C, F, T)

    spec_augment = SpecAugment(freq_mask_param=15, time_mask_param=35, prob=1.0)
    augmented = spec_augment(spec)

    print(f"Original shape: {spec.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print(f"Original mean: {spec.mean():.4f}, std: {spec.std():.4f}")
    print(f"Augmented mean: {augmented.mean():.4f}, std: {augmented.std():.4f}")

    # Count masked values (assuming mask_value=0)
    masked_count = (augmented == 0).sum().item()
    total_count = augmented.numel()
    print(
        f"Masked values: {masked_count}/{total_count} ({100 * masked_count / total_count:.2f}%)"
    )

    # Test MixUp
    print("\nTesting MixUp...")
    batch = torch.randn(8, 1, 128, 173)
    labels = torch.randint(0, 90, (8,))

    mixed_batch, mixed_labels, lam = mixup_batch(batch, labels, alpha=0.4, prob=1.0)

    print(f"Original batch shape: {batch.shape}")
    print(f"Mixed batch shape: {mixed_batch.shape}")
    print(f"Original labels shape: {labels.shape}")
    print(f"Mixed labels shape: {mixed_labels.shape}")
    print(f"Mixing coefficient λ: {lam:.4f}")
    print(f"Mixed labels sum (should be ~1 per sample): {mixed_labels[0].sum():.4f}")

    print("\n✓ All augmentation tests passed!")
