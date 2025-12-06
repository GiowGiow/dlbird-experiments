"""
PyTorch Dataset for Log-Mel Spectrogram-based audio classification.

Replaces MFCC features with dense, image-like spectrograms for 2D deep learning.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings

from src.features.log_mel_spectrogram import extract_segment_with_activity_detection


class AudioSpectrogramDataset(Dataset):
    """
    Dataset for audio classification using Log-Mel Spectrograms.
    
    Features:
    - Extracts Log-Mel Spectrograms (128, T) from audio files
    - Caches spectrograms for fast loading
    - Activity detection to select most informative segments
    - Compatible with 2D CNNs and Transformers (AST, EfficientNet)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing audio metadata with columns:
        - 'species_normalized': species label
        - 'record_id': unique recording ID
    cache_dir : Path
        Directory for cached spectrograms (saves as .npy files)
    split : List[int]
        List of indices for this split (train/val/test)
    species_to_idx : Dict[str, int]
        Mapping from species name to class index
    target_duration : float, default=4.0
        Target duration for audio segments (seconds)
    sr : int, default=22050
        Sample rate (Hz)
    n_mels : int, default=128
        Number of mel frequency bins
    n_fft : int, default=2048
        FFT window size
    hop_length : int, default=512
        Hop length for STFT
    normalize : bool, default=True
        Apply zero mean, unit variance normalization
    use_activity_detection : bool, default=True
        Use energy-based activity detection to select segments
    regenerate_cache : bool, default=False
        Force regenerate cache even if files exist
    augment : Optional[callable], default=None
        Augmentation function to apply to spectrograms (e.g., SpecAugment)
        Should accept torch.Tensor and return torch.Tensor
        
    Notes
    -----
    Output spectrograms have shape (128, T) where:
    - 128 = number of mel frequency bins
    - T = number of time frames (~173 for 4s audio)
    
    Cache file naming: {cache_dir}/{species}/{xeno_canto_id}.npy
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Path,
        split: List[int],
        species_to_idx: Dict[str, int],
        target_duration: float = 4.0,
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        normalize: bool = True,
        use_activity_detection: bool = True,
        regenerate_cache: bool = False,
        augment: Optional[callable] = None,
    ):
        self.df = df.iloc[split].reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.species_to_idx = species_to_idx
        self.target_duration = target_duration
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize
        self.use_activity_detection = use_activity_detection
        self.regenerate_cache = regenerate_cache
        self.augment = augment
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate species mapping
        missing_species = set(self.df['species_normalized'].unique()) - set(species_to_idx.keys())
        if missing_species:
            raise ValueError(f"Species not in mapping: {missing_species}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for a given index."""
        row = self.df.iloc[idx]
        species = row['species_normalized']
        record_id = row['record_id']
        
        # Create species subdirectory
        species_dir = self.cache_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        
        return species_dir / f"{record_id}.npy"
    
    def _get_audio_path(self, idx: int) -> Path:
        """Get audio file path for a given index."""
        row = self.df.iloc[idx]
        
        # Try common audio storage locations
        possible_paths = [
            Path(row.get('file_path', '')),  # Use file_path from df (most reliable)
            Path('data/xeno_canto_raw') / f"{row['record_id']}.mp3",
            Path('data/xeno_canto') / f"{row['record_id']}.mp3",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # If no path exists, raise error
        raise FileNotFoundError(
            f"Audio file not found for record_id {row['record_id']}. "
            f"Tried: {[str(p) for p in possible_paths]}"
        )
    
    def _extract_and_cache_spectrogram(self, idx: int) -> np.ndarray:
        """Extract Log-Mel Spectrogram and cache it."""
        audio_path = self._get_audio_path(idx)
        cache_path = self._get_cache_path(idx)
        
        # Extract spectrogram
        try:
            if self.use_activity_detection:
                lms = extract_segment_with_activity_detection(
                    audio_path,
                    target_duration=self.target_duration,
                    sr=self.sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    normalize=self.normalize,
                )
            else:
                from src.features.log_mel_spectrogram import extract_log_mel_spectrogram
                lms = extract_log_mel_spectrogram(
                    audio_path,
                    sr=self.sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    duration=self.target_duration,
                    normalize=self.normalize,
                )
        except Exception as e:
            warnings.warn(f"Failed to extract LMS from {audio_path}: {e}")
            # Return zeros as fallback
            expected_frames = int((self.sr * self.target_duration - self.n_fft) / self.hop_length) + 1
            lms = np.zeros((self.n_mels, expected_frames), dtype=np.float32)
        
        # Save to cache
        try:
            np.save(cache_path, lms)
        except Exception as e:
            warnings.warn(f"Failed to cache LMS to {cache_path}: {e}")
        
        return lms
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get spectrogram and label for a given index.
        
        Returns
        -------
        spectrogram : torch.Tensor
            Log-Mel Spectrogram with shape (1, n_mels, time_frames)
            Channel dimension added for compatibility with 2D CNNs
            Augmentation applied if self.augment is set
        label : int
            Species class index
        """
        row = self.df.iloc[idx]
        label = self.species_to_idx[row['species_normalized']]
        
        cache_path = self._get_cache_path(idx)
        
        # Load from cache if exists, otherwise extract
        if cache_path.exists() and not self.regenerate_cache:
            try:
                lms = np.load(cache_path)
            except Exception as e:
                warnings.warn(f"Failed to load cached LMS from {cache_path}: {e}")
                lms = self._extract_and_cache_spectrogram(idx)
        else:
            lms = self._extract_and_cache_spectrogram(idx)
        
        # Convert to tensor and add channel dimension
        # Shape: (1, n_mels, time_frames)
        spectrogram = torch.from_numpy(lms).float().unsqueeze(0)
        
        # Apply augmentation if provided (only during training)
        if self.augment is not None:
            spectrogram = self.augment(spectrogram)
        
        return spectrogram, label
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached spectrograms."""
        cached_count = sum(1 for idx in range(len(self)) if self._get_cache_path(idx).exists())
        return {
            'total_samples': len(self),
            'cached_samples': cached_count,
            'missing_samples': len(self) - cached_count,
            'cache_coverage': cached_count / len(self) if len(self) > 0 else 0.0,
        }


def collate_spectrograms(batch):
    """
    Custom collate function for variable-length spectrograms.
    
    Pads spectrograms to the maximum time length in the batch.
    
    Parameters
    ----------
    batch : List[Tuple[torch.Tensor, int]]
        List of (spectrogram, label) tuples
        
    Returns
    -------
    spectrograms : torch.Tensor
        Batch of spectrograms with shape (B, 1, n_mels, max_time)
    labels : torch.Tensor
        Batch of labels with shape (B,)
    """
    spectrograms, labels = zip(*batch)
    
    # Find maximum time dimension
    max_time = max(spec.shape[-1] for spec in spectrograms)
    
    # Pad spectrograms to max_time
    padded_specs = []
    for spec in spectrograms:
        if spec.shape[-1] < max_time:
            pad_size = max_time - spec.shape[-1]
            # Pad on the right with zeros
            spec = torch.nn.functional.pad(spec, (0, pad_size), mode='constant', value=0)
        padded_specs.append(spec)
    
    # Stack into batch
    spectrograms_batch = torch.stack(padded_specs, dim=0)
    labels_batch = torch.tensor(labels, dtype=torch.long)
    
    return spectrograms_batch, labels_batch
