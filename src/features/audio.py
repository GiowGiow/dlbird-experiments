"""MFCC feature extraction and caching for audio data.

This module extracts MFCC static, delta, and delta-delta features from audio files
and caches them as numpy arrays for efficient training.
"""

from pathlib import Path
from typing import Union, Optional
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm


def extract_mfcc_features(
    audio_path: Union[str, Path],
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    target_sr: int = 22050,
    duration: float = 3.0,
) -> Optional[np.ndarray]:
    """Extract MFCC static, delta, and delta-delta features from audio file.

    The output is a (H, W, 3) array where:
    - H = n_mfcc (number of MFCC coefficients)
    - W = number of time frames
    - 3 = [MFCC, Delta, Delta-Delta]

    Args:
        audio_path: Path to audio file
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length for STFT
        n_fft: FFT window size
        target_sr: Target sampling rate
        duration: Target duration in seconds (for length normalization)

    Returns:
        (H, W, 3) numpy array or None if extraction fails
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=target_sr, duration=duration)

        # If shorter than target duration, pad with zeros
        target_length = int(target_sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        elif len(y) > target_length:
            # Center crop
            start = (len(y) - target_length) // 2
            y = y[start : start + target_length]

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft
        )

        # Extract Delta and Delta-Delta
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack into (H, W, 3)
        features = np.stack([mfcc, mfcc_delta, mfcc_delta2], axis=-1)

        return features.astype(np.float32)

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def cache_audio_features(
    df: pd.DataFrame,
    cache_dir: Union[str, Path],
    dataset_name: str,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    target_sr: int = 22050,
    duration: float = 3.0,
) -> int:
    """Extract and cache MFCC features for all audio files in dataframe.

    Features are saved as:
    cache_dir/species/record_id.npy

    Args:
        df: DataFrame with 'file_path', 'species', and 'record_id' columns
        cache_dir: Directory to cache features
        dataset_name: Name of dataset (for logging)
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length for STFT
        n_fft: FFT window size
        target_sr: Target sampling rate
        duration: Target duration in seconds

    Returns:
        Number of successfully cached features
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Caching {dataset_name}"):
        # Get species and record ID
        species = row.get("species", "unknown")
        if "species_normalized" in row:
            species = row["species_normalized"]

        record_id = row.get("record_id", idx)
        if "sample_id" in row:
            record_id = row["sample_id"]

        # Create species directory
        species_safe = str(species).replace("/", "_").replace(" ", "_")
        species_dir = cache_dir / species_safe
        species_dir.mkdir(exist_ok=True)

        # Cache file path
        cache_file = species_dir / f"{record_id}.npy"

        # Skip if already cached
        if cache_file.exists():
            success_count += 1
            continue

        # Extract features
        audio_path = row["file_path"]
        features = extract_mfcc_features(
            audio_path=audio_path,
            n_mfcc=n_mfcc,
            hop_length=hop_length,
            n_fft=n_fft,
            target_sr=target_sr,
            duration=duration,
        )

        if features is not None:
            # Save to cache
            np.save(cache_file, features)
            success_count += 1

    return success_count


def load_cached_features(cache_path: Union[str, Path]) -> np.ndarray:
    """Load cached MFCC features.

    Args:
        cache_path: Path to cached .npy file

    Returns:
        (H, W, 3) numpy array
    """
    return np.load(cache_path)
