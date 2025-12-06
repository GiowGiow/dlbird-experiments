"""
Log-Mel Spectrogram extraction for audio classification.

Replaces MFCC features with dense, image-like spectrograms suitable for
2D deep learning architectures (CNNs, Transformers).
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings


def extract_log_mel_spectrogram(
    audio_path: Path,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    duration: Optional[float] = None,
    segment_start: Optional[float] = None,
    normalize: bool = True,
    apply_highpass: bool = True,
    highpass_cutoff: float = 500.0,
) -> np.ndarray:
    """
    Extract Log-Mel Spectrogram from audio file.
    
    Parameters
    ----------
    audio_path : Path
        Path to audio file
    sr : int, default=22050
        Target sample rate (Hz)
    n_mels : int, default=128
        Number of mel frequency bins
    n_fft : int, default=2048
        FFT window size
    hop_length : int, default=512
        Number of samples between successive frames
    duration : float, optional
        Duration of audio segment to extract (seconds)
        If None, uses entire audio
    segment_start : float, optional
        Start time of segment (seconds)
        If None, extracts from beginning
    normalize : bool, default=True
        Apply zero mean, unit variance normalization
    apply_highpass : bool, default=True
        Apply high-pass filter to remove low-frequency noise
    highpass_cutoff : float, default=500.0
        High-pass filter cutoff frequency (Hz)
        
    Returns
    -------
    log_mel_spec : np.ndarray
        Log-Mel Spectrogram with shape (n_mels, time_frames)
        Typically (128, 130-216) for 3-5 second audio clips
        
    Notes
    -----
    Output shape depends on audio duration:
    - 3s audio → ~130 frames
    - 4s audio → ~173 frames  
    - 5s audio → ~216 frames
    
    Formula: time_frames = ceil((sr * duration - n_fft) / hop_length) + 1
    """
    # Load audio
    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration, offset=segment_start or 0.0)
    except Exception as e:
        raise ValueError(f"Failed to load audio from {audio_path}: {e}")
    
    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")
    
    # Apply high-pass filter to remove low-frequency noise (< 500 Hz)
    # Bird vocalizations are typically > 1 kHz
    if apply_highpass:
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        # Simple high-pass via scipy if available, else skip
        try:
            from scipy.signal import butter, filtfilt
            nyq = sr / 2
            cutoff_norm = highpass_cutoff / nyq
            if cutoff_norm < 1.0:
                b, a = butter(5, cutoff_norm, btype='high')
                audio = filtfilt(b, a, audio)
        except ImportError:
            warnings.warn("scipy not available, skipping precise high-pass filter")
    
    # Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=500.0,  # Minimum frequency (Hz) - skip low frequencies
        fmax=sr // 2,  # Maximum frequency (Nyquist)
        power=2.0,  # Power spectrogram (energy)
    )
    
    # Convert power to dB scale (log compression)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
    
    # Normalize to zero mean, unit variance
    if normalize:
        mean = np.mean(log_mel_spec)
        std = np.std(log_mel_spec)
        if std > 0:
            log_mel_spec = (log_mel_spec - mean) / std
        else:
            # Avoid division by zero for silent audio
            log_mel_spec = log_mel_spec - mean
    
    return log_mel_spec


def extract_segment_with_activity_detection(
    audio_path: Path,
    target_duration: float = 4.0,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    normalize: bool = True,
    top_db: float = 30.0,
) -> np.ndarray:
    """
    Extract Log-Mel Spectrogram from segment with highest audio activity.
    
    Uses energy-based activity detection to identify the most informative
    segment within a longer recording.
    
    Parameters
    ----------
    audio_path : Path
        Path to audio file
    target_duration : float, default=4.0
        Target duration for extracted segment (seconds)
    sr : int, default=22050
        Target sample rate (Hz)
    n_mels : int, default=128
        Number of mel frequency bins
    n_fft : int, default=2048
        FFT window size
    hop_length : int, default=512
        Number of samples between successive frames
    normalize : bool, default=True
        Apply zero mean, unit variance normalization
    top_db : float, default=30.0
        Threshold for non-silent intervals (dB below peak)
        
    Returns
    -------
    log_mel_spec : np.ndarray
        Log-Mel Spectrogram with shape (n_mels, time_frames)
        
    Notes
    -----
    Algorithm:
    1. Load full audio
    2. Detect non-silent intervals
    3. If multiple intervals exist, select segment with highest energy
    4. If audio shorter than target_duration, pad with zeros
    5. Extract Log-Mel Spectrogram from selected segment
    """
    # Load full audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")
    
    audio_duration = len(audio) / sr
    
    # If audio shorter than target, pad and use entire audio
    if audio_duration <= target_duration:
        target_samples = int(target_duration * sr)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        return extract_log_mel_spectrogram(
            audio_path, sr=sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, duration=target_duration, normalize=normalize
        )
    
    # Detect non-silent intervals
    intervals = librosa.effects.split(audio, top_db=top_db)
    
    if len(intervals) == 0:
        # No activity detected, use beginning
        return extract_log_mel_spectrogram(
            audio_path, sr=sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, duration=target_duration, normalize=normalize
        )
    
    # Find segment with highest energy
    target_samples = int(target_duration * sr)
    best_start = 0
    best_energy = 0.0
    
    for start_idx, end_idx in intervals:
        # Check if this interval is long enough
        if end_idx - start_idx >= target_samples:
            # Extract segment and compute energy
            segment = audio[start_idx:start_idx + target_samples]
            energy = np.sum(segment ** 2)
            if energy > best_energy:
                best_energy = energy
                best_start = start_idx
    
    # If no single interval long enough, use the longest one
    if best_energy == 0:
        longest_interval = max(intervals, key=lambda x: x[1] - x[0])
        best_start = longest_interval[0]
    
    # Convert sample index to time
    start_time = best_start / sr
    
    # Extract Log-Mel Spectrogram from selected segment
    return extract_log_mel_spectrogram(
        audio_path, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, duration=target_duration,
        segment_start=start_time, normalize=normalize
    )


def validate_spectrogram(log_mel_spec: np.ndarray, expected_mels: int = 128) -> Tuple[bool, str]:
    """
    Validate Log-Mel Spectrogram shape and value ranges.
    
    Parameters
    ----------
    log_mel_spec : np.ndarray
        Log-Mel Spectrogram to validate
    expected_mels : int, default=128
        Expected number of mel bins
        
    Returns
    -------
    is_valid : bool
        Whether spectrogram is valid
    message : str
        Validation message
    """
    if log_mel_spec.ndim != 2:
        return False, f"Expected 2D array, got {log_mel_spec.ndim}D"
    
    if log_mel_spec.shape[0] != expected_mels:
        return False, f"Expected {expected_mels} mel bins, got {log_mel_spec.shape[0]}"
    
    if log_mel_spec.shape[1] < 100:
        return False, f"Too few time frames: {log_mel_spec.shape[1]} (expected ≥100)"
    
    if log_mel_spec.shape[1] > 300:
        return False, f"Too many time frames: {log_mel_spec.shape[1]} (expected ≤300)"
    
    if np.isnan(log_mel_spec).any():
        return False, "Contains NaN values"
    
    if np.isinf(log_mel_spec).any():
        return False, "Contains infinite values"
    
    return True, f"Valid: shape {log_mel_spec.shape}"


def visualize_spectrogram(
    log_mel_spec: np.ndarray,
    title: str = "Log-Mel Spectrogram",
    sr: int = 22050,
    hop_length: int = 512,
    save_path: Optional[Path] = None,
):
    """
    Visualize Log-Mel Spectrogram with proper time/frequency axes.
    
    Parameters
    ----------
    log_mel_spec : np.ndarray
        Log-Mel Spectrogram with shape (n_mels, time_frames)
    title : str
        Plot title
    sr : int, default=22050
        Sample rate used for extraction
    hop_length : int, default=512
        Hop length used for extraction
    save_path : Path, optional
        If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create time axis
    n_frames = log_mel_spec.shape[1]
    times = np.arange(n_frames) * hop_length / sr
    
    # Display spectrogram
    img = librosa.display.specshow(
        log_mel_spec,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        hop_length=hop_length,
        ax=ax,
        cmap='viridis'
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Mel Frequency', fontsize=12)
    
    # Add colorbar
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram to {save_path}")
    else:
        plt.show()
    
    plt.close()
