"""
Unit tests for Log-Mel Spectrogram extraction.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf

from src.features.log_mel_spectrogram import (
    extract_log_mel_spectrogram,
    extract_segment_with_activity_detection,
    validate_spectrogram,
)


@pytest.fixture
def dummy_audio_file():
    """Create a temporary audio file for testing."""
    # Generate 5 seconds of dummy audio (440 Hz sine wave)
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        yield Path(f.name)
    
    # Cleanup
    Path(f.name).unlink()


def test_extract_log_mel_spectrogram_shape(dummy_audio_file):
    """Test that LMS extraction produces correct shape."""
    lms = extract_log_mel_spectrogram(
        dummy_audio_file,
        sr=22050,
        n_mels=128,
        duration=4.0,
    )
    
    assert lms.ndim == 2, "LMS should be 2D array"
    assert lms.shape[0] == 128, f"Expected 128 mel bins, got {lms.shape[0]}"
    
    # For 4s audio: frames ≈ (22050*4 - 2048) / 512 + 1 ≈ 173
    assert 160 < lms.shape[1] < 190, f"Expected ~173 frames, got {lms.shape[1]}"


def test_extract_log_mel_spectrogram_values(dummy_audio_file):
    """Test that LMS values are in reasonable range."""
    lms = extract_log_mel_spectrogram(
        dummy_audio_file,
        sr=22050,
        n_mels=128,
        normalize=True,
    )
    
    # After normalization, should be roughly zero mean, unit variance
    assert abs(np.mean(lms)) < 0.1, f"Mean should be ~0, got {np.mean(lms):.4f}"
    assert 0.8 < np.std(lms) < 1.2, f"Std should be ~1, got {np.std(lms):.4f}"
    
    # No NaN or Inf values
    assert not np.isnan(lms).any(), "LMS contains NaN values"
    assert not np.isinf(lms).any(), "LMS contains infinite values"


def test_extract_log_mel_spectrogram_reproducibility(dummy_audio_file):
    """Test that extraction is reproducible."""
    lms1 = extract_log_mel_spectrogram(dummy_audio_file, duration=3.0)
    lms2 = extract_log_mel_spectrogram(dummy_audio_file, duration=3.0)
    
    np.testing.assert_array_equal(lms1, lms2, err_msg="LMS extraction not reproducible")


def test_extract_segment_with_activity_detection(dummy_audio_file):
    """Test activity detection-based extraction."""
    lms = extract_segment_with_activity_detection(
        dummy_audio_file,
        target_duration=4.0,
        sr=22050,
        n_mels=128,
    )
    
    assert lms.ndim == 2
    assert lms.shape[0] == 128
    assert 160 < lms.shape[1] < 190


def test_validate_spectrogram():
    """Test spectrogram validation."""
    # Valid spectrogram
    valid_lms = np.random.randn(128, 173)
    is_valid, msg = validate_spectrogram(valid_lms, expected_mels=128)
    assert is_valid, f"Valid LMS rejected: {msg}"
    
    # Invalid: wrong number of mel bins
    invalid_lms = np.random.randn(64, 173)
    is_valid, msg = validate_spectrogram(invalid_lms, expected_mels=128)
    assert not is_valid, "Invalid LMS accepted (wrong mel bins)"
    
    # Invalid: too few frames
    invalid_lms = np.random.randn(128, 50)
    is_valid, msg = validate_spectrogram(invalid_lms, expected_mels=128)
    assert not is_valid, "Invalid LMS accepted (too few frames)"
    
    # Invalid: contains NaN
    invalid_lms = np.random.randn(128, 173)
    invalid_lms[0, 0] = np.nan
    is_valid, msg = validate_spectrogram(invalid_lms, expected_mels=128)
    assert not is_valid, "Invalid LMS accepted (contains NaN)"


def test_different_durations(dummy_audio_file):
    """Test extraction with different duration parameters."""
    durations = [3.0, 4.0, 5.0]
    expected_frames = {
        3.0: (120, 140),  # ~130 frames
        4.0: (160, 190),  # ~173 frames
        5.0: (200, 230),  # ~216 frames
    }
    
    for dur in durations:
        lms = extract_log_mel_spectrogram(dummy_audio_file, duration=dur)
        assert lms.shape[0] == 128
        min_frames, max_frames = expected_frames[dur]
        assert min_frames < lms.shape[1] < max_frames, \
            f"Duration {dur}s: expected {min_frames}-{max_frames} frames, got {lms.shape[1]}"


def test_segment_start_parameter(dummy_audio_file):
    """Test extraction from specific time offset."""
    # Extract from beginning
    lms_start = extract_log_mel_spectrogram(
        dummy_audio_file, duration=2.0, segment_start=0.0
    )
    
    # Extract from middle
    lms_middle = extract_log_mel_spectrogram(
        dummy_audio_file, duration=2.0, segment_start=1.5
    )
    
    # Should have same shape but different values
    assert lms_start.shape == lms_middle.shape
    assert not np.allclose(lms_start, lms_middle), \
        "Different segments should produce different spectrograms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
