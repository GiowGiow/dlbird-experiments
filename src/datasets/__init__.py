"""PyTorch dataset classes."""

from src.datasets.audio import AudioMFCCDataset
from src.datasets.audio_spectrogram import AudioSpectrogramDataset, collate_spectrograms
from src.datasets.image import ImageDataset

__all__ = ['AudioMFCCDataset', 'AudioSpectrogramDataset', 'collate_spectrograms', 'ImageDataset']
