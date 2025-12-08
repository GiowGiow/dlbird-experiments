"""Model architectures."""

from .audio_cnn import AudioCNN
from .audio_cnn_v2 import AudioCNNv2
from .audio_vit import AudioViT
from .audio_ast import AudioAST, AudioASTSmall

__all__ = ["AudioCNN", "AudioCNNv2", "AudioViT", "AudioAST", "AudioASTSmall"]
