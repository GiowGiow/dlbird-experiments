"""Compact CNN architecture for audio MFCC features.

This model processes 3-channel MFCC stacks (static, delta, delta-delta)
with a lightweight convolutional architecture suitable for fast training.
"""

import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    """Compact ConvNet for 3-channel MFCC input.

    Architecture:
    - Input: (3, H, W) where H=40 (MFCC coeffs), W=time frames
    - 2x Conv blocks (32 filters)
    - MaxPool
    - 2x Conv blocks (64 filters)
    - MaxPool
    - 2x Conv blocks (128 filters)
    - AdaptiveAvgPool
    - FC(256) + Dropout + FC(num_classes)

    Estimated params: ~1-5M
    """

    def __init__(self, num_classes: int, dropout: float = 0.5):
        """
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability for FC layer
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) tensor

        Returns:
            logits: (B, num_classes) tensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
