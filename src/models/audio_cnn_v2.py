"""AudioCNN v2 - Increased capacity architecture for audio MFCC features.

This model provides significantly more capacity than AudioCNN to handle
89-class bird species classification with severe class imbalance.

Key improvements over AudioCNN:
- Wider channels: [64, 128, 256, 512] vs [32, 64, 128]
- 5 conv blocks vs 3 conv blocks
- Larger FC layer: 512 vs 256 hidden units
- ~1M parameters vs ~343K parameters
"""

import torch
import torch.nn as nn


class AudioCNNv2(nn.Module):
    """High-capacity ConvNet for 3-channel MFCC input.

    Architecture:
    - Input: (3, H, W) where H=20 (MFCC coeffs), W=time frames (~500)
    - Block 1: 3 -> 64 channels
    - Block 2: 64 -> 128 channels
    - Block 3: 128 -> 256 channels
    - Block 4: 256 -> 512 channels
    - Block 5: 512 -> 512 channels (deeper)
    - AdaptiveAvgPool
    - FC(512) + Dropout + FC(num_classes)

    Target params: ~1M (sufficient for 89 classes with 10K params/class rule)
    """

    def __init__(self, num_classes: int, dropout: float = 0.5):
        """
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability for FC layers
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 64 channels
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 256 -> 512 channels
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 5: 512 -> 512 channels (deeper network)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) tensor where B=batch, H=20 MFCC, W~=500 frames

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


if __name__ == "__main__":
    # Test architecture
    model = AudioCNNv2(num_classes=89)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"AudioCNNv2 Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(4, 3, 20, 500)  # batch=4, channels=3, H=20, W=500
    y = model(x)
    print(f"\nForward pass test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Expected output shape: (4, 89)")

    # Verify output shape
    assert y.shape == (4, 89), f"Output shape mismatch: {y.shape} vs (4, 89)"
    print("\n✓ Architecture test passed!")

    # Compare with AudioCNN
    from audio_cnn import AudioCNN

    old_model = AudioCNN(num_classes=89)
    old_params = sum(p.numel() for p in old_model.parameters())

    print(f"\nComparison:")
    print(f"  AudioCNN:   {old_params:,} parameters")
    print(f"  AudioCNNv2: {total_params:,} parameters")
    print(
        f"  Increase:   {total_params / old_params:.2f}x ({total_params - old_params:,} more)"
    )
