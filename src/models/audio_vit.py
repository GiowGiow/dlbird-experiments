"""ViT adapter for audio MFCC features."""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
import torch.nn.functional as F


class AudioViT(nn.Module):
    """ViT-B/16 adapted for MFCC input.

    Treats (H, W, 3) MFCC stacks as images:
    - Resizes to 224x224
    - Uses pretrained ViT-B/16
    - Fine-tunes classification head
    """

    def __init__(
        self, num_classes: int, pretrained: str = "google/vit-base-patch16-224"
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Pretrained model identifier
        """
        super().__init__()

        # Load pretrained ViT
        self.vit = ViTForImageClassification.from_pretrained(
            pretrained, num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) tensor - will be resized to 224x224

        Returns:
            logits: (B, num_classes) tensor
        """
        # Resize to 224x224 for ViT
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Forward through ViT
        outputs = self.vit(pixel_values=x)
        return outputs.logits
