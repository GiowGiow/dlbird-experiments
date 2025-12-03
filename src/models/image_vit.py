"""ViT model for image classification."""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification


class ImageViT(nn.Module):
    """ViT-B/16 for bird image classification.

    Fine-tunes pretrained Vision Transformer.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: str = "google/vit-base-patch16-224",
        freeze_backbone: bool = False,
        num_unfreeze_blocks: int = 2,
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Pretrained model identifier
            freeze_backbone: If True, freeze encoder except last N blocks
            num_unfreeze_blocks: Number of encoder blocks to keep trainable
        """
        super().__init__()

        # Load pretrained ViT
        self.vit = ViTForImageClassification.from_pretrained(
            pretrained, num_labels=num_classes, ignore_mismatched_sizes=True
        )

        # Optionally freeze backbone
        if freeze_backbone:
            # Freeze all encoder layers
            for param in self.vit.vit.encoder.parameters():
                param.requires_grad = False

            # Unfreeze last N blocks
            num_layers = len(self.vit.vit.encoder.layer)
            for i in range(num_layers - num_unfreeze_blocks, num_layers):
                for param in self.vit.vit.encoder.layer[i].parameters():
                    param.requires_grad = True

            # Keep classifier head trainable
            for param in self.vit.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, 224, 224) image tensor

        Returns:
            logits: (B, num_classes) tensor
        """
        outputs = self.vit(pixel_values=x)
        return outputs.logits
