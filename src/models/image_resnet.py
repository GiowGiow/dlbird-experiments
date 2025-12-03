"""ResNet model for image classification."""

import torch
import torch.nn as nn
from torchvision import models


class ImageResNet(nn.Module):
    """ResNet-18 for bird image classification.

    Fine-tunes pretrained ResNet-18 with custom classification head.
    """

    def __init__(
        self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Load pretrained ImageNet weights
            freeze_backbone: If True, freeze all layers except final block and head
        """
        super().__init__()

        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Replace final FC layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

        # Optionally freeze backbone (keep only last block + head trainable)
        if freeze_backbone:
            for name, param in self.resnet.named_parameters():
                if not ("layer4" in name or "fc" in name):
                    param.requires_grad = False

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) image tensor

        Returns:
            logits: (B, num_classes) tensor
        """
        return self.resnet(x)


class ImageResNet50(nn.Module):
    """ResNet-50 for bird image classification."""

    def __init__(
        self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Load pretrained ImageNet weights
            freeze_backbone: If True, freeze all layers except final block and head
        """
        super().__init__()

        self.resnet = models.resnet50(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.resnet.named_parameters():
                if not ("layer4" in name or "fc" in name):
                    param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)
