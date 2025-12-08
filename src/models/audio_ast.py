"""
Audio Spectrogram Transformer (AST) for bird species classification.

Implements AST with AudioSet pretraining for transfer learning on bird vocalizations.
"""

import torch
import torch.nn as nn
from transformers import ASTForAudioClassification, ASTConfig
from typing import Optional
import warnings


class AudioAST(nn.Module):
    """
    Audio Spectrogram Transformer wrapper for bird species classification.

    Uses pretrained AST model from MIT with AudioSet weights for transfer learning.
    The model processes Log-Mel Spectrograms (128, T) as image-like inputs.

    Parameters
    ----------
    num_classes : int
        Number of bird species classes (90 for Xeno-Canto dataset)
    pretrained_model : str, default='MIT/ast-finetuned-audioset-10-10-0.4593'
        Hugging Face model identifier for pretrained AST
    freeze_backbone : bool, default=False
        Whether to freeze backbone parameters (only train classification head)
    dropout : float, default=0.1
        Dropout probability for classification head

    Notes
    -----
    Architecture:
    - Input: (B, 1, 128, T) spectrograms
    - Backbone: AST with 12 transformer layers (86M parameters)
    - Head: Linear(768, num_classes)
    - Output: (B, num_classes) logits

    The backbone is pretrained on AudioSet (2M audio clips, 527 classes),
    providing strong representations for audio classification tasks.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # Load pretrained AST model
        try:
            self.ast = ASTForAudioClassification.from_pretrained(
                pretrained_model,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,  # Replace head for our num_classes
            )
        except Exception as e:
            warnings.warn(
                f"Failed to load pretrained AST: {e}. Using random initialization."
            )
            config = ASTConfig(num_labels=num_classes)
            self.ast = ASTForAudioClassification(config)

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

        # Add dropout to classification head if needed
        if dropout > 0.0:
            # Replace classifier with dropout version
            hidden_size = self.ast.config.hidden_size
            self.ast.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )

    def _freeze_backbone(self):
        """Freeze all parameters except classification head."""
        for name, param in self.ast.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        print(f"Froze backbone parameters. Only training classification head.")

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.ast.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print(f"Unfroze all parameters. Training full model.")

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through AST model.

        Parameters
        ----------
        spectrograms : torch.Tensor
            Input spectrograms with shape (B, 1, n_mels, time_frames)
            Typically (B, 1, 128, 173) for 4-second audio clips

        Returns
        -------
        logits : torch.Tensor
            Classification logits with shape (B, num_classes)
        """
        # AST expects input shape (B, n_mels, time_frames)
        # Remove channel dimension if present
        if spectrograms.dim() == 4 and spectrograms.shape[1] == 1:
            spectrograms = spectrograms.squeeze(1)

        # AST expects max_length=1024 frames, pad if needed
        # Our spectrograms are ~173 frames, so pad to 1024
        max_length = 1024
        current_length = spectrograms.shape[-1]

        if current_length < max_length:
            # Pad on the right with zeros
            pad_size = max_length - current_length
            spectrograms = torch.nn.functional.pad(
                spectrograms, (0, pad_size), mode="constant", value=0
            )
        elif current_length > max_length:
            # Truncate if somehow longer than expected
            spectrograms = spectrograms[:, :, :max_length]

        # Forward through AST
        outputs = self.ast(spectrograms)
        logits = outputs.logits

        return logits

    def get_param_groups(self, backbone_lr: float = 5e-5, head_lr: float = 1e-3):
        """
        Get parameter groups with different learning rates for backbone and head.

        This enables discriminative learning rates:
        - Backbone (pretrained): lower LR to preserve learned features
        - Head (random init): higher LR for faster adaptation

        Parameters
        ----------
        backbone_lr : float, default=5e-5
            Learning rate for backbone parameters
        head_lr : float, default=1e-3
            Learning rate for classification head

        Returns
        -------
        param_groups : list
            List of parameter groups for optimizer
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if "classifier" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": head_lr, "name": "head"},
        ]

        return param_groups

    def count_parameters(self) -> dict:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
        }


class AudioASTSmall(AudioAST):
    """
    Smaller AST variant for faster training (if needed).

    Uses AST with reduced layers/dimensions as fallback if GPU memory limited.
    """

    def __init__(self, num_classes: int, **kwargs):
        # Use base AST config but with fewer layers
        super().__init__(num_classes, pretrained_model=None, **kwargs)

        # Configure smaller AST
        config = ASTConfig(
            num_labels=num_classes,
            num_hidden_layers=6,  # Reduced from 12
            hidden_size=384,  # Reduced from 768
            num_attention_heads=6,
        )
        self.ast = ASTForAudioClassification(config)


def test_ast_model():
    """Test AST model forward pass and shapes."""
    print("=" * 80)
    print("AST Model Test")
    print("=" * 80)

    # Create model
    model = AudioAST(num_classes=90)
    params = model.count_parameters()

    print(f"\nModel parameters:")
    print(f"  Total:     {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen:    {params['frozen']:,}")

    # Test forward pass
    batch_size = 4
    n_mels = 128
    time_frames = 173

    # Create dummy input
    x = torch.randn(batch_size, 1, n_mels, time_frames)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(x)

    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, 90)")

    assert logits.shape == (batch_size, 90), f"Shape mismatch: {logits.shape}"

    print(f"\n✅ AST model test passed!")

    # Test parameter groups
    param_groups = model.get_param_groups(backbone_lr=5e-5, head_lr=1e-3)
    print(f"\nParameter groups:")
    for group in param_groups:
        n_params = sum(p.numel() for p in group["params"])
        print(f"  {group['name']:10s}: {n_params:,} params, LR={group['lr']}")

    print("=" * 80)


if __name__ == "__main__":
    test_ast_model()
