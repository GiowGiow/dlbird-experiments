"""
Custom loss functions for handling class imbalance in multi-class classification.

This module implements focal loss, an advanced loss function designed to address
severe class imbalance by dynamically down-weighting well-classified examples
and focusing training on hard examples.

Focal Loss Theory
-----------------
Focal Loss (Lin et al., 2017) modifies the standard cross-entropy loss to reduce
the contribution of easy examples and focus on hard, misclassified examples.

Formula:
    FL(p_t) = -α_t(1 - p_t)^γ * log(p_t)

Where:
    - p_t: Predicted probability for the true class
    - α_t: Class-specific weighting factor (optional, typically 0.25-0.75)
    - γ: Focusing parameter (typically 2.0)

Key Properties:
    - When γ=0, focal loss reduces to standard cross-entropy loss
    - When p_t→1 (confident correct prediction): (1-p_t)^γ→0, loss→0
    - When p_t→0 (incorrect prediction): (1-p_t)^γ→1, loss remains high
    - Higher γ increases focus on hard examples

Advantages over Class Weights:
    1. Sample-specific adaptation (not just class-based)
    2. Smooth, continuous down-weighting (no discrete weight spikes)
    3. Better gradient flow in early training
    4. Proven effective for extreme imbalance (1000:1+)

References:
    Lin, T. Y., et al. (2017). "Focal loss for dense object detection."
    ICCV 2017. https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with extreme class imbalance.
    
    This loss function addresses class imbalance by down-weighting the loss
    contribution from easy examples and focusing on hard, misclassified examples.
    
    Args:
        gamma (float): Focusing parameter. Higher values increase focus on hard examples.
                      Recommended: 2.0 (from original paper)
                      Range: [0.0, 5.0], where gamma=0 is equivalent to CrossEntropyLoss
        alpha (float, optional): Class weighting factor. If None, no class weighting is applied.
                                If float, applies uniform weighting to all classes.
                                Recommended: 0.25 for binary, 0.25-0.5 for multi-class
                                Range: [0.0, 1.0]
        reduction (str): Specifies the reduction to apply to the output:
                        'none' | 'mean' | 'sum'. Default: 'mean'
    
    Shape:
        - Input: (N, C) where N is batch size, C is number of classes
        - Target: (N,) where each value is 0 ≤ target[i] ≤ C-1
        - Output: scalar if reduction='mean' or 'sum', (N,) if reduction='none'
    
    Examples:
        >>> # Standard focal loss with gamma=2.0, no alpha weighting
        >>> loss_fn = FocalLoss(gamma=2.0)
        >>> logits = torch.randn(32, 89)  # batch_size=32, num_classes=89
        >>> targets = torch.randint(0, 89, (32,))
        >>> loss = loss_fn(logits, targets)
        
        >>> # Focal loss with alpha weighting
        >>> loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        >>> loss = loss_fn(logits, targets)
        
        >>> # Test equivalence to CrossEntropyLoss when gamma=0
        >>> focal_loss = FocalLoss(gamma=0.0)
        >>> ce_loss = nn.CrossEntropyLoss()
        >>> # focal_loss(logits, targets) ≈ ce_loss(logits, targets)
    
    Notes:
        - Numerical stability is ensured by clamping probabilities to [1e-7, 1-1e-7]
        - Works with any number of classes (tested with 89 classes)
        - Compatible with automatic mixed precision (AMP) training
        - Gradients are stable and do not explode for reasonable gamma values (≤5.0)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        
        # Validate gamma parameter
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0.0, got {gamma}")
        if gamma > 10.0:
            raise ValueError(f"gamma > 10.0 may cause numerical instability, got {gamma}")
        
        # Validate alpha parameter
        if alpha is not None:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        
        # Validate reduction parameter
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
        
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits from model (before softmax), shape (N, C)
            targets: Ground truth class indices, shape (N,)
        
        Returns:
            Focal loss value (scalar if reduction='mean'/'sum', shape (N,) if 'none')
        """
        # Input validation
        if inputs.dim() != 2:
            raise ValueError(f"inputs must be 2D (N, C), got shape {inputs.shape}")
        if targets.dim() != 1:
            raise ValueError(f"targets must be 1D (N,), got shape {targets.shape}")
        if inputs.size(0) != targets.size(0):
            raise ValueError(f"batch size mismatch: inputs {inputs.size(0)} vs targets {targets.size(0)}")
        
        num_classes = inputs.size(1)
        if targets.min() < 0 or targets.max() >= num_classes:
            raise ValueError(f"targets must be in [0, {num_classes-1}], got range [{targets.min()}, {targets.max()}]")
        
        # Compute softmax probabilities
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Gather probabilities for true classes
        # Shape: (N,)
        probs_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_probs_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Clamp probabilities for numerical stability
        # Prevents log(0) and (1-1)^gamma
        probs_t = torch.clamp(probs_t, min=1e-7, max=1.0 - 1e-7)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1.0 - probs_t) ** self.gamma
        
        # Compute focal loss: -alpha * focal_weight * log(p_t)
        focal_loss = -focal_weight * log_probs_t
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        # Apply reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
    
    def __repr__(self):
        return (f"FocalLoss(gamma={self.gamma}, alpha={self.alpha}, "
                f"reduction='{self.reduction}')")
