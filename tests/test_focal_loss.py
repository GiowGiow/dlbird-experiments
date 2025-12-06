"""
Unit tests for FocalLoss implementation.

Tests cover:
1. Forward pass correctness
2. Gradient flow stability (no NaN/inf)
3. Equivalence to CrossEntropyLoss when gamma=0
4. Numerical stability with extreme probabilities
5. Input validation
"""

import torch
import torch.nn as nn
import pytest
from src.training.losses import FocalLoss


class TestFocalLossForwardPass:
    """Test FocalLoss forward pass computation."""
    
    def test_forward_pass_shape(self):
        """Test that output shape is correct."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(32, 89)  # batch_size=32, num_classes=89
        targets = torch.randint(0, 89, (32,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.dim() == 0, "Loss should be scalar with reduction='mean'"
        assert loss.item() > 0, "Loss should be positive"
    
    def test_forward_pass_no_reduction(self):
        """Test forward pass with reduction='none'."""
        loss_fn = FocalLoss(gamma=2.0, reduction='none')
        logits = torch.randn(32, 89)
        targets = torch.randint(0, 89, (32,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.shape == (32,), f"Expected shape (32,), got {loss.shape}"
        assert (loss > 0).all(), "All losses should be positive"
    
    def test_forward_pass_with_alpha(self):
        """Test forward pass with alpha weighting."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        logits = torch.randn(32, 89)
        targets = torch.randint(0, 89, (32,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.dim() == 0
        assert loss.item() > 0
    
    def test_focal_weight_behavior(self):
        """Test that focal weight down-weights easy examples."""
        loss_fn = FocalLoss(gamma=2.0, reduction='none')
        
        # Create logits where target class has high probability (easy example)
        logits_easy = torch.zeros(1, 10)
        logits_easy[0, 0] = 10.0  # Very confident for class 0
        targets_easy = torch.tensor([0])
        
        # Create logits where target class has low probability (hard example)
        logits_hard = torch.zeros(1, 10)
        logits_hard[0, 0] = -10.0  # Very low confidence for class 0
        targets_hard = torch.tensor([0])
        
        loss_easy = loss_fn(logits_easy, targets_easy)
        loss_hard = loss_fn(logits_hard, targets_hard)
        
        # Hard example should have much higher loss
        assert loss_hard > loss_easy, "Hard example should have higher loss"
        # Easy example should have very low loss (nearly 0)
        assert loss_easy < 0.1, f"Easy example loss should be near 0, got {loss_easy.item()}"


class TestFocalLossGradientFlow:
    """Test gradient flow stability."""
    
    def test_gradients_no_nan_inf(self):
        """Test that gradients don't contain NaN or inf values."""
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(32, 89, requires_grad=True)
        targets = torch.randint(0, 89, (32,))
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert not torch.isnan(logits.grad).any(), "Gradients contain NaN"
        assert not torch.isinf(logits.grad).any(), "Gradients contain inf"
    
    def test_gradients_with_high_gamma(self):
        """Test gradient stability with higher gamma values."""
        loss_fn = FocalLoss(gamma=5.0)
        
        logits = torch.randn(32, 89, requires_grad=True)
        targets = torch.randint(0, 89, (32,))
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()
        
        # Gradients should have reasonable magnitude
        grad_norm = logits.grad.norm().item()
        assert grad_norm < 1000, f"Gradient norm too large: {grad_norm}"


class TestFocalLossEquivalence:
    """Test equivalence to CrossEntropyLoss when gamma=0."""
    
    def test_gamma_zero_equals_cross_entropy(self):
        """Test that FocalLoss(gamma=0) ≈ CrossEntropyLoss."""
        focal_loss = FocalLoss(gamma=0.0)
        ce_loss = nn.CrossEntropyLoss()
        
        # Test multiple random samples
        for _ in range(10):
            logits = torch.randn(32, 89)
            targets = torch.randint(0, 89, (32,))
            
            focal_output = focal_loss(logits, targets)
            ce_output = ce_loss(logits, targets)
            
            # Should be very close (within floating point error)
            diff = torch.abs(focal_output - ce_output).item()
            assert diff < 1e-5, f"Difference too large: {diff}"
    
    def test_gamma_zero_with_alpha(self):
        """Test that alpha weighting works with gamma=0."""
        alpha = 0.25
        focal_loss = FocalLoss(gamma=0.0, alpha=alpha)
        ce_loss = nn.CrossEntropyLoss()
        
        logits = torch.randn(32, 89)
        targets = torch.randint(0, 89, (32,))
        
        focal_output = focal_loss(logits, targets)
        ce_output = ce_loss(logits, targets)
        
        # Focal with alpha should be approximately alpha * CE
        expected = alpha * ce_output
        diff = torch.abs(focal_output - expected).item()
        assert diff < 1e-5, f"Difference too large: {diff}"


class TestFocalLossNumericalStability:
    """Test numerical stability with edge cases."""
    
    def test_extreme_logits(self):
        """Test with very large and very small logits."""
        loss_fn = FocalLoss(gamma=2.0)
        
        # Very confident predictions
        logits = torch.zeros(8, 10)
        logits[0, 0] = 100.0  # Extremely confident
        logits[1, 1] = -100.0  # Extremely wrong
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        
        loss = loss_fn(logits, targets)
        
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is inf"
        assert loss.item() > 0, "Loss should be positive"
    
    def test_all_correct_predictions(self):
        """Test when all predictions are correct and confident."""
        loss_fn = FocalLoss(gamma=2.0, reduction='none')
        
        # Perfect predictions
        logits = torch.zeros(8, 10)
        for i in range(8):
            logits[i, i] = 10.0  # Very confident correct predictions
        targets = torch.arange(8)
        
        losses = loss_fn(logits, targets)
        
        # All losses should be very small (nearly 0)
        assert (losses < 0.01).all(), "Losses should be near 0 for confident correct predictions"
    
    def test_all_wrong_predictions(self):
        """Test when all predictions are wrong."""
        loss_fn = FocalLoss(gamma=2.0, reduction='none')
        
        # Wrong predictions
        logits = torch.zeros(8, 10)
        for i in range(8):
            logits[i, (i + 1) % 10] = 10.0  # Confident but wrong
        targets = torch.arange(8)
        
        losses = loss_fn(logits, targets)
        
        # All losses should be high
        assert (losses > 1.0).all(), "Losses should be high for wrong predictions"


class TestFocalLossInputValidation:
    """Test input validation and error handling."""
    
    def test_invalid_gamma_negative(self):
        """Test that negative gamma raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be >= 0.0"):
            FocalLoss(gamma=-1.0)
    
    def test_invalid_gamma_too_large(self):
        """Test that very large gamma raises warning."""
        with pytest.raises(ValueError, match="gamma > 10.0"):
            FocalLoss(gamma=15.0)
    
    def test_invalid_alpha(self):
        """Test that alpha outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            FocalLoss(gamma=2.0, alpha=1.5)
        
        with pytest.raises(ValueError, match="alpha must be in"):
            FocalLoss(gamma=2.0, alpha=-0.1)
    
    def test_invalid_reduction(self):
        """Test that invalid reduction raises ValueError."""
        with pytest.raises(ValueError, match="reduction must be"):
            FocalLoss(gamma=2.0, reduction='invalid')
    
    def test_mismatched_batch_size(self):
        """Test that mismatched batch sizes raise ValueError."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(32, 89)
        targets = torch.randint(0, 89, (16,))  # Wrong batch size
        
        with pytest.raises(ValueError, match="batch size mismatch"):
            loss_fn(logits, targets)
    
    def test_invalid_target_range(self):
        """Test that targets outside valid range raise ValueError."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(32, 89)
        targets = torch.randint(0, 100, (32,))  # Targets >= num_classes
        
        with pytest.raises(ValueError, match="targets must be in"):
            loss_fn(logits, targets)
    
    def test_wrong_input_dimensions(self):
        """Test that wrong input dimensions raise ValueError."""
        loss_fn = FocalLoss(gamma=2.0)
        
        # 3D logits
        logits = torch.randn(32, 89, 10)
        targets = torch.randint(0, 89, (32,))
        with pytest.raises(ValueError, match="inputs must be 2D"):
            loss_fn(logits, targets)
        
        # 2D targets
        logits = torch.randn(32, 89)
        targets = torch.randint(0, 89, (32, 1))
        with pytest.raises(ValueError, match="targets must be 1D"):
            loss_fn(logits, targets)


class TestFocalLossRepr:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean')
        repr_str = repr(loss_fn)
        
        assert 'FocalLoss' in repr_str
        assert 'gamma=2.0' in repr_str
        assert 'alpha=0.25' in repr_str
        assert "reduction='mean'" in repr_str
