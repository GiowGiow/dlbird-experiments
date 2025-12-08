# Feature Specification: Phase 2 - Focal Loss and Architecture Improvements

## Metadata

- **Feature ID**: 003-phase2-focal-loss-improvements
- **Feature Name**: Focal Loss and Architecture Improvements
- **Status**: Planning
- **Priority**: High
- **Created**: 2024-12-04
- **Last Updated**: 2024-12-04

## Problem Statement

Phase 1 experiments (class-weighted loss + normalization) **failed catastrophically**, degrading validation accuracy from 39.5% to 6.44-25.63%. The root causes identified:

1. **Class weights too aggressive**: Balanced method creates 45x weight ratios causing numerical instability
2. **Insufficient model capacity**: AudioCNN (343K params) lacks capacity for 89 classes with severe imbalance
3. **Normalization insufficient**: Feature scaling doesn't address fundamental class imbalance
4. **Severe imbalance**: 1216:1 ratio between most/least common species

**Current Performance**:
- AudioCNN: 39.5% accuracy, F1-macro 0.109
- AudioViT: 39.0% accuracy, F1-macro ~0.10
- **Target**: F1-macro > 0.25, accuracy > 50%

## Objectives

### Primary Goal
Improve F1-macro from 0.109 to >0.25 through:
1. Focal Loss (more sophisticated than class weights)
2. Increased model capacity
3. Targeted improvements without full architecture redesign

### Success Criteria
- ✅ F1-macro ≥ 0.25 (130% improvement)
- ✅ Validation accuracy ≥ 50% (26% improvement)
- ✅ No catastrophic degradation like Phase 1
- ✅ Rare species F1 improves (currently near 0)
- ⚠️ Training time ≤ 2x baseline

## Background

### Phase 1 Lessons Learned

**What Failed**:
- Class-weighted CrossEntropyLoss with balanced/sqrt methods
- Feature normalization alone
- Aggressive reweighting schemes (weights up to 45x)

**Why It Failed**:
- Numerical instability from extreme weight ratios
- Over-bias toward rare classes at expense of common classes
- Model capacity insufficient for weighted optimization landscape
- Normalization doesn't address class imbalance

**What Worked**:
- ✅ Flexible CLI infrastructure (`--model`, `--weight-method`, etc.)
- ✅ Normalization implementation (features correctly scaled)
- ✅ Training framework with AMP, early stopping, checkpointing

### Focal Loss Theory

Focal Loss (Lin et al., 2017) addresses class imbalance by:
1. **Down-weighting easy examples**: Reduces loss for well-classified samples
2. **Focusing on hard examples**: Maintains high loss for misclassified samples
3. **Self-adjusting**: Adapts based on prediction confidence, not class frequency

**Formula**: `FL(p_t) = -α_t(1-p_t)^γ * log(p_t)`

Where:
- `p_t`: Predicted probability for true class
- `α_t`: Class-specific weight (optional, typically 0.25-0.75)
- `γ`: Focusing parameter (typically 2.0)
- High `p_t` (confident correct) → low loss weight `(1-p_t)^γ → 0`
- Low `p_t` (incorrect) → high loss weight `(1-p_t)^γ → 1`

**Advantages over Class Weights**:
- Smooth, continuous down-weighting (no 45x spikes)
- Sample-specific adaptation (not just class-based)
- Better gradient flow in early training
- Proven effective for extreme imbalance (1000:1+)

## Proposed Solution

### Architecture: Three-Pronged Approach

#### 1. Focal Loss Implementation
- Replace CrossEntropyLoss with FocalLoss
- Hyperparameters: `α=0.25`, `γ=2.0` (standard values from paper)
- Support both focal-only and focal+reweighting modes
- Validate gradient stability

#### 2. Model Capacity Increase
**AudioCNN Enhancements**:
- Current: 343K params, 4 conv layers
- Proposed: ~1M params through:
  - Increase channel depth: [32, 64, 128, 256] → [64, 128, 256, 512]
  - Add one more conv block (5 total)
  - Increase FC layer: 256 → 512 hidden units
- Expected: Better feature discrimination for 89 classes

**AudioViT Modifications**:
- Current: 85M params (overkill), trains slowly
- Proposed: Use smaller ViT variant
  - google/vit-small-patch16-224 (22M params)
  - Or custom smaller ViT (embedding dim 384 → 256)

#### 3. Training Refinements
- Learning rate warmup (5 epochs) to stabilize focal loss
- Cosine annealing schedule for better convergence
- Gradient accumulation for effective batch size increase
- Monitor per-class metrics during training

### Implementation Plan

**Phase 2A: Focal Loss** (Priority 1)
1. Implement FocalLoss class in `src/training/losses.py`
2. Add `--loss-type` argument to training script
3. Train AudioCNN with focal loss (γ=2.0, α=0.25)
4. Compare with baseline: expect +5-10 pp accuracy, +0.10-0.15 F1-macro

**Phase 2B: Increased Capacity** (Priority 2)
5. Create AudioCNNv2 with 1M params
6. Train with focal loss
7. Compare with Phase 2A: expect +3-5 pp accuracy

**Phase 2C: ViT Optimization** (Priority 3)
8. Implement smaller ViT variant
9. Train with focal loss
10. Evaluate training efficiency

### Non-Goals (Future Work)
- ❌ Data augmentation (too complex for Phase 2)
- ❌ Transfer learning (requires external datasets)
- ❌ Ensemble methods (adds deployment complexity)
- ❌ Class-balanced sampling (Phase 1 showed weighting issues)

## Technical Specifications

### Focal Loss Implementation

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor in [0, 1] (0.25 standard)
            gamma: Focusing parameter >= 0 (2.0 standard)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # probability of true class
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha  # Can be per-class if needed
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```

### AudioCNNv2 Architecture

```python
class AudioCNNv2(nn.Module):
    """Increased capacity CNN for audio classification.
    
    Changes from AudioCNN:
    - Channels: [32,64,128,256] → [64,128,256,512]
    - Layers: 4 → 5 conv blocks
    - FC hidden: 256 → 512
    - Params: 343K → ~1M
    """
    def __init__(self, num_classes=89):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64 channels
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 512 channels
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 512 channels (deeper)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
```

### Training Configuration

```python
# Phase 2A: Focal Loss Baseline
focal_config = {
    'model': 'AudioCNN',
    'loss_type': 'focal',
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'epochs': 50,
    'batch_size': 64,
    'lr': 1e-3,
    'warmup_epochs': 5,
    'scheduler': 'cosine',
    'grad_accumulation': 1,
}

# Phase 2B: Increased Capacity
capacity_config = {
    'model': 'AudioCNNv2',
    'loss_type': 'focal',
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'epochs': 50,
    'batch_size': 32,  # Reduced for larger model
    'lr': 1e-3,
    'warmup_epochs': 5,
    'scheduler': 'cosine',
    'grad_accumulation': 2,  # Effective batch 64
}
```

## User Stories

### US1: Implement Focal Loss
**As a** ML engineer  
**I want** focal loss to replace class-weighted loss  
**So that** I can handle class imbalance without numerical instability

**Acceptance Criteria**:
- [ ] FocalLoss class implemented in `src/training/losses.py`
- [ ] Unit tests validate focal loss computation
- [ ] Training script accepts `--loss-type focal`
- [ ] Trainer uses FocalLoss when specified
- [ ] Gradient norms stable (no explosions)

### US2: Train AudioCNN with Focal Loss
**As a** researcher  
**I want** to train AudioCNN with focal loss  
**So that** I can compare against Phase 1 results

**Acceptance Criteria**:
- [ ] Training completes 50 epochs or early stops
- [ ] Validation accuracy > 40% (beats baseline)
- [ ] F1-macro > 0.15 (38% improvement over baseline)
- [ ] No catastrophic collapse like Phase 1
- [ ] Checkpoint and history saved

### US3: Implement AudioCNNv2
**As a** ML engineer  
**I want** larger CNN with ~1M params  
**So that** model has sufficient capacity for 89 classes

**Acceptance Criteria**:
- [ ] AudioCNNv2 class created
- [ ] Parameter count verified (~1M ±10%)
- [ ] Forward pass tested with sample input
- [ ] Model integrated into training script
- [ ] Architecture documented

### US4: Train AudioCNNv2 with Focal Loss
**As a** researcher  
**I want** to train larger model with focal loss  
**So that** I can measure capacity impact

**Acceptance Criteria**:
- [ ] Training completes successfully
- [ ] Validation accuracy > Phase 2A results
- [ ] F1-macro > 0.20 (target closer to 0.25)
- [ ] Training time ≤ 2x baseline
- [ ] Results documented in Phase 2 report

### US5: Hyperparameter Sensitivity Analysis
**As a** researcher  
**I want** to test focal loss γ values [1.0, 2.0, 3.0]  
**So that** I understand sensitivity to focusing parameter

**Acceptance Criteria**:
- [ ] Train AudioCNN with γ=1.0
- [ ] Train AudioCNN with γ=3.0
- [ ] Compare all three: γ∈{1.0, 2.0, 3.0}
- [ ] Document which γ performs best
- [ ] Update default γ if needed

## Dependencies

### Prerequisites (Existing)
- ✅ Training infrastructure (`src/training/trainer.py`)
- ✅ CLI argument parsing (`scripts/03_train_audio.py`)
- ✅ AudioCNN baseline implementation
- ✅ Dataset with 89 species, 11K recordings
- ✅ Train/val/test splits (70/15/15)
- ✅ MFCC feature cache

### New Dependencies
- PyTorch >= 1.10 (for stable focal loss implementation)
- No new external packages required

### Blocks
- None (self-contained improvements)

## Risks and Mitigations

### Risk 1: Focal Loss Still Fails
**Probability**: Medium  
**Impact**: High  
**Mitigation**: 
- Test on small subset first (10 epochs)
- Monitor gradient norms closely
- Have fallback: temperature scaling + standard CE
- Prepare Plan C: data augmentation

### Risk 2: Increased Capacity Overfits
**Probability**: Medium  
**Impact**: Medium  
**Mitigation**:
- Strong regularization (dropout 0.5, weight decay 1e-4)
- Early stopping (patience=7)
- Monitor train/val gap carefully
- Use gradient accumulation to maintain effective batch size

### Risk 3: Training Time Exceeds Budget
**Probability**: Low  
**Impact**: Medium  
**Mitigation**:
- Use AMP (already enabled)
- Profile and optimize bottlenecks
- Consider reducing val frequency if needed
- AudioCNNv2 still only 1M params (manageable on RTX 3060)

### Risk 4: Rare Species Still Ignored
**Probability**: Medium  
**Impact**: High  
**Mitigation**:
- Monitor per-class metrics during training
- Try focal loss with per-class α if needed
- Consider separate model for rare species if necessary
- Document findings for Phase 3 planning

## Success Metrics

### Primary Metrics
| Metric | Baseline | Phase 1 | Phase 2 Target | Measurement |
|--------|----------|---------|----------------|-------------|
| F1-Macro | 0.109 | 0.02-0.08 | **≥0.25** | Validation set |
| Accuracy | 39.5% | 6-26% | **≥50%** | Validation set |
| Rare Class F1 | ~0 | ~0 | **>0.10** | Classes with <50 samples |

### Secondary Metrics
- Training stability: No gradient explosions (norm < 10.0)
- Training time: ≤ 2x baseline (~2 hours for AudioCNN)
- Model size: AudioCNNv2 ≤ 1.2M params
- Convergence: Early stopping before epoch 40

### Experiment Matrix
| Experiment | Model | Loss | Expected Acc | Expected F1 |
|------------|-------|------|--------------|-------------|
| Baseline | AudioCNN | CE | 39.5% | 0.109 |
| Phase 2A | AudioCNN | Focal(γ=2) | 45-50% | 0.15-0.20 |
| Phase 2B | AudioCNNv2 | Focal(γ=2) | 50-55% | 0.20-0.25 |
| Phase 2C (opt) | AudioCNN | Focal(γ=1) | 43-48% | 0.14-0.19 |
| Phase 2D (opt) | AudioCNN | Focal(γ=3) | 46-51% | 0.16-0.21 |

## Timeline

### Phase 2A: Focal Loss (Week 1)
- Day 1: Implement FocalLoss class, unit tests (4 hours)
- Day 2: Integrate into Trainer and training script (4 hours)
- Day 3-4: Train AudioCNN with focal loss, monitor (2 hours setup + 12 hours GPU)
- Day 5: Analyze results, document findings (4 hours)

### Phase 2B: Increased Capacity (Week 2)
- Day 6: Implement AudioCNNv2, validate architecture (4 hours)
- Day 7-8: Train AudioCNNv2 with focal loss (2 hours setup + 16 hours GPU)
- Day 9: Compare Phase 2A vs 2B, document (4 hours)

### Phase 2C: Hyperparameter Sweep (Optional, Week 3)
- Day 10-12: Train with γ∈{1.0, 3.0} (24 hours GPU)
- Day 13: Comprehensive Phase 2 analysis and report (6 hours)

**Total Estimated**: 2-3 weeks wall time, ~30 hours human time, ~50 hours GPU time

## Future Enhancements (Phase 3)

If Phase 2 succeeds (F1 ≥ 0.20) but doesn't reach target (0.25):

1. **Data Augmentation**: Time stretching, pitch shifting, SpecAugment
2. **Transfer Learning**: Pre-train on BirdCLEF or full Xeno-Canto
3. **Ensemble Methods**: Combine multiple models
4. **Hierarchical Classification**: Genus → Species
5. **Semi-Supervised Learning**: Use unlabeled Xeno-Canto data

If Phase 2 fails (F1 < 0.15):
- Pivot to multimodal approach (combine audio + images)
- Consider this a fundamental dataset limitation
- Focus on improving data quality instead of model complexity

## References

1. Lin et al. (2017) "Focal Loss for Dense Object Detection" - Original focal loss paper
2. Phase 1 Results: `.specify/specs/002-phase1-critical-fixes/PHASE1_RESULTS.md`
3. Constitution: `.specify/memory/constitution.md`
4. Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples" - Alternative to focal loss
5. Buda et al. (2018) "A systematic study of the class imbalance problem" - Survey of imbalance techniques

## Appendix

### Phase 1 vs Phase 2 Comparison

| Aspect | Phase 1 (Failed) | Phase 2 (Proposed) |
|--------|------------------|-------------------|
| Loss Function | Weighted CE | Focal Loss |
| Weight Range | [0.037, 45.0] | Continuous [0, 1] |
| Adaptation | Class-based | Sample-based |
| Stability | Poor (collapsed) | Better (proven) |
| Model Capacity | 343K params | 343K → 1M params |
| Normalization | Yes (insufficient) | Keep (validated) |
| Results | 6-26% acc | Target 50%+ acc |

### Constitutional Alignment

- ✅ **Reproducibility**: All experiments fully reproducible with seeds
- ✅ **Clean Code**: Focal loss is well-documented, standard implementation
- ✅ **Simplicity**: Focal loss simpler than complex reweighting schemes
- ✅ **Baselines**: Building on validated Phase 1 infrastructure
- ✅ **Traceability**: Clear experiment tracking and artifact storage

---

**Approval Required**: Technical Lead, Research Lead  
**Next Step**: Create `plan.md` with detailed implementation plan
