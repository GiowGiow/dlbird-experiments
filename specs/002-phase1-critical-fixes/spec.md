# Feature Specification: Phase 1 - Critical Fixes

**Spec ID**: 002-phase1-critical-fixes  
**Status**: 📋 Ready for Implementation  
**Created**: 2025-12-04  
**Last Updated**: 2025-12-04  
**Phase**: Phase 1 - Critical Fixes (Days 1-2)  
**Prerequisites**: 001-validation-phase (Complete)

---

## Executive Summary

Implement critical fixes to address the two primary root causes of poor audio model performance identified during validation: severe class imbalance (1216:1 ratio) and missing feature normalization.

**Problem**: Audio models achieve only 39.5% accuracy and 0.109 F1-macro due to:
1. SEVERE class imbalance causing model to ignore rare species (PRIMARY - 60% of problem)
2. Non-normalized MFCC features (mean=-8.80, std=62.53) causing slow convergence (SECONDARY - 20% of problem)

**Solution**: Implement class-weighted loss and feature normalization to address both issues simultaneously.

**Expected Impact**: 
- F1-macro: 0.109 → 0.25-0.35 (+130-220% improvement)
- Accuracy: 39.5% → 50-60% (+26-52% improvement)
- Training convergence: Faster and more stable

---

## User Stories

### US1: Class-Weighted Loss Implementation
**As a** ML researcher  
**I want** class-weighted loss function in the training loop  
**So that** the model learns to recognize rare species equally well

**Acceptance Criteria**:
- Class weights loaded from `artifacts/validation/recommended_class_weights.json`
- Weights applied in `nn.CrossEntropyLoss` during training
- Training script accepts `--use-class-weights` flag
- Weights correctly mapped to species order in dataset
- Training logs show weighted loss values

### US2: Feature Normalization
**As a** ML researcher  
**I want** MFCC features normalized to zero mean and unit variance  
**So that** training converges faster with stable gradients

**Acceptance Criteria**:
- Per-channel standardization: `(x - mean) / (std + epsilon)`
- Statistics: MFCC (mean=-8.80, std=62.53), Delta (mean=0.02, std=1.69)
- Normalization applied in dataset `__getitem__` method
- Dataset accepts `normalize=True` parameter
- Validation: batch statistics show mean≈0, std≈1

### US3: Baseline Model Retraining
**As a** ML researcher  
**I want** AudioCNN and AudioViT retrained with both fixes applied  
**So that** I can measure the combined impact of interventions

**Acceptance Criteria**:
- AudioCNN trained with class weights + normalization
- AudioViT trained with class weights + normalization
- Training runs for 50 epochs each
- Checkpoints saved as `baseline_v2_balanced_normalized`
- Training curves smooth with no instability

### US4: Phase 1 Results Validation
**As a** ML researcher  
**I want** comprehensive evaluation of Phase 1 models  
**So that** I can quantify improvements and decide on Phase 2 progression

**Acceptance Criteria**:
- F1-macro >0.25 (Go criteria for Phase 2)
- Accuracy >0.50
- Per-class F1 scores improved for rare species
- Confusion matrices show better balance
- Results documented in `artifacts/results/baseline_v2_comparison.json`

---

## Functional Requirements

### FR1: Trainer Class Modification
- **ID**: FR1.1  
  **Requirement**: Add `class_weights` parameter to `Trainer.__init__`  
  **Priority**: MUST HAVE  
  **File**: `src/training/trainer.py`

- **ID**: FR1.2  
  **Requirement**: Pass class_weights to `nn.CrossEntropyLoss(weight=class_weights)`  
  **Priority**: MUST HAVE  
  **File**: `src/training/trainer.py`

- **ID**: FR1.3  
  **Requirement**: Move class_weights tensor to correct device (CPU/CUDA)  
  **Priority**: MUST HAVE  
  **File**: `src/training/trainer.py`

### FR2: Training Script Updates
- **ID**: FR2.1  
  **Requirement**: Add `--use-class-weights` command-line argument  
  **Priority**: MUST HAVE  
  **File**: `scripts/03_train_audio.py`

- **ID**: FR2.2  
  **Requirement**: Load class weights from JSON file  
  **Priority**: MUST HAVE  
  **File**: `scripts/03_train_audio.py`

- **ID**: FR2.3  
  **Requirement**: Map weights to species order in dataset  
  **Priority**: MUST HAVE  
  **File**: `scripts/03_train_audio.py`

- **ID**: FR2.4  
  **Requirement**: Convert weights to PyTorch tensor and pass to Trainer  
  **Priority**: MUST HAVE  
  **File**: `scripts/03_train_audio.py`

### FR3: Dataset Normalization
- **ID**: FR3.1  
  **Requirement**: Add `normalize` parameter to `AudioMFCCDataset.__init__`  
  **Priority**: MUST HAVE  
  **File**: `src/datasets/audio.py`

- **ID**: FR3.2  
  **Requirement**: Store channel-wise statistics (mean, std) in dataset  
  **Priority**: MUST HAVE  
  **File**: `src/datasets/audio.py`

- **ID**: FR3.3  
  **Requirement**: Apply standardization in `__getitem__`: `(features - mean) / (std + 1e-8)`  
  **Priority**: MUST HAVE  
  **File**: `src/datasets/audio.py`

- **ID**: FR3.4  
  **Requirement**: Default `normalize=True` for training datasets  
  **Priority**: SHOULD HAVE  
  **File**: `src/datasets/audio.py`

### FR4: Validation & Testing
- **ID**: FR4.1  
  **Requirement**: Test class weight loading and tensor creation  
  **Priority**: MUST HAVE  
  **Test**: Manual test with print statements

- **ID**: FR4.2  
  **Requirement**: Validate normalization by checking batch statistics  
  **Priority**: MUST HAVE  
  **Test**: Load batch, compute mean/std, verify ≈0/1

- **ID**: FR4.3  
  **Requirement**: Run 1 epoch training test before full training  
  **Priority**: SHOULD HAVE  
  **Test**: Quick smoke test

- **ID**: FR4.4  
  **Requirement**: Generate Phase 1 comparison report  
  **Priority**: MUST HAVE  
  **Output**: `artifacts/results/baseline_v2_comparison.json`

---

## Technical Specifications

### Class Weight Calculation

**Method**: Balanced weighting (sklearn convention)
```
weight[c] = n_samples / (n_classes * n_samples_per_class[c])
```

**Source**: `artifacts/validation/recommended_class_weights.json` (use "balanced" method)

**Integration**:
```python
weights_dict = json.load(open('artifacts/validation/recommended_class_weights.json'))
class_weights = [weights_dict['balanced'][species] for species in species_list]
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

### Feature Normalization

**Statistics** (from validation):
- MFCC channel (0): mean = -8.80, std = 62.53
- Delta channel (1): mean = 0.02, std = 1.69  
- Delta² channel (2): mean ≈ 0.00, std ≈ 1.00 (already normalized, skip)

**Implementation**:
```python
if self.normalize:
    # MFCC channel
    features[0] = (features[0] - (-8.80)) / (62.53 + 1e-8)
    # Delta channel
    features[1] = (features[1] - 0.02) / (1.69 + 1e-8)
    # Delta² already normalized, no change needed
```

---

## Success Criteria & Go/No-Go Decision

### Phase 1 Checkpoint Success Criteria

**✅ GO to Phase 2** if:
- F1-macro > 0.25 (target: 0.25-0.35)
- Accuracy > 0.50 (target: 0.50-0.60)
- Training curves stable (no divergence)
- Rare species F1 > 0.10 (any improvement from ~0)

**⚠️ INVESTIGATE** if:
- F1-macro between 0.20-0.25
- Unexpected training instability
- No improvement for rare species

**❌ NO-GO** if:
- F1-macro < 0.20
- Training diverges or fails
- Performance worse than baseline v1

---

## Implementation Timeline

**Total Estimated Time**: 5-8 hours over 2 days

**Day 1** (3-5 hours):
- Fix 1: Class-weighted loss (2-3 hours)
  - Modify Trainer class
  - Update training script
  - Test with 1 epoch
- Fix 2: Feature normalization (1-2 hours)
  - Modify AudioMFCCDataset
  - Validate batch statistics

**Day 2** (2-3 hours):
- Retrain AudioCNN (1-2 hours GPU time)
- Retrain AudioViT (1-2 hours GPU time)
- Evaluate and document results (30 min)

**Can run models sequentially or in parallel if multiple GPUs available**

---

## Risks & Mitigation

### Risk 1: Class weights don't improve F1-macro
**Likelihood**: Low  
**Impact**: High  
**Mitigation**: Try alternative weighting schemes (effective samples, sqrt), or switch to oversampling

### Risk 2: Normalization causes training instability
**Likelihood**: Very Low  
**Impact**: Medium  
**Mitigation**: Use batch normalization in model architecture, reduce learning rate

### Risk 3: Combined fixes don't reach 70% target
**Likelihood**: Medium  
**Impact**: Medium  
**Mitigation**: This is expected for Phase 1 (target: 50-60%). Phase 2 focuses on reaching 70%+

### Risk 4: Cannot reach Phase 1 checkpoint (F1 <0.20)
**Likelihood**: Very Low  
**Impact**: High  
**Mitigation**: Re-assess approach, check for bugs in implementation, review validation analysis

---

## Out of Scope

The following are explicitly out of scope for Phase 1:
- Mel-spectrogram features (Phase 2)
- Data augmentation (Phase 2)
- Architecture changes (Phase 3)
- Pretrained models (Phase 3)
- Multi-modal fusion (Phase 4)
- Ensemble methods (Phase 4)

---

## Dependencies

**Code Dependencies**:
- `src/training/trainer.py` - Existing trainer class
- `src/datasets/audio.py` - AudioMFCCDataset class
- `scripts/03_train_audio.py` - Training script

**Data Dependencies**:
- `artifacts/validation/recommended_class_weights.json` - Pre-computed weights
- `artifacts/audio_mfcc_cache/` - Cached MFCC features
- `artifacts/splits/xeno_canto_audio_splits.json` - Train/val/test splits

**Environment**:
- PyTorch with CUDA support
- 16GB+ GPU RAM
- ~50GB disk space for checkpoints

---

## References

**Validation Documents**:
- VALIDATION_SUMMARY_EXECUTIVE.md
- VALIDATION_RESULTS.md

**Planning Documents**:
- IMPLEMENTATION_PLAN.md (Phase 1 section)
- QUICK_START_IMPLEMENTATION.md

**Artifacts**:
- artifacts/validation/recommended_class_weights.json
- artifacts/validation/feature_statistics.json
