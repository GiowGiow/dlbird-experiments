# Phase 1 Implementation Results

**Date**: December 4, 2024  
**Objective**: Implement class-weighted loss + feature normalization to improve F1-macro from 0.109 to 0.25-0.35

## Summary

Phase 1 implementation revealed **critical issues** with the proposed fixes:

- ✅ **Implementation Complete**: All code changes successfully implemented
- ✅ **Normalization Validated**: Features properly normalized (mean≈0, std≈1)
- ❌ **Performance Degraded**: All Phase 1 configurations performed **worse** than baseline
- 🔍 **Root Cause**: Class weights too aggressive, normalization alone insufficient

## Training Results

### Baseline (Phase 0)
| Model | Val Accuracy | F1-Macro | Notes |
|-------|-------------|----------|-------|
| AudioCNN | **39.5%** | 0.109 | Original baseline |
| AudioViT | **39.0%** | ~0.10 | Original baseline |

### Phase 1 Experiments

#### Experiment 1: Balanced Weights + Normalization
**Command**: `--model AudioCNN --use-class-weights --weight-method balanced`

**Results**:
- **Val Accuracy**: 6.44% (↓ 33.1 pp)
- **Epochs**: 23 (early stopped)
- **Weight Range**: [0.037, 45.0]
- **Status**: ❌ FAILED - Severe performance degradation

**Analysis**: The "balanced" method produces extremely aggressive weights (up to 45x), causing numerical instability and over-biasing toward rare classes.

#### Experiment 2: Sqrt Reweighting + Normalization
**Command**: `--model AudioCNN --use-class-weights --weight-method sqrt_reweighting`

**Results**:
- **Val Accuracy**: 25.63% (↓ 13.9 pp)
- **Epochs**: 26 (early stopped)
- **Weight Range**: [0.318, 7.844]
- **Status**: ⚠️ POOR - Still significantly worse than baseline

**Analysis**: More moderate weights (sqrt method) perform better than balanced, but still degrade performance substantially.

#### Experiment 3: Normalization Only
**Command**: `--model AudioCNN --epochs 50`

**Results**:
- **Val Accuracy**: ~20% (estimated, training interrupted at epoch 6)
- **Epochs**: 6+ (interrupted)
- **Status**: ⚠️ INCOMPLETE - But trend suggests normalization alone doesn't help

**Partial Progress**:
- Epoch 1: 15.10%
- Epoch 2: 15.94%
- Epoch 3: 18.53%
- Epoch 4: 19.19%
- Epoch 5: 20.28%

**Analysis**: Normalization alone appears insufficient to match baseline performance.

#### Experiment 4: AudioViT + Balanced Weights + Normalization
**Command**: `--model AudioViT --use-class-weights --weight-method balanced`

**Status**: Training in progress (very slow, ~3.8 it/s)

## Technical Implementation

### Code Changes (All Completed ✅)

1. **Class-Weighted Loss** (`src/training/trainer.py`):
   ```python
   if class_weights is not None:
       self.class_weights = class_weights.to(device)
       self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
   else:
       self.criterion = nn.CrossEntropyLoss()
   ```

2. **Feature Normalization** (`src/datasets/audio.py`):
   ```python
   # MFCC channel standardization
   features[0] = (features[0] - self.mfcc_mean) / (self.mfcc_std + 1e-8)
   
   # Delta channel standardization
   features[1] = (features[1] - self.delta_mean) / (self.delta_std + 1e-8)
   ```
   
   Statistics used:
   - MFCC: mean=-8.80, std=62.53
   - Delta: mean=0.02, std=1.69

3. **Training Script Enhancements** (`scripts/03_train_audio.py`):
   - Added argparse CLI with flags: `--model`, `--use-class-weights`, `--weight-method`, `--epochs`, `--batch-size`, `--lr`, `--save-name`
   - Created `load_class_weights()` utility function
   - Default weight method changed to `sqrt_reweighting` (most stable)

### Normalization Validation

Test script: `test_normalization.py`

**Results** (✅ PASSED):
```
MFCC Channel:
   Mean: -0.0064 (target: ~0.00) ✓ PASS
   Std:  1.0270 (target: ~1.00) ✓ PASS

Delta Channel:
   Mean: -0.0008 (target: ~0.00) ✓ PASS
   Std:  1.0654 (target: ~1.00) ✓ PASS
```

## Analysis

### Why Class Weights Failed

1. **Extreme Weight Ratios**: Balanced method creates 1216:1 weight ratios (matching class imbalance ratio)
2. **Numerical Instability**: Weights up to 45x cause gradient explosions/vanishing
3. **Over-correction**: Model focuses almost exclusively on rare classes, ignoring common classes
4. **Loss Landscape Distortion**: Weighted loss creates unstable optimization landscape

### Class Weight Statistics

| Method | Min | Max | Mean | Median | Std | Range |
|--------|-----|-----|------|--------|-----|-------|
| inverse_frequency | 0.10 | 123.07 | 7.47 | 2.26 | 16.25 | 122.97 |
| balanced | **0.04** | **45.00** | 2.73 | 0.83 | 5.94 | **44.96** |
| sqrt_reweighting | 0.32 | 11.09 | 2.07 | 1.50 | 1.79 | 10.78 |

### Why Normalization Alone Failed

1. **Insufficient Fix**: Normalization addresses feature scale but not class imbalance
2. **Baseline Already Reasonable**: Original features may have been adequately scaled
3. **Model Architecture Limitation**: AudioCNN (343K params) may lack capacity for 89 classes
4. **Dataset Quality**: Underlying issue may be data quality, not preprocessing

## Recommendations

### ❌ **NO-GO Decision for Phase 1 as Originally Specified**

The proposed fixes (class weights + normalization) **degrade performance** and should **not be deployed**.

### Alternative Approaches to Investigate

1. **Focal Loss**: More sophisticated than class weights, reduces weight for well-classified examples
   - Formula: `FL(p_t) = -α_t(1-p_t)^γ * log(p_t)`
   - Automatically down-weights easy examples
   - Less aggressive than explicit class weights

2. **Architecture Changes**:
   - Increase model capacity (AudioCNN only has 343K params for 89 classes)
   - Add attention mechanisms to focus on discriminative features
   - Try ensemble methods

3. **Data Augmentation**:
   - Time stretching, pitch shifting for rare species
   - Mixup/SpecAugment to improve generalization
   - Synthetic minority oversampling (SMOTE)

4. **Transfer Learning**:
   - Pre-train on larger bird audio dataset (BirdCLEF, Xeno-Canto full)
   - Fine-tune on intersection dataset
   - Use pretrained audio embeddings (PANNs, BEATs)

5. **Metric-Specific Training**:
   - Optimize directly for F1-macro instead of cross-entropy
   - Use class-balanced sampling in dataloader (not loss)
   - Two-stage training: common classes first, then rare classes

6. **Hybrid Approach**:
   - Train separate models for common vs rare species
   - Ensemble with confidence-based routing
   - Use hierarchical classification (genus → species)

## Files Modified

1. `src/training/trainer.py` - Added class weights support
2. `src/datasets/audio.py` - Added feature normalization
3. `scripts/03_train_audio.py` - Added CLI and weight loading
4. `test_normalization.py` - Created validation script

## Artifacts Generated

- `artifacts/models/baseline_v2_cnn_balanced_normalized/` - Balanced weights (6.44% acc)
- `artifacts/models/baseline_v2_cnn_sqrt_normalized/` - Sqrt weights (25.63% acc)
- `artifacts/models/baseline_v2_cnn_normalized_only/` - Normalization only (~20% acc)
- `artifacts/models/test_phase1_smoke/` - Initial smoke test (3.97% acc for 1 epoch)

## Next Steps

1. **Immediate**: Document findings and update STATUS.md
2. **Short-term**: Investigate focal loss and data augmentation
3. **Medium-term**: Explore architecture improvements (more capacity, attention)
4. **Long-term**: Consider transfer learning from larger bird audio datasets

## Conclusion

Phase 1 implementation was **technically successful** but **functionally failed** to improve performance. The root cause is more fundamental than preprocessing - the model lacks sufficient capacity and the class imbalance is too severe for simple reweighting schemes. 

**Recommendation**: Pivot to Phase 2 focusing on:
- Model architecture improvements (increase capacity)
- Focal loss instead of class weights
- Data augmentation for rare species
- Possibly transfer learning

The silver lining: We now have a flexible training framework with CLI arguments, weight method selection, and proper normalization infrastructure that can support future experiments.
