# Implementation Plan: Audio Model Improvements

**Project**: SpeckitDLBird - Bird Species Classification  
**Date**: December 4, 2025  
**Status**: 📋 Planning Phase  
**Based On**: Validation results from comprehensive QA analysis

---

## Executive Summary

### Current State
- ✅ Validation complete - root causes identified
- Audio models: 39.5% accuracy, F1-macro 0.109
- Image models: 92.6% accuracy, F1-macro 0.925
- **Gap**: 2.3x performance difference

### Root Causes Identified
1. 🔴 **SEVERE class imbalance** (1216:1 ratio) - PRIMARY (60% of problem)
2. 🟡 **Missing feature normalization** - SECONDARY (20% of problem)
3. 🟢 **Suboptimal feature representation** - TERTIARY (20% of problem)

### Success Criteria
- **Phase 1 Target**: 50-60% accuracy, F1-macro 0.25-0.35
- **Phase 2 Target**: 65-75% accuracy, F1-macro 0.50-0.60
- **Stretch Goal**: 75-85% accuracy, F1-macro 0.65-0.75

---

## Implementation Phases

### Phase 1: Critical Fixes (Priority: HIGH, Duration: 1-2 days)
**Goal**: Fix root causes identified in validation
**Expected Impact**: +20-30% improvement
**Must Complete Before**: Any new experiments

### Phase 2: Feature Engineering (Priority: HIGH, Duration: 3-5 days)
**Goal**: Improve audio feature representation
**Expected Impact**: +10-20% additional improvement

### Phase 3: Architecture Optimization (Priority: MEDIUM, Duration: 1-2 weeks)
**Goal**: Explore better model architectures
**Expected Impact**: +5-15% additional improvement

### Phase 4: Advanced Techniques (Priority: MEDIUM, Duration: 2-4 weeks)
**Goal**: Multi-modal fusion, ensemble methods
**Expected Impact**: +5-10% additional improvement

---

## Phase 1: Critical Fixes (Days 1-2)

### 1.1 Implement Class-Weighted Loss

**Priority**: 🔴 CRITICAL  
**Estimated Time**: 2-3 hours  
**Expected Impact**: F1-macro 0.109 → 0.25-0.35 (+130-220%)

#### Tasks

**Task 1.1.1**: Load and prepare class weights (30 min)
- [ ] Load `artifacts/validation/recommended_class_weights.json`
- [ ] Parse "balanced" method weights
- [ ] Create species-to-weight mapping
- [ ] Convert to PyTorch tensor

**Files to modify**:
- `src/training/trainer.py`

**Code location**:
```python
# In Trainer.__init__ or train method
def __init__(self, model, device, class_weights=None):
    self.class_weights = class_weights
    if class_weights is not None:
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        self.criterion = nn.CrossEntropyLoss()
```

**Task 1.1.2**: Update training scripts (1 hour)
- [ ] Add `--use-class-weights` argument to training scripts
- [ ] Load weights from JSON
- [ ] Map to correct species order
- [ ] Pass to Trainer

**Files to modify**:
- `scripts/03_train_audio.py`
- Any other training scripts

**Code to add**:
```python
import json
import torch

# Add argument parser
parser.add_argument('--use-class-weights', action='store_true',
                   help='Use class weights for imbalanced data')

# Load weights
if args.use_class_weights:
    with open('artifacts/validation/recommended_class_weights.json') as f:
        weights_data = json.load(f)
    
    # Use balanced method (recommended)
    class_weights_dict = weights_data['balanced']
    
    # Create tensor in correct species order
    weights_list = [class_weights_dict[species] for species in species_list]
    class_weights = torch.tensor(weights_list, dtype=torch.float).to(device)
    print(f"Using class weights - range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
else:
    class_weights = None
```

**Task 1.1.3**: Test with AudioCNN (30 min)
- [ ] Run training with class weights: `python scripts/03_train_audio.py --model AudioCNN --use-class-weights`
- [ ] Monitor training loss and validation metrics
- [ ] Verify weights are being applied

**Validation**:
```bash
# Check training output shows class weights loaded
# Training should converge faster
# Validation F1-macro should improve significantly
```

**Success Criteria**:
- ✅ Training runs without errors
- ✅ Loss function uses weighted criterion
- ✅ F1-macro improves from 0.109 to >0.20

---

### 1.2 Implement Feature Normalization

**Priority**: 🔴 CRITICAL  
**Estimated Time**: 1-2 hours  
**Expected Impact**: +5-10% accuracy, faster convergence

#### Tasks

**Task 1.2.1**: Add normalization to dataset (1 hour)
- [ ] Update `AudioMFCCDataset.__init__` to store normalization stats
- [ ] Update `__getitem__` to normalize features
- [ ] Add option to enable/disable normalization

**Files to modify**:
- `src/datasets/audio.py`

**Code changes**:
```python
class AudioMFCCDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Union[str, Path],
        indices: List[int],
        species_to_idx: Dict[str, int],
        transform=None,
        normalize=True,  # NEW parameter
    ):
        """
        Args:
            ...
            normalize: If True, apply per-channel standardization
        """
        self.df = df.iloc[indices].reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.species_to_idx = species_to_idx
        self.transform = transform
        self.normalize = normalize
        
        # Normalization statistics from validation
        if normalize:
            self.mfcc_mean = -8.80
            self.mfcc_std = 62.53
            self.delta_mean = 0.02
            self.delta_std = 1.69
            # Delta² already normalized (mean≈0, std≈1)
            print("Feature normalization ENABLED")
        else:
            print("Feature normalization DISABLED")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load cached features and return tensor and label."""
        row = self.df.iloc[idx]
        
        # Get species and record ID
        species = row.get("species_normalized", row.get("species", "unknown"))
        record_id = row.get("record_id", row.get("sample_id", idx))
        
        # Build cache path
        species_safe = str(species).replace("/", "_").replace(" ", "_")
        cache_file = self.cache_dir / species_safe / f"{record_id}.npy"
        
        # Load cached features
        if not cache_file.exists():
            raise FileNotFoundError(f"Cached features not found: {cache_file}")
        
        features = np.load(cache_file)  # (H, W, 3)
        
        # Transpose to (3, H, W) for PyTorch
        features = np.transpose(features, (2, 0, 1))
        
        # Normalize per channel if enabled
        if self.normalize:
            features[0] = (features[0] - self.mfcc_mean) / (self.mfcc_std + 1e-8)
            features[1] = (features[1] - self.delta_mean) / (self.delta_std + 1e-8)
            # features[2] (Delta²) already normalized
        
        # Convert to tensor
        features = torch.from_numpy(features).float()
        
        # Apply transform if provided
        if self.transform:
            features = self.transform(features)
        
        # Get label
        label = self.species_to_idx[species]
        
        return features, label
```

**Task 1.2.2**: Update training/evaluation scripts (30 min)
- [ ] Update all dataset instantiations to use `normalize=True`
- [ ] Add argument to control normalization
- [ ] Update notebooks

**Files to modify**:
- Training notebooks (if any)
- Evaluation scripts
- Any code creating `AudioMFCCDataset`

**Task 1.2.3**: Verify normalization (30 min)
- [ ] Load sample batch
- [ ] Check feature statistics (should be mean≈0, std≈1)
- [ ] Train short test run

**Validation**:
```python
# Check normalized features
dataloader = DataLoader(dataset, batch_size=32)
batch, labels = next(iter(dataloader))
print(f"MFCC channel - mean: {batch[:,0].mean():.4f}, std: {batch[:,0].std():.4f}")
print(f"Delta channel - mean: {batch[:,1].mean():.4f}, std: {batch[:,1].std():.4f}")
print(f"Delta² channel - mean: {batch[:,2].mean():.4f}, std: {batch[:,2].std():.4f}")
# Should all be close to 0 and 1
```

**Success Criteria**:
- ✅ Features normalized (mean≈0, std≈1)
- ✅ Training converges faster
- ✅ Loss curves smoother

---

### 1.3 Retrain Baseline Models

**Priority**: 🔴 CRITICAL  
**Estimated Time**: 2-4 hours (training time)  
**Expected Impact**: Establish new baseline for comparison

#### Tasks

**Task 1.3.1**: Train AudioCNN with fixes (1-2 hours)
- [ ] Run: `python scripts/03_train_audio.py --model AudioCNN --use-class-weights --epochs 50`
- [ ] Monitor training curves
- [ ] Save checkpoints
- [ ] Log results

**Expected results**:
- Accuracy: 0.395 → 0.50-0.60
- F1-macro: 0.109 → 0.25-0.35
- F1-weighted: 0.332 → 0.45-0.55

**Task 1.3.2**: Train AudioViT with fixes (1-2 hours)
- [ ] Run: `python scripts/03_train_audio.py --model AudioViT --use-class-weights --epochs 50`
- [ ] Compare with AudioCNN
- [ ] Evaluate AudioViT resizing issue impact

**Task 1.3.3**: Document baseline v2 (30 min)
- [ ] Save results as `baseline_v2_balanced_normalized.json`
- [ ] Create comparison table (v1 vs v2)
- [ ] Update `VALIDATION_RESULTS.md` with outcomes

**Files to create**:
- `artifacts/results/baseline_v2_comparison.json`
- Update `VALIDATION_RESULTS.md` with "Phase 1 Results" section

**Comparison template**:
```json
{
  "baseline_v1_unbalanced": {
    "audio_cnn": {"accuracy": 0.395, "f1_macro": 0.109, "f1_weighted": 0.332},
    "audio_vit": {"accuracy": 0.344, "f1_macro": 0.161, "f1_weighted": 0.317}
  },
  "baseline_v2_balanced_normalized": {
    "audio_cnn": {"accuracy": 0.XXX, "f1_macro": 0.XXX, "f1_weighted": 0.XXX},
    "audio_vit": {"accuracy": 0.XXX, "f1_macro": 0.XXX, "f1_weighted": 0.XXX}
  },
  "improvement": {
    "audio_cnn": {"accuracy": "+XX%", "f1_macro": "+XX%", "f1_weighted": "+XX%"},
    "audio_vit": {"accuracy": "+XX%", "f1_macro": "+XX%", "f1_weighted": "+XX%"}
  }
}
```

**Success Criteria**:
- ✅ Both models trained successfully
- ✅ F1-macro improves significantly (>+100%)
- ✅ Training curves show stable convergence
- ✅ Results documented

---

### Phase 1 Summary

**Deliverables**:
1. ✅ Class-weighted loss implemented
2. ✅ Feature normalization added
3. ✅ Baseline v2 trained and evaluated
4. ✅ Results documented

**Expected Outcomes**:
- AudioCNN: 50-60% accuracy, F1-macro 0.25-0.35
- AudioViT: Similar or better performance
- Stable training with faster convergence
- Clear baseline for Phase 2 experiments

**Risk Mitigation**:
- If F1-macro doesn't improve >0.20, check:
  - Class weights loaded correctly
  - Species mapping is correct
  - No bugs in loss function
- If training unstable:
  - Reduce learning rate
  - Check normalization stats
  - Try gradient clipping

**Phase 1 Checkpoint**: 
- [ ] All tasks complete
- [ ] Results meet success criteria
- [ ] Ready to proceed to Phase 2

---

## Phase 2: Feature Engineering (Days 3-7)

### 2.1 Mel-Spectrogram Representation

**Priority**: 🔴 HIGH  
**Estimated Time**: 2-3 days  
**Expected Impact**: +10-20% improvement

#### Rationale
- MFCCs compress audio to 40 coefficients (information loss)
- Mel-spectrograms preserve more frequency detail
- Can treat audio as 224×224 "image" for CNN/ViT
- Eliminates AudioViT resizing problem

#### Tasks

**Task 2.1.1**: Implement mel-spectrogram extraction (4 hours)
- [ ] Create `src/features/melspec.py`
- [ ] Extract mel-spectrograms (224×224 resolution)
- [ ] Save as images or numpy arrays
- [ ] Cache similar to MFCC pipeline

**New file**: `src/features/melspec.py`
```python
"""Mel-spectrogram feature extraction and caching."""

import librosa
import numpy as np
from pathlib import Path

def extract_melspec_features(
    audio_path,
    n_mels=224,
    hop_length=512,
    n_fft=2048,
    target_sr=22050,
    duration=3.0,
):
    """Extract mel-spectrogram as 224×224 image."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=target_sr, duration=duration)
    
    # Pad/crop to fixed length
    target_length = int(target_sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        start = (len(y) - target_length) // 2
        y = y[start:start + target_length]
    
    # Extract mel-spectrogram
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
    )
    
    # Convert to dB scale
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    # Resize to 224×224 if needed
    from scipy.ndimage import zoom
    if melspec_db.shape != (224, 224):
        zoom_factors = (224 / melspec_db.shape[0], 224 / melspec_db.shape[1])
        melspec_db = zoom(melspec_db, zoom_factors, order=1)
    
    # Normalize to [0, 1]
    melspec_db = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-8)
    
    # Stack 3 channels (grayscale to RGB-like)
    melspec_3ch = np.stack([melspec_db] * 3, axis=-1)  # (224, 224, 3)
    
    return melspec_3ch.astype(np.float32)
```

**Task 2.1.2**: Create mel-spectrogram dataset (2 hours)
- [ ] Create `src/datasets/melspec.py`
- [ ] Similar to `AudioMFCCDataset` but for mel-specs
- [ ] Compatible with existing training pipeline

**Task 2.1.3**: Cache mel-spectrograms (2 hours)
- [ ] Run caching script for all audio files
- [ ] Save to `artifacts/melspec_cache/`
- [ ] Verify cache completeness

**Task 2.1.4**: Train models on mel-specs (4 hours)
- [ ] Train AudioCNN on mel-specs
- [ ] Train AudioViT on mel-specs (now proper 224×224 input!)
- [ ] Compare with MFCC baseline

**Expected results**:
- AudioCNN: 60-70% accuracy, F1-macro 0.40-0.50
- AudioViT: 65-75% accuracy (should improve significantly!)

**Success Criteria**:
- ✅ Mel-specs extracted and cached
- ✅ Models train successfully
- ✅ Performance improves over MFCC baseline
- ✅ AudioViT benefits from proper input format

---

### 2.2 Data Augmentation (SpecAugment)

**Priority**: 🟡 MEDIUM  
**Estimated Time**: 1-2 days  
**Expected Impact**: +5-10% improvement

#### Tasks

**Task 2.2.1**: Implement SpecAugment (3 hours)
- [ ] Create `src/datasets/augmentations.py`
- [ ] Implement time masking
- [ ] Implement frequency masking
- [ ] Add to dataset transforms

**Code to add**:
```python
"""Audio data augmentation techniques."""

import torch
import numpy as np

class SpecAugment:
    """SpecAugment for mel-spectrograms."""
    
    def __init__(self, freq_mask=15, time_mask=35, n_freq_masks=1, n_time_masks=1):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec):
        """Apply SpecAugment to spectrogram tensor (C, H, W)."""
        spec = spec.clone()
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask)
            f0 = np.random.randint(0, spec.shape[1] - f)
            spec[:, f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask)
            t0 = np.random.randint(0, spec.shape[2] - t)
            spec[:, :, t0:t0+t] = 0
        
        return spec
```

**Task 2.2.2**: Add augmentation to training (1 hour)
- [ ] Update dataset to use augmentation (train only)
- [ ] Add `--augment` flag to training scripts
- [ ] Test with/without augmentation

**Task 2.2.3**: Evaluate impact (2 hours)
- [ ] Train with augmentation
- [ ] Compare with non-augmented baseline
- [ ] Check if overfitting reduced

**Success Criteria**:
- ✅ Augmentation improves generalization
- ✅ Test performance improves
- ✅ Training/validation gap narrows

---

### 2.3 Longer Audio Duration

**Priority**: 🟡 MEDIUM  
**Estimated Time**: 1 day  
**Expected Impact**: +5-15% improvement

#### Tasks

**Task 2.3.1**: Analyze optimal duration (1 hour)
- [ ] Check audio duration distribution (from validation)
- [ ] Determine optimal duration (5-7 seconds)
- [ ] Estimate computational impact

**Task 2.3.2**: Re-extract features with longer duration (2 hours)
- [ ] Update feature extraction to 5 or 7 seconds
- [ ] Re-cache features
- [ ] Update dataset

**Task 2.3.3**: Train and evaluate (3 hours)
- [ ] Train models with longer duration
- [ ] Compare with 3-second baseline
- [ ] Analyze per-species improvements

**Success Criteria**:
- ✅ Longer duration captures more vocalizations
- ✅ Performance improves, especially for complex calls
- ✅ Computational cost acceptable

---

### Phase 2 Summary

**Deliverables**:
1. ✅ Mel-spectrogram pipeline implemented
2. ✅ SpecAugment added
3. ✅ Optimal duration determined
4. ✅ All experiments documented

**Expected Outcomes**:
- AudioCNN: 60-70% accuracy, F1-macro 0.40-0.50
- AudioViT: 65-75% accuracy, F1-macro 0.50-0.60
- Stable training with better generalization

---

## Phase 3: Architecture Optimization (Weeks 2-3)

### 3.1 Audio-Pretrained Models

**Priority**: 🟡 MEDIUM-HIGH  
**Estimated Time**: 3-5 days  
**Expected Impact**: +15-25% improvement

#### Options to Explore

**Option 3.1.1**: PANNs (Pretrained Audio Neural Networks)
- Pretrained on AudioSet (2 million clips)
- CNN-based architectures
- Available in `panns` library

**Option 3.1.2**: AudioMAE (Audio Masked Autoencoder)
- Self-supervised pretraining
- Transformer-based
- State-of-the-art for audio classification

**Option 3.1.3**: Wav2Vec 2.0
- Originally for speech
- Can adapt for general audio
- Strong feature extraction

#### Tasks

**Task 3.1.1**: Evaluate pretrained models (1 day)
- [ ] Load PANNs pretrained model
- [ ] Test on bird audio samples
- [ ] Compare feature quality with MFCC/melspec

**Task 3.1.2**: Fine-tune on bird species (2-3 days)
- [ ] Replace classification head
- [ ] Freeze early layers, fine-tune later layers
- [ ] Train with class-weighted loss

**Task 3.1.3**: Compare architectures (1 day)
- [ ] PANNs vs AudioMAE vs custom CNN
- [ ] Performance vs computational cost
- [ ] Select best model

**Success Criteria**:
- ✅ Pretrained models improve over from-scratch training
- ✅ Transfer learning effective for bird audio
- ✅ Best model identified

---

### 3.2 Architecture Search

**Priority**: 🟢 MEDIUM  
**Estimated Time**: 2-3 days  
**Expected Impact**: +5-10% improvement

#### Tasks

**Task 3.2.1**: Try different CNN architectures (1 day)
- [ ] EfficientNet (parameter-efficient)
- [ ] ResNet variants (ResNet-34, ResNet-50)
- [ ] DenseNet (dense connections)

**Task 3.2.2**: Optimize AudioCNN (1 day)
- [ ] Experiment with depth (more/fewer layers)
- [ ] Try different filter sizes
- [ ] Adjust dropout rates

**Task 3.2.3**: Hyperparameter tuning (1 day)
- [ ] Learning rate search
- [ ] Batch size optimization
- [ ] Optimizer comparison (Adam vs AdamW vs SGD)

**Success Criteria**:
- ✅ Optimal architecture identified
- ✅ Hyperparameters tuned
- ✅ Performance improved

---

### Phase 3 Summary

**Deliverables**:
1. ✅ Audio-pretrained models evaluated
2. ✅ Architecture search complete
3. ✅ Best model configuration identified

**Expected Outcomes**:
- Best audio model: 70-80% accuracy, F1-macro 0.60-0.70
- Clear understanding of architecture impact

---

## Phase 4: Advanced Techniques (Weeks 4-5)

### 4.1 Multi-Modal Fusion

**Priority**: 🟢 MEDIUM  
**Estimated Time**: 3-5 days  
**Expected Impact**: +5-10% improvement

#### Approaches

**Approach 4.1.1**: Late Fusion
- Combine predictions from audio and image models
- Weighted average or voting
- Simple, effective

**Approach 4.1.2**: Early Fusion
- Concatenate audio and image features
- Joint classification head
- More complex but potentially better

#### Tasks

**Task 4.1.1**: Late fusion (2 days)
- [ ] Load audio and image model predictions
- [ ] Experiment with weighting schemes
- [ ] Optimize weights on validation set

**Task 4.1.2**: Early fusion (3 days)
- [ ] Extract features from both modalities
- [ ] Concatenate feature vectors
- [ ] Train joint classifier

**Success Criteria**:
- ✅ Fusion improves over single modalities
- ✅ Complementary information exploited
- ✅ Best fusion strategy identified

---

### 4.2 Ensemble Methods

**Priority**: 🟢 LOW-MEDIUM  
**Estimated Time**: 2-3 days  
**Expected Impact**: +3-5% improvement

#### Tasks

**Task 4.2.1**: Model ensemble (1 day)
- [ ] Combine multiple audio models (CNN, ViT, PANNs)
- [ ] Average predictions
- [ ] Test different ensemble strategies

**Task 4.2.2**: Feature ensemble (1 day)
- [ ] Combine MFCC and mel-spec models
- [ ] Leverage different representations

**Task 4.2.3**: Optimize ensemble (1 day)
- [ ] Find optimal model weights
- [ ] Balance performance and computational cost

**Success Criteria**:
- ✅ Ensemble outperforms single models
- ✅ Marginal improvements achieved

---

### Phase 4 Summary

**Deliverables**:
1. ✅ Multi-modal fusion implemented
2. ✅ Ensemble methods explored
3. ✅ Final best model determined

**Expected Outcomes**:
- Final system: 75-85% accuracy, F1-macro 0.65-0.75
- Comprehensive comparison of all approaches

---

## Experiment Tracking

### Tracking System

Use structured experiment tracking for all experiments:

**Template**:
```yaml
experiment:
  id: audio_exp_XXX
  name: Descriptive name
  date: YYYY-MM-DD
  phase: 1/2/3/4
  
baseline:
  model: Reference model
  accuracy: 0.XXX
  f1_macro: 0.XXX
  f1_weighted: 0.XXX
  
changes:
  - Change 1
  - Change 2
  - Change 3
  
hyperparameters:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
  optimizer: Adam
  ...
  
results:
  accuracy: 0.XXX
  f1_macro: 0.XXX
  f1_weighted: 0.XXX
  training_time: XX min
  best_epoch: XX
  
analysis:
  improvement: +XX%
  observations:
    - Key observation 1
    - Key observation 2
  next_steps:
    - What to try next
```

### Tracking Location
- Save to: `artifacts/experiments/`
- Naming: `exp_XXX_description.yaml`
- Keep experiment log: `artifacts/experiments/experiment_log.md`

---

## Risk Management

### Potential Risks

**Risk 1**: Class-weighted loss doesn't improve F1-macro significantly
- **Mitigation**: Try oversampling minority classes (WeightedRandomSampler)
- **Fallback**: Use focal loss for hard examples

**Risk 2**: Feature normalization causes training instability
- **Mitigation**: Verify statistics, try different normalization methods
- **Fallback**: Use batch normalization in model instead

**Risk 3**: Mel-spectrograms don't improve over MFCC
- **Mitigation**: Tune mel-spectrogram parameters (n_mels, duration)
- **Fallback**: Try CQT (Constant-Q Transform) or raw waveforms

**Risk 4**: AudioViT still underperforms on audio
- **Mitigation**: Use audio-specific transformers (AudioMAE)
- **Fallback**: Focus on CNN architectures

**Risk 5**: Cannot reach 70%+ accuracy
- **Mitigation**: Collect more data for rare species
- **Fallback**: Accept data limitations, focus on common species

---

## Success Metrics

### Phase 1 Success
- [ ] F1-macro: >0.25 (target: 0.25-0.35)
- [ ] Accuracy: >0.50 (target: 0.50-0.60)
- [ ] Training stability: Smooth loss curves
- [ ] Time: <2 days to complete

### Phase 2 Success
- [ ] F1-macro: >0.40 (target: 0.40-0.50)
- [ ] Accuracy: >0.65 (target: 0.65-0.75)
- [ ] Mel-specs outperform MFCC
- [ ] Time: <1 week to complete

### Phase 3 Success
- [ ] F1-macro: >0.60 (target: 0.60-0.70)
- [ ] Accuracy: >0.70 (target: 0.70-0.80)
- [ ] Pretrained models show clear benefit
- [ ] Time: <2 weeks to complete

### Phase 4 Success (Stretch)
- [ ] F1-macro: >0.65 (target: 0.65-0.75)
- [ ] Accuracy: >0.75 (target: 0.75-0.85)
- [ ] Multi-modal fusion adds value
- [ ] Time: <1 month total project time

---

## Timeline

### Week 1
- **Days 1-2**: Phase 1 (Critical Fixes)
  - Implement class-weighted loss
  - Add feature normalization
  - Retrain baseline
- **Days 3-5**: Phase 2 Start (Mel-Spectrograms)
  - Extract mel-specs
  - Train models
  - Initial evaluation

### Week 2
- **Days 6-7**: Phase 2 Continue (Augmentation)
  - Implement SpecAugment
  - Train with augmentation
- **Days 8-10**: Phase 2 Complete (Duration)
  - Test longer audio
  - Finalize Phase 2

### Week 3
- **Days 11-15**: Phase 3 (Architecture)
  - Audio-pretrained models
  - Architecture search
  - Hyperparameter tuning

### Week 4-5
- **Days 16-25**: Phase 4 (Advanced)
  - Multi-modal fusion
  - Ensemble methods
  - Final optimization

### Week 5
- **Days 26-30**: Finalization
  - Documentation
  - Final evaluation
  - Paper/report writing

---

## Resource Requirements

### Computational
- **GPU**: Required (CUDA-capable)
- **RAM**: 16GB+ recommended
- **Storage**: ~50GB for caches and checkpoints
- **Training time**: ~100 GPU hours total

### Human
- **Phase 1**: 8-16 hours (2 days)
- **Phase 2**: 20-30 hours (1 week)
- **Phase 3**: 30-40 hours (2 weeks)
- **Phase 4**: 20-30 hours (2 weeks)
- **Total**: ~80-120 hours (1 month)

### Tools/Libraries
- PyTorch, torchvision, torchaudio
- librosa (audio processing)
- scikit-learn (metrics)
- transformers (Hugging Face)
- PANNs (if using pretrained audio models)
- wandb or mlflow (experiment tracking)

---

## Documentation Requirements

### Per Experiment
- [ ] Experiment YAML file
- [ ] Training logs
- [ ] Confusion matrices
- [ ] Sample predictions (especially errors)

### Per Phase
- [ ] Phase summary document
- [ ] Performance comparison tables
- [ ] Key insights and learnings

### Final
- [ ] Complete project report
- [ ] Best model card (architecture, hyperparameters, performance)
- [ ] Deployment guide (if applicable)
- [ ] Future work recommendations

---

## Validation and Testing

### After Each Phase
1. Run comprehensive evaluation on test set
2. Generate confusion matrices
3. Analyze per-class performance
4. Check for overfitting (train vs val/test gap)
5. Document results

### Code Quality
- Add docstrings to new functions
- Write unit tests for critical functions
- Keep code style consistent (PEP 8)
- Review and refactor after each phase

### Reproducibility
- Fix random seeds
- Document environment (requirements.txt or uv.lock)
- Save model checkpoints
- Version control all code changes

---

## Milestones and Checkpoints

### Milestone 1: Phase 1 Complete (Day 2)
- [ ] Class weights implemented and tested
- [ ] Feature normalization working
- [ ] Baseline v2 established
- [ ] F1-macro >0.25

**Go/No-Go Decision**: If F1-macro <0.20, investigate before Phase 2

### Milestone 2: Phase 2 Complete (Day 10)
- [ ] Mel-spectrograms implemented
- [ ] Augmentation working
- [ ] Duration optimized
- [ ] Accuracy >0.65

**Go/No-Go Decision**: If accuracy <0.60, re-evaluate approach

### Milestone 3: Phase 3 Complete (Day 20)
- [ ] Pretrained models evaluated
- [ ] Architecture optimized
- [ ] Accuracy >0.70

**Go/No-Go Decision**: If accuracy <0.70, skip Phase 4, focus on refinement

### Milestone 4: Phase 4 Complete (Day 30)
- [ ] Multi-modal fusion explored
- [ ] Best final model determined
- [ ] Documentation complete

---

## Next Steps (Immediate)

### Today (Next 4 hours)
1. [ ] Review and approve this implementation plan
2. [ ] Set up experiment tracking system
3. [ ] Create experiment tracking template
4. [ ] Prepare development environment

### Tomorrow (Day 1)
1. [ ] **START PHASE 1**
2. [ ] Implement class-weighted loss (Task 1.1)
3. [ ] Implement feature normalization (Task 1.2)
4. [ ] Start baseline retraining (Task 1.3)

### Day 2
1. [ ] Complete baseline retraining
2. [ ] Evaluate Phase 1 results
3. [ ] Document outcomes
4. [ ] **Phase 1 checkpoint**: Go/No-Go decision

---

## Conclusion

This implementation plan provides a structured, phased approach to improving audio model performance from 39.5% to 75-85% accuracy over approximately 1 month.

**Key Principles**:
- ✅ Fix root causes first (Phase 1)
- ✅ Iterate on features before architecture (Phase 2)
- ✅ Leverage transfer learning (Phase 3)
- ✅ Combine best techniques (Phase 4)
- ✅ Track everything systematically
- ✅ Validate at each checkpoint

**Success Factors**:
- Clear validation has identified root causes
- Realistic expectations based on data limitations
- Incremental approach with checkpoints
- Risk mitigation strategies in place
- Comprehensive tracking and documentation

**Expected Outcome**: 
A robust audio classification system achieving 75-85% accuracy with proper handling of class imbalance, optimized features, and state-of-the-art architectures.

---

**Plan Status**: 📋 Ready for Execution  
**Start Date**: December 5, 2025  
**Expected Completion**: January 5, 2026  
**Next Action**: Begin Phase 1, Task 1.1 (Class-Weighted Loss)

---

**Document Version**: 1.0  
**Last Updated**: December 4, 2025  
**Approved By**: [Pending Review]
