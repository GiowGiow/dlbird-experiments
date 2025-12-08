# Validation Checklist: Pre-Expansion Quality Assurance

**Project**: SpeckitDLBird - Bird Species Classification  
**Date**: December 4, 2025  
**Purpose**: Validate current implementation completeness, clarity, and consistency before audio model improvements

---

## Executive Summary

**Current Status**:
- ✅ Image Models: **Strong Performance** (ResNet-18: 84.8%, ViT: 92.6% accuracy)
- ⚠️ Audio Models: **Poor Performance** (CNN: 39.5%, ViT: 34.4% accuracy)
- 🎯 **Goal**: Validate existing implementation before audio experimentation

**Key Findings from Results**:
- Image modality significantly outperforms audio (2-3x better accuracy)
- ViT architecture excels for images, underperforms for audio
- Audio models show low F1-macro scores (0.11-0.16) indicating class imbalance issues
- Large gap between accuracy and F1-macro suggests poor rare-species performance

---

## 1. Requirements Completeness Checklist

### 1.1 Data Pipeline Requirements

#### ✅ Dataset Indexing
- [ ] **Xeno-Canto Audio Indexing**
  - [ ] All audio files successfully indexed?
  - [ ] Metadata parsed correctly (species, quality, location)?
  - [ ] File paths validated and accessible?
  - [ ] Expected columns present: `record_id`, `species`, `file_path`, `species_normalized`
  - [ ] **Verification Command**: 
    ```python
    df = pd.read_parquet("artifacts/xeno_canto_filtered.parquet")
    print(f"Records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Species count: {df['species_normalized'].nunique()}")
    ```

- [ ] **CUB-200 Image Indexing**
  - [ ] All images successfully indexed?
  - [ ] Image paths validated?
  - [ ] Species labels correctly mapped?
  - [ ] Expected columns present: `sample_id`, `species_normalized`, `file_path`
  - [ ] **Verification Command**:
    ```python
    df = pd.read_parquet("artifacts/cub_filtered.parquet")
    print(f"Images: {len(df)}")
    print(f"Species: {df['species_normalized'].nunique()}")
    print(f"Path validity: {df['file_path'].apply(Path).apply(lambda p: p.exists()).all()}")
    ```

#### ⚠️ Species Intersection & Filtering
- [ ] **Species Normalization**
  - [ ] Consistent normalization applied across datasets?
  - [ ] Authorship information removed?
  - [ ] Punctuation standardized?
  - [ ] Case normalized (lowercase)?
  - [ ] **Test**: Check that "American Crow" and "american crow" map to same normalized form

- [ ] **Intersection Computation**
  - [ ] Species intersection correctly computed?
  - [ ] Both datasets filtered to common species?
  - [ ] Minimum sample threshold applied (≥2 samples per species)?
  - [ ] **Critical**: Verify intersection saved at `artifacts/intersection_metadata.json`
  - [ ] **Verification**:
    ```python
    with open("artifacts/intersection_metadata.json") as f:
        meta = json.load(f)
    print(f"Common species: {meta['num_common_species']}")
    print(f"Audio samples: {meta['xeno_canto_samples']}")
    print(f"Image samples: {meta['cub_samples']}")
    ```

#### ⚠️ Data Splits
- [ ] **Split Generation**
  - [ ] Stratified splits created (train/val/test)?
  - [ ] Class distribution preserved across splits?
  - [ ] No data leakage (same individual in multiple splits)?
  - [ ] Split ratios correct (e.g., 70/15/15 or 80/10/10)?
  - [ ] Random seed fixed for reproducibility (SEED=42)?
  - [ ] **Verification**:
    ```python
    with open("artifacts/splits/xeno_canto_audio_splits.json") as f:
        splits = json.load(f)
    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    # Check no overlap
    assert len(set(splits['train']) & set(splits['test'])) == 0
    ```

### 1.2 Feature Extraction Requirements

#### ⚠️ Audio MFCC Features (CRITICAL for Audio Performance)
- [ ] **Feature Extraction Parameters**
  - [ ] n_mfcc = 40 (documented and consistent)?
  - [ ] hop_length = 512 (reasonable for 22050 Hz)?
  - [ ] n_fft = 2048 (covers sufficient frequency range)?
  - [ ] target_sr = 22050 Hz (standard audio SR)?
  - [ ] duration = 3.0 seconds (sufficient for bird calls)?
  - [ ] **Issue**: 3 seconds may be too short for some vocalizations

- [ ] **Feature Shape Validation**
  - [ ] Output shape is (H=40, W=~130, C=3)?
  - [ ] W (time frames) calculation: `W ≈ (3.0 * 22050) / 512 ≈ 129`
  - [ ] C=3 represents [MFCC, Delta, Delta²]?
  - [ ] **Verification**:
    ```python
    sample = np.load("artifacts/audio_mfcc_cache/xeno_canto/american_crow/<record_id>.npy")
    print(f"Feature shape: {sample.shape}")
    assert sample.shape[0] == 40, "n_mfcc mismatch"
    assert sample.shape[2] == 3, "Channel count mismatch"
    ```

- [ ] **Audio Preprocessing Quality**
  - [ ] Audio loading successful for all files?
  - [ ] Padding applied for short audio (<3s)?
  - [ ] Center cropping applied for long audio (>3s)?
  - [ ] Zero-padding vs. repeat-padding choice justified?
  - [ ] **Red Flag**: Random cropping may lose important call segments
  - [ ] **Test**: Listen to padded/cropped samples to verify quality

- [ ] **Feature Normalization**
  - [ ] Are MFCC features normalized (per-channel or global)?
  - [ ] **Missing**: No mention of feature scaling/normalization in code
  - [ ] **Critical**: MFCCs have different scales than deltas/delta-deltas
  - [ ] **Action Required**: Check if normalization needed
  - [ ] **Test**:
    ```python
    sample = np.load("<path>.npy")
    print(f"MFCC range: [{sample[:,:,0].min():.2f}, {sample[:,:,0].max():.2f}]")
    print(f"Delta range: [{sample[:,:,1].min():.2f}, {sample[:,:,1].max():.2f}]")
    print(f"Delta² range: [{sample[:,:,2].min():.2f}, {sample[:,:,2].max():.2f}]")
    ```

- [ ] **Caching System**
  - [ ] All features successfully cached?
  - [ ] Cache directory structure correct (species/record_id.npy)?
  - [ ] No missing cache files during training?
  - [ ] Cache validation before training starts?
  - [ ] **Verification**: Count cache files vs. expected count

#### ✅ Image Features (Working Well)
- [ ] **Image Preprocessing**
  - [ ] Resize to 224×224 (standard for pretrained models)?
  - [ ] Normalization using ImageNet stats?
  - [ ] Data augmentation applied (train only)?
  - [ ] Test transforms different from train (no augmentation)?

### 1.3 Model Architecture Requirements

#### ⚠️ Audio Models (Performance Issues)
- [ ] **AudioCNN Architecture**
  - [ ] Input shape correct: (batch, 3, 40, ~130)?
  - [ ] Convolutional blocks appropriate for MFCC dimensions?
  - [ ] **Issue**: MaxPool may lose temporal detail
  - [ ] Adaptive pooling to (1,1) too aggressive?
  - [ ] Final FC layers appropriate size (128→256→num_classes)?
  - [ ] Dropout rate (0.5) optimal or too high?
  - [ ] **Parameter Count**: ~323K params (verified?)
  - [ ] **Verification**:
    ```python
    model = AudioCNN(num_classes=143)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dummy = torch.randn(1, 3, 40, 130)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    ```

- [ ] **AudioViT Architecture**
  - [ ] Input resized from (3,40,~130) to (3,224,224)?
  - [ ] **Critical Issue**: Resizing destroys temporal structure!
  - [ ] Pretrained ViT weights from visual domain appropriate?
  - [ ] **Red Flag**: ViT expects square images, MFCC is rectangular
  - [ ] Interpolation method for resizing justified?
  - [ ] Fine-tuning strategy (frozen layers vs. full fine-tune)?

#### ✅ Image Models (Strong Performance)
- [ ] **ImageResNet**
  - [ ] Pretrained weights loaded correctly?
  - [ ] Final FC layer replaced with correct num_classes?
  - [ ] Architecture appropriate for bird images?

- [ ] **ImageViT**
  - [ ] Pretrained ViT-B/16 loaded correctly?
  - [ ] Classification head replaced?
  - [ ] Selective unfreezing strategy reasonable?

### 1.4 Training Requirements

#### ⚠️ Training Configuration
- [ ] **Hyperparameters Documented**
  - [ ] Learning rate values recorded?
  - [ ] Batch size consistent and justified?
  - [ ] Number of epochs sufficient?
  - [ ] Optimizer choice (Adam/AdamW/SGD)?
  - [ ] Weight decay value?
  - [ ] **Missing**: No hyperparameter config file or documentation

- [ ] **Training Features**
  - [ ] Mixed precision (AMP) enabled?
  - [ ] Gradient clipping applied (prevent explosion)?
  - [ ] Early stopping implemented (patience value)?
  - [ ] Learning rate scheduling used?
  - [ ] Checkpointing saves best model (by val accuracy or val loss)?

- [ ] **Reproducibility**
  - [ ] Random seeds fixed (Python, NumPy, PyTorch)?
  - [ ] Deterministic operations enabled (torch.backends.cudnn)?
  - [ ] Same environment across runs (uv.lock)?

#### ⚠️ Class Imbalance Handling
- [ ] **Critical Issue**: F1-macro much lower than accuracy
  - [ ] Are class weights used in loss function?
  - [ ] Is oversampling/undersampling applied?
  - [ ] **Action Required**: Analyze class distribution
  - [ ] **Verification**:
    ```python
    df = pd.read_parquet("artifacts/xeno_canto_filtered.parquet")
    counts = df['species_normalized'].value_counts()
    print(f"Min samples: {counts.min()}")
    print(f"Max samples: {counts.max()}")
    print(f"Ratio: {counts.max() / counts.min():.1f}x")
    ```

### 1.5 Evaluation Requirements

#### ⚠️ Metrics Validation
- [ ] **Accuracy Computation**
  - [ ] Correct implementation (predictions == labels)?
  - [ ] Computed on test set only?
  - [ ] No data leakage from train/val?

- [ ] **F1 Scores**
  - [ ] F1-macro: unweighted average across classes?
  - [ ] F1-weighted: weighted by support?
  - [ ] **Issue**: Large gap between F1-macro and F1-weighted indicates class imbalance
  - [ ] Per-class F1 scores computed and saved?

- [ ] **Confusion Matrix**
  - [ ] Normalized confusion matrix generated?
  - [ ] Saved as image file?
  - [ ] Interpretable for many classes (143 species)?

#### ⚠️ Results Storage
- [ ] **Results Files**
  - [ ] JSON files saved for all 4 models?
  - [ ] Contains: accuracy, f1_macro, f1_weighted?
  - [ ] Contains: predictions and labels arrays?
  - [ ] Summary JSON aggregates all models?
  - [ ] Files exist and readable:
    - `artifacts/results/audio_cnn_results.json`
    - `artifacts/results/audio_vit_results.json`
    - `artifacts/results/image_resnet18_results.json`
    - `artifacts/results/image_vit_results.json`
    - `artifacts/results/results_summary.json`

---

## 2. Clarity Checklist

### 2.1 Code Documentation

#### ⚠️ Docstrings
- [ ] **Module-level docstrings**
  - [ ] All modules have purpose description?
  - [ ] Key functions/classes listed?

- [ ] **Function/Class docstrings**
  - [ ] Args types and descriptions clear?
  - [ ] Returns types and descriptions clear?
  - [ ] Raises exceptions documented?
  - [ ] Examples provided for complex functions?

- [ ] **Inline comments**
  - [ ] Complex logic explained?
  - [ ] Magic numbers explained (e.g., 40 MFCCs, 224 image size)?
  - [ ] TODOs marked for future improvements?

#### ⚠️ Naming Conventions
- [ ] **Variable names**
  - [ ] Descriptive (not `x`, `y`, `tmp`)?
  - [ ] Consistent (snake_case for Python)?
  - [ ] Abbreviations explained (MFCC, SR, ViT)?

- [ ] **File/Module names**
  - [ ] Purpose clear from name?
  - [ ] Consistent structure (dataset vs. datasets)?

### 2.2 Notebook Clarity

#### ✅ Notebook Structure (05_evaluate.ipynb)
- [ ] **Markdown cells**
  - [ ] Clear section headers?
  - [ ] Objectives stated upfront?
  - [ ] Explanations for each step?
  - [ ] Results interpretation provided?

- [ ] **Code cells**
  - [ ] Logical grouping (one task per cell)?
  - [ ] Output suppressed when not needed?
  - [ ] Imports organized (stdlib, third-party, local)?

- [ ] **Visualizations**
  - [ ] Plots have titles and labels?
  - [ ] Confusion matrices readable?
  - [ ] Comparison plots clear?

### 2.3 Error Messages

#### ⚠️ Error Handling
- [ ] **Informative errors**
  - [ ] File not found errors include path?
  - [ ] Shape mismatches include expected vs. actual?
  - [ ] Missing dependencies suggest installation?

- [ ] **Validation checks**
  - [ ] Input validation before processing?
  - [ ] Early failure with clear message?
  - [ ] Suggestions for fixing issues?

---

## 3. Consistency Checklist

### 3.1 Data Consistency

#### ⚠️ Species Names
- [ ] **Normalization**
  - [ ] Same normalization applied everywhere?
  - [ ] Function: `src/utils/species.py::normalize_species_name()`
  - [ ] Used consistently in indexing, splitting, feature extraction?
  - [ ] **Test**: Check species names in all parquet files match

#### ⚠️ File Paths
- [ ] **Path handling**
  - [ ] All paths use `pathlib.Path`?
  - [ ] Absolute vs. relative paths clear?
  - [ ] Platform-independent (Windows, Linux, macOS)?

- [ ] **Directory structure**
  - [ ] Consistent naming (snake_case)?
  - [ ] Artifacts directory structure documented?
  - [ ] Cache structure matches expectations?

### 3.2 Hyperparameter Consistency

#### ⚠️ Audio Parameters
- [ ] **MFCC extraction**
  - [ ] Same parameters in notebook and script?
  - [ ] Parameters documented in one place?
  - [ ] **Check**: Compare `02_audio_features.ipynb` vs. `src/features/audio.py`

- [ ] **Audio duration**
  - [ ] Consistent 3.0 seconds everywhere?
  - [ ] Padding/cropping strategy same?

#### ⚠️ Training Parameters
- [ ] **Seeds**
  - [ ] SEED = 42 used everywhere?
  - [ ] Set before splitting, feature extraction, training?

- [ ] **Batch sizes**
  - [ ] Consistent across train/val/test?
  - [ ] Same for audio and image models (if hardware allows)?

- [ ] **Learning rates**
  - [ ] Documented for each model?
  - [ ] Different for CNN vs. ViT justified?

### 3.3 Code Style Consistency

#### ⚠️ Formatting
- [ ] **Style guide**
  - [ ] Follow PEP 8?
  - [ ] Line length limit (79 or 120)?
  - [ ] Imports sorted?

- [ ] **Type hints**
  - [ ] Consistent use of type hints?
  - [ ] All public functions annotated?
  - [ ] Return types specified?

### 3.4 Version Consistency

#### ⚠️ Dependencies
- [ ] **Package versions**
  - [ ] Locked in `uv.lock`?
  - [ ] Compatible versions (PyTorch + CUDA)?
  - [ ] Recorded in `artifacts/env.json`?

- [ ] **Python version**
  - [ ] Python 3.12 consistently used?
  - [ ] No incompatible features used?

---

## 4. Critical Issues to Address Before Audio Improvements

### 🚨 Priority 1: Audio Feature Representation (CRITICAL)

**Issue**: MFCC features may not capture sufficient information

**Validation Steps**:
1. [ ] **Visualize MFCC features** for different species
   ```python
   import matplotlib.pyplot as plt
   import librosa.display
   
   sample = np.load("artifacts/audio_mfcc_cache/xeno_canto/american_crow/<id>.npy")
   plt.figure(figsize=(12, 8))
   plt.subplot(3, 1, 1)
   librosa.display.specshow(sample[:,:,0], x_axis='time', y_axis='mel')
   plt.title('MFCC')
   plt.subplot(3, 1, 2)
   librosa.display.specshow(sample[:,:,1], x_axis='time', y_axis='mel')
   plt.title('Delta')
   plt.subplot(3, 1, 3)
   librosa.display.specshow(sample[:,:,2], x_axis='time', y_axis='mel')
   plt.title('Delta-Delta')
   plt.tight_layout()
   plt.show()
   ```

2. [ ] **Check feature statistics**
   - [ ] Are MFCC values normalized?
   - [ ] Do different species have distinguishable patterns?
   - [ ] Are deltas providing useful information?

3. [ ] **Test alternative representations**
   - [ ] Mel-spectrograms instead of MFCCs
   - [ ] Raw waveforms with 1D CNNs
   - [ ] Log-mel spectrograms
   - [ ] CQT (Constant-Q Transform) for bird calls

**Expected Outcome**: Determine if poor performance is due to features or model

---

### 🚨 Priority 2: AudioViT Input Resizing (CRITICAL)

**Issue**: Resizing (3, 40, 130) → (3, 224, 224) destroys temporal structure

**Validation Steps**:
1. [ ] **Visualize resized features**
   ```python
   import torch.nn.functional as F
   
   original = torch.from_numpy(sample).permute(2, 0, 1)  # (3, 40, 130)
   resized = F.interpolate(original.unsqueeze(0), size=(224, 224), mode='bilinear')
   
   # Compare original vs resized
   plt.subplot(1, 2, 1)
   plt.imshow(original[0])
   plt.title('Original 40x130')
   plt.subplot(1, 2, 2)
   plt.imshow(resized[0, 0])
   plt.title('Resized 224x224')
   plt.show()
   ```

2. [ ] **Check if ViT can handle non-square inputs**
   - [ ] Investigate ViT positional encoding
   - [ ] Test with rectangular inputs
   - [ ] Consider padding to square instead of resizing

3. [ ] **Alternative**: Convert MFCC to mel-spectrogram image
   - [ ] Use librosa to generate spectrogram as 224×224 image
   - [ ] Treat audio as "image" of spectrogram

**Expected Outcome**: Fix or replace AudioViT approach

---

### 🚨 Priority 3: Feature Normalization (HIGH PRIORITY)

**Issue**: No mention of MFCC normalization in code

**Validation Steps**:
1. [ ] **Check current feature ranges**
   ```python
   # Load multiple samples and check statistics
   samples = []
   for path in cache_dir.glob("*/*.npy")[:100]:
       samples.append(np.load(path))
   samples = np.array(samples)
   
   print(f"MFCC - Mean: {samples[:,:,:,0].mean():.2f}, Std: {samples[:,:,:,0].std():.2f}")
   print(f"Delta - Mean: {samples[:,:,:,1].mean():.2f}, Std: {samples[:,:,:,1].std():.2f}")
   print(f"Delta² - Mean: {samples[:,:,:,2].mean():.2f}, Std: {samples[:,:,:,2].std():.2f}")
   ```

2. [ ] **Implement normalization strategy**
   - [ ] Per-channel (MFCC, Delta, Delta²) standardization?
   - [ ] Global standardization across all features?
   - [ ] Min-max scaling to [0, 1] or [-1, 1]?

3. [ ] **Add normalization to dataset**
   ```python
   # In AudioMFCCDataset.__getitem__
   features = (features - mean) / (std + 1e-8)
   ```

**Expected Outcome**: Properly normalized features for stable training

---

### 🚨 Priority 4: Class Imbalance (HIGH PRIORITY)

**Issue**: F1-macro (0.11) << Accuracy (0.40) indicates severe class imbalance

**Validation Steps**:
1. [ ] **Analyze class distribution**
   ```python
   df = pd.read_parquet("artifacts/xeno_canto_filtered.parquet")
   counts = df['species_normalized'].value_counts()
   
   plt.figure(figsize=(12, 6))
   plt.hist(counts, bins=50)
   plt.xlabel('Samples per species')
   plt.ylabel('Number of species')
   plt.title('Audio Dataset Class Distribution')
   plt.show()
   
   print(f"Min: {counts.min()}, Max: {counts.max()}, Median: {counts.median()}")
   ```

2. [ ] **Check test set distribution**
   ```python
   with open("artifacts/splits/xeno_canto_audio_splits.json") as f:
       splits = json.load(f)
   
   test_df = df.iloc[splits['test']]
   test_counts = test_df['species_normalized'].value_counts()
   print(f"Species with <5 test samples: {(test_counts < 5).sum()}")
   ```

3. [ ] **Implement class balancing**
   - [ ] Weighted loss function (class weights inversely proportional to frequency)
   - [ ] Oversampling minority classes (WeightedRandomSampler)
   - [ ] Focal loss for hard examples

**Expected Outcome**: Identify extent of imbalance and mitigation strategy

---

### ⚠️ Priority 5: Audio Duration

**Issue**: 3 seconds may be insufficient for some bird vocalizations

**Validation Steps**:
1. [ ] **Analyze original audio durations**
   ```python
   durations = []
   for path in df['file_path']:
       y, sr = librosa.load(path, sr=None, duration=None)
       durations.append(len(y) / sr)
   
   plt.hist(durations, bins=50)
   plt.xlabel('Duration (seconds)')
   plt.ylabel('Count')
   plt.title('Original Audio Durations')
   plt.axvline(3.0, color='red', linestyle='--', label='Current cutoff')
   plt.legend()
   plt.show()
   
   print(f"Median: {np.median(durations):.1f}s")
   print(f"% < 3s: {(np.array(durations) < 3).mean()*100:.1f}%")
   ```

2. [ ] **Test with different durations**
   - [ ] Try 5 seconds, 7 seconds, 10 seconds
   - [ ] Use dynamic length with padding to max length in batch

**Expected Outcome**: Determine optimal audio duration

---

### ⚠️ Priority 6: Training Configuration Documentation

**Issue**: Hyperparameters not centrally documented

**Validation Steps**:
1. [ ] **Extract hyperparameters from notebook/script**
   - [ ] Check `03_train_audio.py` or training notebooks
   - [ ] Document learning rate, batch size, epochs, optimizer

2. [ ] **Create hyperparameter config file**
   ```python
   # configs/audio_cnn.yaml
   model:
     type: AudioCNN
     num_classes: 143
     dropout: 0.5
   
   training:
     learning_rate: 0.001
     batch_size: 32
     epochs: 100
     optimizer: Adam
     weight_decay: 0.0001
     early_stopping_patience: 10
   
   data:
     n_mfcc: 40
     hop_length: 512
     duration: 3.0
   ```

**Expected Outcome**: Reproducible and documented hyperparameters

---

## 5. Validation Execution Plan

### Phase 1: Data Validation (Day 1)
1. Run data indexing verification scripts
2. Check species intersection and filtering
3. Validate split generation and stratification
4. Verify no data leakage

### Phase 2: Feature Validation (Day 1-2)
1. Visualize MFCC features for 10 species
2. Check feature statistics (mean, std, range)
3. Verify cache integrity
4. Test feature normalization

### Phase 3: Model Validation (Day 2)
1. Check model architectures (forward pass, parameter count)
2. Verify checkpoint loading
3. Test inference on single batch
4. Validate output shapes

### Phase 4: Results Validation (Day 2-3)
1. Recompute metrics from saved predictions
2. Verify confusion matrix correctness
3. Analyze per-class performance
4. Identify systematic errors

### Phase 5: Documentation & Consistency (Day 3)
1. Review all docstrings
2. Check naming consistency
3. Verify hyperparameter consistency
4. Update documentation

---

## 6. Sign-off Checklist

### Before Proceeding to Audio Improvements:
- [ ] All Priority 1 issues investigated and addressed
- [ ] Feature representation validated or alternative identified
- [ ] Class imbalance quantified and mitigation planned
- [ ] Training configuration documented
- [ ] Results reproducible
- [ ] Code reviewed and cleaned
- [ ] Documentation updated

### Green Light Criteria:
- [ ] Can explain why audio models underperform
- [ ] Have hypothesis for improvement
- [ ] Have validation plan for new experiments
- [ ] Can reproduce current results
- [ ] Have baseline to compare against

---

## 7. Next Steps (After Validation)

### Immediate Experiments (Week 1):
1. **Mel-Spectrogram Features**: Replace MFCC with mel-spectrograms
2. **Feature Normalization**: Standardize features
3. **Class Balancing**: Implement weighted loss or oversampling
4. **Longer Audio Duration**: Test with 5-7 seconds

### Follow-up Experiments (Week 2):
1. **Architecture Search**: Try different CNN architectures
2. **Audio-Pretrained Models**: Use PANNs or other audio models
3. **Data Augmentation**: SpecAugment, mixup, time stretching
4. **Ensemble Methods**: Combine multiple models

### Evaluation (Ongoing):
1. Track all experiments in experiment tracking system (wandb, mlflow)
2. Compare against baseline (current results)
3. Maintain experiment log with hyperparameters and results
4. Document findings and insights

---

## Appendix A: Validation Scripts

### Script 1: Data Integrity Check
```python
#!/usr/bin/env python3
"""Validate data integrity."""

import pandas as pd
import json
from pathlib import Path

def validate_data():
    artifacts = Path("artifacts")
    
    # Check Xeno-Canto
    xc_df = pd.read_parquet(artifacts / "xeno_canto_filtered.parquet")
    print(f"✓ Xeno-Canto: {len(xc_df)} records, {xc_df['species_normalized'].nunique()} species")
    
    # Check CUB
    cub_df = pd.read_parquet(artifacts / "cub_filtered.parquet")
    print(f"✓ CUB: {len(cub_df)} images, {cub_df['species_normalized'].nunique()} species")
    
    # Check intersection
    with open(artifacts / "intersection_metadata.json") as f:
        meta = json.load(f)
    print(f"✓ Intersection: {meta['num_common_species']} species")
    
    # Check splits
    with open(artifacts / "splits/xeno_canto_audio_splits.json") as f:
        audio_splits = json.load(f)
    print(f"✓ Audio splits: {len(audio_splits['train'])}/{len(audio_splits['val'])}/{len(audio_splits['test'])}")
    
    # Check no overlap
    assert len(set(audio_splits['train']) & set(audio_splits['test'])) == 0
    print("✓ No data leakage in splits")

if __name__ == "__main__":
    validate_data()
```

### Script 2: Feature Statistics
```python
#!/usr/bin/env python3
"""Check MFCC feature statistics."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_feature_stats():
    cache_dir = Path("artifacts/audio_mfcc_cache/xeno_canto")
    
    samples = []
    for i, path in enumerate(cache_dir.glob("*/*.npy")):
        if i >= 100:  # Sample 100 files
            break
        samples.append(np.load(path))
    
    samples = np.array(samples)
    
    print("Feature Statistics:")
    print(f"  MFCC - Mean: {samples[:,:,:,0].mean():.2f}, Std: {samples[:,:,:,0].std():.2f}")
    print(f"  Delta - Mean: {samples[:,:,:,1].mean():.2f}, Std: {samples[:,:,:,1].std():.2f}")
    print(f"  Delta² - Mean: {samples[:,:,:,2].mean():.2f}, Std: {samples[:,:,:,2].std():.2f}")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, name in enumerate(['MFCC', 'Delta', 'Delta²']):
        axes[i].hist(samples[:,:,:,i].flatten(), bins=100, alpha=0.7)
        axes[i].set_title(f'{name} Distribution')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("artifacts/feature_distributions.png")
    print("✓ Feature distributions saved")

if __name__ == "__main__":
    check_feature_stats()
```

### Script 3: Class Balance Analysis
```python
#!/usr/bin/env python3
"""Analyze class distribution and imbalance."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_class_balance():
    df = pd.read_parquet("artifacts/xeno_canto_filtered.parquet")
    counts = df['species_normalized'].value_counts()
    
    print(f"Class Distribution:")
    print(f"  Total species: {len(counts)}")
    print(f"  Min samples: {counts.min()}")
    print(f"  Max samples: {counts.max()}")
    print(f"  Median samples: {counts.median()}")
    print(f"  Mean samples: {counts.mean():.1f}")
    print(f"  Imbalance ratio: {counts.max() / counts.min():.1f}x")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(counts, bins=50)
    plt.xlabel('Samples per species')
    plt.ylabel('Number of species')
    plt.title('Audio Dataset Class Distribution')
    plt.axvline(counts.median(), color='red', linestyle='--', label=f'Median: {counts.median():.0f}')
    plt.legend()
    plt.savefig("artifacts/class_distribution.png")
    print("✓ Class distribution plot saved")
    
    # Recommend class weights
    total = len(df)
    n_classes = len(counts)
    weights = total / (n_classes * counts)
    print(f"\n  Suggested class weight range: [{weights.min():.2f}, {weights.max():.2f}]")

if __name__ == "__main__":
    analyze_class_balance()
```

---

**End of Validation Checklist**

**Next Step**: Execute validation scripts and address critical issues before audio model improvements.
