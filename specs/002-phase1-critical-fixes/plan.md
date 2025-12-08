# Implementation Plan: Phase 1 - Critical Fixes

**Spec ID**: 002-phase1-critical-fixes  
**Plan Version**: 1.0  
**Created**: 2025-12-04  
**Status**: Ready for Implementation

---

## Overview

This plan details the implementation of class-weighted loss and feature normalization to address the primary root causes of poor audio model performance.

**Estimated Time**: 5-8 hours over 2 days  
**Complexity**: Low-Medium  
**Risk Level**: Low

---

## Implementation Steps

### Step 1: Modify Trainer Class (1 hour)

**File**: `src/training/trainer.py`

**Changes**:

1. Add `class_weights` parameter to `__init__`:
```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, 
                 optimizer, scheduler=None, device='cuda',
                 class_weights=None):  # NEW PARAMETER
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Create criterion with optional class weights
        if class_weights is not None:
            self.class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
```

2. No other changes needed - loss computation remains the same

**Validation**:
- Create dummy tensor: `torch.ones(90)`
- Instantiate Trainer with class_weights
- Verify no errors raised

---

### Step 2: Update Training Script (1-2 hours)

**File**: `scripts/03_train_audio.py`

**Changes**:

1. Add command-line argument:
```python
parser.add_argument('--use-class-weights', action='store_true',
                    help='Use class-balanced weighting in loss')
```

2. Load class weights from JSON:
```python
def load_class_weights(species_list, method='balanced'):
    """Load class weights from validation artifacts.
    
    Args:
        species_list: Ordered list of species names matching dataset
        method: 'balanced', 'effective', or 'sqrt'
    
    Returns:
        torch.FloatTensor of shape (num_classes,)
    """
    weights_path = 'artifacts/validation/recommended_class_weights.json'
    with open(weights_path, 'r') as f:
        weights_dict = json.load(f)[method]
    
    # Map to dataset species order
    class_weights = [weights_dict[species] for species in species_list]
    return torch.FloatTensor(class_weights)
```

3. Integrate into training setup:
```python
# After dataset creation
species_list = train_dataset.species_list  # or get from utils

class_weights = None
if args.use_class_weights:
    print("Loading class weights...")
    class_weights = load_class_weights(species_list, method='balanced')
    print(f"Class weights shape: {class_weights.shape}")
    print(f"Weight range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")

# Create trainer with weights
trainer = Trainer(model, train_loader, val_loader,
                  optimizer, scheduler, device,
                  class_weights=class_weights)
```

**Validation**:
- Run with `--use-class-weights`: verify weights loaded
- Check shape matches number of species (90)
- Verify weight range makes sense (should vary significantly)

---

### Step 3: Implement Feature Normalization (1 hour)

**File**: `src/datasets/audio.py`

**Changes**:

1. Add normalization parameter and statistics:
```python
class AudioMFCCDataset(Dataset):
    def __init__(self, audio_files, labels, cache_dir, 
                 species_to_idx, normalize=True):
        self.audio_files = audio_files
        self.labels = labels
        self.cache_dir = cache_dir
        self.species_to_idx = species_to_idx
        self.normalize = normalize
        
        # Normalization statistics from validation
        if self.normalize:
            self.mfcc_mean = -8.80
            self.mfcc_std = 62.53
            self.delta_mean = 0.02
            self.delta_std = 1.69
```

2. Normalize in `__getitem__`:
```python
def __getitem__(self, idx):
    # Load cached features
    features = self._load_cached_features(idx)  # Shape: (3, 40, 130)
    
    # Apply normalization
    if self.normalize:
        # Channel 0: MFCC
        features[0] = (features[0] - self.mfcc_mean) / (self.mfcc_std + 1e-8)
        # Channel 1: Delta
        features[1] = (features[1] - self.delta_mean) / (self.delta_std + 1e-8)
        # Channel 2: Delta² already normalized, skip
    
    label = self.labels[idx]
    return features, label
```

3. Update dataset creation in training script:
```python
train_dataset = AudioMFCCDataset(
    train_files, train_labels, cache_dir,
    species_to_idx, normalize=True  # Enable normalization
)
```

**Validation**:
```python
# Test normalization
loader = DataLoader(train_dataset, batch_size=32)
batch = next(iter(loader))
features, labels = batch

print("Batch statistics after normalization:")
print(f"  MFCC - mean: {features[:,0].mean():.4f}, std: {features[:,0].std():.4f}")
print(f"  Delta - mean: {features[:,1].mean():.4f}, std: {features[:,1].std():.4f}")
# Expected: mean ≈ 0, std ≈ 1
```

---

### Step 4: Test Implementation (30 min)

**Single Epoch Test**:
```bash
python scripts/03_train_audio.py \
    --model AudioCNN \
    --use-class-weights \
    --epochs 1 \
    --batch-size 32
```

**Checks**:
- No errors during training
- Loss values reasonable (not NaN or inf)
- Training completes successfully
- Checkpoint saved

---

### Step 5: Full Training Runs (2-4 hours GPU time)

**AudioCNN Training**:
```bash
python scripts/03_train_audio.py \
    --model AudioCNN \
    --use-class-weights \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --save-name baseline_v2_cnn_balanced_normalized
```

**AudioViT Training**:
```bash
python scripts/03_train_audio.py \
    --model AudioViT \
    --use-class-weights \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001 \
    --save-name baseline_v2_vit_balanced_normalized
```

**Expected Training Time**:
- AudioCNN: ~1-2 hours
- AudioViT: ~1-2 hours

**Can run in parallel if multiple GPUs available**

---

### Step 6: Evaluation (30 min)

**Run Evaluation**:
```bash
python scripts/05_evaluate.py
```

**Generate Comparison Report**:
```python
import json

results = {
    'baseline_v1': {
        'audio_cnn': {'accuracy': 0.395, 'f1_macro': 0.109},
        'audio_vit': {'accuracy': 0.390, 'f1_macro': 0.108}
    },
    'baseline_v2': {
        'audio_cnn': {...},  # New results
        'audio_vit': {...}   # New results
    },
    'improvements': {
        'f1_macro_delta': ...,
        'accuracy_delta': ...,
        'relative_improvement_pct': ...
    }
}

with open('artifacts/results/baseline_v2_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
```

**Analysis**:
- Compare confusion matrices (v1 vs v2)
- Per-class F1 scores for rare species
- Training curve stability
- Document findings

---

## Checkpoint Decision

### Go/No-Go Criteria

**✅ GO to Phase 2** if:
- F1-macro > 0.25 ✓
- Accuracy > 0.50 ✓
- Training stable ✓
- Rare species F1 > 0.10 ✓

**⚠️ INVESTIGATE** if:
- 0.20 < F1-macro < 0.25
- Training shows instability
- No improvement for rare species

**❌ NO-GO / DEBUG** if:
- F1-macro < 0.20
- Training fails or diverges
- Performance worse than baseline

---

## Rollback Plan

If Phase 1 fails:

1. **Disable class weights**: Test normalization alone
2. **Disable normalization**: Test class weights alone
3. **Try alternative weighting**: Use "effective" or "sqrt" methods
4. **Check for bugs**: Verify weight mapping, normalization stats
5. **Reduce learning rate**: If training unstable

---

## Code Quality Checklist

- [ ] Code follows project style guide
- [ ] Docstrings added for new functions
- [ ] Type hints where applicable
- [ ] No hardcoded paths (use config/args)
- [ ] Print statements for debugging removed
- [ ] Checkpoints saved with clear naming convention
- [ ] Results logged to JSON files
- [ ] Git commit with descriptive message

---

## Files Modified

- `src/training/trainer.py` - Add class_weights parameter
- `src/datasets/audio.py` - Add normalization
- `scripts/03_train_audio.py` - Add --use-class-weights flag
- `artifacts/results/baseline_v2_comparison.json` - New results file

**No breaking changes to existing code**

---

## Next Steps (After Phase 1)

If Phase 1 checkpoint met:

1. **Review Phase 1 results** document
2. **Update STATUS.md** with Phase 1 completion
3. **Plan Phase 2** (mel-spectrograms, augmentation)
4. **Create spec 003-phase2-feature-engineering**

If Phase 1 checkpoint not met:

1. **Debug and iterate** on fixes
2. **Try alternative approaches** (oversampling, focal loss)
3. **Re-assess expectations** based on data constraints
4. **Document learnings** for future reference

---

## Resources

**Validation Artifacts**:
- `artifacts/validation/recommended_class_weights.json`
- `artifacts/validation/feature_statistics.json`

**Documentation**:
- spec.md - Feature requirements
- quickstart.md - Immediate action steps
- VALIDATION_SUMMARY_EXECUTIVE.md - Root cause analysis

**Code References**:
- PyTorch CrossEntropyLoss docs: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- sklearn class_weight.compute_class_weight
