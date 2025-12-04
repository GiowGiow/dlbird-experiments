# Pre-Experimentation Validation Protocol

**Project**: SpeckitDLBird - Bird Species Classification  
**Phase**: Pre-Audio Improvement Validation  
**Date**: December 4, 2025  
**Objective**: Validate implementation before expanding experiments

---

## Executive Summary

Before proceeding with audio model improvements (mel-spectrograms, augmentation, architecture changes), we must validate that our current implementation is correct, complete, and well-understood. Poor audio performance (39.5% vs. 92.6% for images) could be due to:

1. **Feature representation issues** (MFCC may not be optimal)
2. **Implementation bugs** (normalization, resizing artifacts)
3. **Training issues** (class imbalance, hyperparameters)
4. **Fundamental limitations** (dataset quality, duration)

This validation protocol ensures we understand current limitations before investing in improvements.

---

## What Has Been Created

### 📋 Documentation

1. **`VALIDATION_CHECKLIST.md`** (56 sections, 100+ checks)
   - Comprehensive checklist organized by:
     - Requirements Completeness (data, features, models, training, evaluation)
     - Clarity (documentation, error handling)
     - Consistency (naming, parameters, versions)
   - Critical issues prioritized by impact
   - Actionable validation steps with code examples
   - Sign-off criteria before proceeding

2. **`QA_SUMMARY.md`** (This Document)
   - High-level overview of QA process
   - Critical issues identified
   - Validation execution plan
   - Success criteria and next steps
   - Experiment tracking template

3. **`README_VALIDATION.md`** (You're reading it!)
   - Quick start guide for validation
   - Step-by-step execution instructions
   - Interpretation guidelines

### 🔧 Validation Scripts

Located in `scripts/`:

1. **`validate_data.py`**
   - Checks dataset integrity (Xeno-Canto, CUB-200)
   - Validates species intersection
   - Verifies split integrity (no leakage)
   - Confirms cache completeness

2. **`validate_features.py`**
   - Analyzes feature shapes and consistency
   - Computes feature statistics
   - Checks normalization status
   - Detects quality issues (NaN, Inf, zeros)
   - Generates diagnostic visualizations

3. **`validate_class_balance.py`**
   - Quantifies class imbalance
   - Identifies rare vs. common species
   - Computes recommended class weights
   - Creates distribution visualizations
   - Analyzes split-level balance

4. **`run_all_validations.py`**
   - Master orchestration script
   - Runs all validations in sequence
   - Produces comprehensive report
   - Determines go/no-go decision

---

## Quick Start: Running Validations

### Step 1: Run All Validations

```bash
# From project root
cd /home/giovanni/ufmg/speckitdlbird

# Activate environment (if using virtual env)
# source venv/bin/activate

# Run all validations
python scripts/run_all_validations.py
```

This will:
- Check data integrity
- Analyze feature quality
- Quantify class imbalance
- Generate diagnostic plots
- Produce summary report

**Expected Time**: 5-10 minutes depending on dataset size

### Step 2: Review Outputs

```bash
# View validation artifacts
ls -lh artifacts/validation/

# Should contain:
# - feature_distributions.png
# - sample_features_*.png
# - class_distribution_audio_dataset.png
# - class_distribution_image_dataset.png
# - recommended_class_weights.json
```

### Step 3: Interpret Results

Open generated plots and review terminal output for:

✅ **Green Flags** (Good to proceed):
- All validation checks pass
- Feature shapes consistent
- No NaN/Inf values
- Class imbalance quantified
- Splits have no leakage

⚠️ **Yellow Flags** (Warnings, but OK):
- Some class imbalance (expected)
- Feature ranges not normalized (fixable)
- Minor missing data (<1%)

🚨 **Red Flags** (Must fix before proceeding):
- Data leakage between splits
- Corrupted features (NaN, Inf, all zeros)
- Severe inconsistencies in feature shapes
- Missing critical files

### Step 4: Review Checklist

```bash
# Open and review
cat VALIDATION_CHECKLIST.md | less

# Or use your preferred editor
code VALIDATION_CHECKLIST.md
```

Go through each section and mark items as complete or flag issues.

---

## Critical Issues to Investigate

### 🔴 Issue 1: MFCC Feature Quality

**Question**: Are MFCC features adequate for bird species discrimination?

**How to Check**:
1. Run `python scripts/validate_features.py`
2. Review `artifacts/validation/sample_features_*.png`
3. Look for:
   - Visual differences between species
   - Clear temporal patterns
   - Distinct frequency structures

**What to Look For**:
- ✅ Features show clear patterns that differ by species
- ⚠️ Features look similar across different species
- 🚨 Features are all zeros or show no structure

**Action If Issue Found**:
- Consider mel-spectrograms instead of MFCCs
- Try different MFCC parameters (more coefficients, different hop length)
- Validate audio preprocessing (loading, resampling, padding)

### 🔴 Issue 2: Feature Normalization

**Question**: Are features properly normalized for neural network training?

**How to Check**:
1. Run `python scripts/validate_features.py`
2. Look for "Normalization Check" section in output
3. Check if mean ≈ 0 and std ≈ 1 for each channel

**What to Look For**:
- ✅ MFCC: mean≈0, std≈1 (normalized)
- ⚠️ MFCC: mean>>0 or std>>1 (NOT normalized)
- 🚨 Vastly different scales between channels

**Action If Issue Found**:
```python
# Add to AudioMFCCDataset.__getitem__
# Compute global mean/std first, then:
features = (features - self.mean) / (self.std + 1e-8)
```

### 🔴 Issue 3: AudioViT Input Resizing

**Question**: Does resizing (3,40,130) → (3,224,224) destroy information?

**How to Check**:
1. Run visualization in notebook:
```python
import torch.nn.functional as F
import matplotlib.pyplot as plt

sample = np.load("path/to/sample.npy")
original = torch.from_numpy(sample).permute(2, 0, 1)  # (3, 40, 130)
resized = F.interpolate(original.unsqueeze(0), size=(224,224), mode='bilinear')[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(original[0], aspect='auto')
axes[0].set_title('Original 40×130')
axes[1].imshow(resized[0], aspect='auto')
axes[1].set_title('Resized 224×224')
plt.show()
```

**What to Look For**:
- ⚠️ Temporal structure smeared/blurred
- 🚨 Patterns no longer distinguishable

**Action If Issue Found**:
- Use mel-spectrograms rendered at 224×224 directly
- Pad instead of resize
- Use audio-specific transformers (not ViT)

### 🔴 Issue 4: Class Imbalance

**Question**: Is class imbalance causing low F1-macro scores?

**How to Check**:
1. Run `python scripts/validate_class_balance.py`
2. Check "Imbalance ratio" in output
3. Review `artifacts/validation/class_distribution_*.png`

**What to Look For**:
- ✅ Imbalance < 5x (acceptable)
- ⚠️ Imbalance 5-20x (needs mitigation)
- 🚨 Imbalance > 20x (severe, must address)

**Action If Issue Found**:
```python
# Option 1: Weighted loss
class_weights = torch.tensor(weights_list).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Option 2: Weighted sampler
from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(dataset))
dataloader = DataLoader(dataset, sampler=sampler, ...)
```

---

## Validation Decision Tree

```
┌─────────────────────────────────────┐
│ Run scripts/run_all_validations.py │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │ All checks    │
       │ passed?       │
       └───┬───────┬───┘
           │       │
       Yes │       │ No
           │       │
           ▼       ▼
    ┌──────────┐  ┌──────────────────┐
    │ Review   │  │ Are issues       │
    │ warnings │  │ critical?        │
    └────┬─────┘  └────┬──────┬──────┘
         │             │      │
         │         Yes │      │ No
         │             │      │
         ▼             ▼      ▼
    ┌──────────┐  ┌─────────┐ ┌──────────┐
    │ Document │  │ Fix bugs│ │ Document │
    │ baseline │  │ first!  │ │ & proceed│
    └────┬─────┘  └────┬────┘ └────┬─────┘
         │             │            │
         └─────────────┴────────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Mark VALIDATION_   │
             │ CHECKLIST complete │
             └────────┬───────────┘
                      │
                      ▼
             ┌────────────────────┐
             │ Proceed with audio │
             │ improvements       │
             └────────────────────┘
```

---

## Success Criteria

### ✅ Green Light to Proceed

All of the following must be true:

1. **Data Validation**
   - ✅ No data leakage (train/val/test splits separate)
   - ✅ Species intersection computed correctly
   - ✅ All expected files present and readable
   - ✅ Feature cache complete and valid

2. **Feature Validation**
   - ✅ Feature shapes consistent
   - ✅ No corrupted features (NaN, Inf)
   - ✅ Feature statistics documented
   - ✅ Normalization status understood

3. **Class Balance**
   - ✅ Imbalance quantified
   - ✅ Mitigation strategy planned
   - ✅ Class weights computed

4. **Understanding**
   - ✅ Can explain current poor audio performance
   - ✅ Have hypotheses for improvement
   - ✅ Know what experiments to run next

### ⚠️ Proceed with Caution

Some warnings acceptable if:

- Non-critical warnings documented
- Workarounds identified
- Risks understood and accepted

### 🚨 Do Not Proceed

If any of these are true:

- Data leakage detected
- Severe feature corruption
- Cannot reproduce current results
- Critical bugs in pipeline

---

## After Validation: Next Steps

### Immediate (This Week)

1. **Fix Critical Issues**
   ```bash
   # If normalization needed
   python scripts/add_feature_normalization.py
   
   # If class weights needed
   python scripts/compute_class_weights.py
   ```

2. **Document Baseline**
   - Mark all checklist items complete
   - Save validation reports
   - Document known issues

3. **Plan Experiments**
   - Review `05_evaluate.ipynb` recommendations
   - Prioritize by expected impact
   - Set up experiment tracking

### Short-Term (Next 2 Weeks)

1. **Experiment 1: Mel-Spectrograms**
   - Goal: Replace MFCC with mel-spectrograms
   - Expected: 10-20% accuracy improvement
   - Timeline: 2-3 days

2. **Experiment 2: Class Balancing**
   - Goal: Improve F1-macro score
   - Expected: F1-macro from 0.11 → 0.25+
   - Timeline: 1-2 days

3. **Experiment 3: Audio Augmentation**
   - Goal: Regularization and generalization
   - Expected: 5-10% improvement
   - Timeline: 2-3 days

### Medium-Term (Next Month)

1. Architecture search
2. Audio-pretrained models
3. Multi-modal fusion
4. Hyperparameter optimization

---

## Troubleshooting

### Problem: Validation script fails with import error

**Solution**:
```bash
# Make sure you're in project root
cd /home/giovanni/ufmg/speckitdlbird

# Check Python path
python -c "import sys; print(sys.path)"

# Run with explicit path
PYTHONPATH=/home/giovanni/ufmg/speckitdlbird python scripts/validate_data.py
```

### Problem: Cannot find cached features

**Solution**:
```bash
# Check if cache exists
ls artifacts/audio_mfcc_cache/xeno_canto/

# If missing, regenerate
python scripts/02_splits_and_features.py
# or run notebook: notebooks/02_audio_features.ipynb
```

### Problem: Plots not generating

**Solution**:
```bash
# Install matplotlib if missing
pip install matplotlib seaborn

# Check if validation directory exists
mkdir -p artifacts/validation

# Run with verbose output
python scripts/validate_features.py --verbose
```

---

## Contact and Support

If validation reveals unexpected issues:

1. **Document the Issue**
   - What validation failed?
   - What was the error message?
   - What data exhibits the problem?

2. **Check Known Issues**
   - Review `VALIDATION_CHECKLIST.md` critical issues
   - Check `QA_SUMMARY.md` for similar problems

3. **Debug Systematically**
   - Isolate the problem (data vs. code vs. environment)
   - Test with minimal example
   - Compare with working baseline

---

## Summary

This validation protocol ensures:

✅ **Correctness**: Implementation is bug-free and working as intended  
✅ **Completeness**: All components present and functional  
✅ **Understanding**: Know why current performance is poor  
✅ **Readiness**: Prepared for systematic experimentation  

**Next Action**: Run `python scripts/run_all_validations.py` and review outputs!

---

**Document Version**: 1.0  
**Last Updated**: December 4, 2025  
**Status**: Ready for execution
