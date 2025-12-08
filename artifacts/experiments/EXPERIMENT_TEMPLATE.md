# Experiment Tracking Template

Use this template for every experiment. Copy and fill out, save to `artifacts/experiments/`.

---

## Experiment Information

**Experiment ID**: `audio_exp_XXX`  
**Experiment Name**: [Descriptive name]  
**Date**: YYYY-MM-DD  
**Phase**: [1 / 2 / 3 / 4]  
**Experimenter**: [Your name]  
**Status**: [🔄 Running / ✅ Complete / ❌ Failed / ⏸️ Paused]

---

## Objective

**Goal**: [What are you trying to achieve?]

**Hypothesis**: [What do you expect to happen?]

**Expected Impact**: [+X% accuracy, +Y% F1-macro]

---

## Baseline

**Reference Model**: [e.g., AudioCNN baseline_v1]

| Metric | Baseline Value | Target Value |
|--------|---------------|--------------|
| Accuracy | 0.XXX | 0.XXX |
| F1-Macro | 0.XXX | 0.XXX |
| F1-Weighted | 0.XXX | 0.XXX |
| Precision | 0.XXX | 0.XXX |
| Recall | 0.XXX | 0.XXX |

**Baseline Location**: `artifacts/results/[baseline_name].json`

---

## Changes from Baseline

### Code Changes
1. [File modified and description of change]
2. [Another change]
3. [...]

### Data Changes
- [ ] New features extracted
- [ ] Different preprocessing
- [ ] Changed data splits
- [ ] Other: [describe]

### Model Changes
- [ ] Different architecture
- [ ] New layers added/removed
- [ ] Pretrained weights used
- [ ] Other: [describe]

### Training Changes
- [ ] Different loss function
- [ ] New optimizer
- [ ] Changed learning rate schedule
- [ ] Added data augmentation
- [ ] Other: [describe]

---

## Hyperparameters

### Model Architecture
```yaml
model_type: [AudioCNN / AudioViT / etc.]
num_classes: 90
input_shape: [3, 40, 130]  # [channels, height, width]
# Add model-specific parameters:
# - num_conv_layers: X
# - hidden_dim: X
# - dropout: X
# - etc.
```

### Training Configuration
```yaml
batch_size: 32
epochs: 50
learning_rate: 0.001
optimizer: Adam
optimizer_params:
  betas: [0.9, 0.999]
  weight_decay: 0.0001

scheduler: ReduceLROnPlateau
scheduler_params:
  factor: 0.5
  patience: 5
  min_lr: 1e-6

loss_function: CrossEntropyLoss
loss_params:
  # e.g., class_weights: True
  
early_stopping:
  enabled: true
  patience: 10
  monitor: val_f1_macro
  mode: max
```

### Data Configuration
```yaml
dataset: Xeno-Canto
train_samples: 7753
val_samples: 1661
test_samples: 1662

features:
  type: [MFCC / mel-spectrogram / raw waveform]
  # Add feature-specific params:
  # - n_mfcc: 40
  # - duration: 3.0
  # - sample_rate: 22050
  # - etc.

augmentation:
  enabled: [true / false]
  # If true, list augmentations:
  # - SpecAugment:
  #     freq_mask: 15
  #     time_mask: 35

normalization:
  enabled: [true / false]
  method: [standardization / min-max / per-channel]
```

### Computational Resources
```yaml
gpu: [CUDA device / CPU]
gpu_memory: [X GB]
training_time: [X hours]
num_workers: 4
mixed_precision: [true / false]
```

---

## Execution

### Commands Run
```bash
# Training command
python scripts/03_train_audio.py \
  --model AudioCNN \
  --use-class-weights \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001

# Evaluation command
python scripts/05_evaluate.py \
  --model-path artifacts/checkpoints/audio_cnn_best.pth \
  --model-type AudioCNN
```

### Files Modified
- `src/training/trainer.py` - Added class weights support
- `src/datasets/audio.py` - Added normalization
- `scripts/03_train_audio.py` - Added CLI arguments

### Files Created
- `artifacts/checkpoints/audio_exp_XXX_best.pth` - Best model checkpoint
- `artifacts/checkpoints/audio_exp_XXX_final.pth` - Final epoch checkpoint
- `artifacts/experiments/audio_exp_XXX_training_log.txt` - Full training log
- `artifacts/experiments/audio_exp_XXX_results.json` - Evaluation results

---

## Results

### Training Metrics

| Epoch | Train Loss | Val Loss | Val Accuracy | Val F1-Macro | Val F1-Weighted |
|-------|-----------|----------|--------------|--------------|-----------------|
| 1 | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| ... | ... | ... | ... | ... | ... |
| Best | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Final | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

**Best Epoch**: XX  
**Training Time**: X hours Y minutes  
**Convergence**: [Converged / Did not converge / Overfitting / Underfitting]

### Test Set Performance

| Metric | Value | vs Baseline | Change |
|--------|-------|-------------|--------|
| Accuracy | 0.XXX | 0.XXX | +X.X% |
| Precision | 0.XXX | 0.XXX | +X.X% |
| Recall | 0.XXX | 0.XXX | +X.X% |
| F1-Macro | 0.XXX | 0.XXX | +X.X% |
| F1-Weighted | 0.XXX | 0.XXX | +X.X% |

**Improvement Summary**:
- [ ] ✅ Meets success criteria (F1-macro >X.XX)
- [ ] ❌ Does not meet success criteria
- [ ] 🟡 Partial improvement

### Confusion Matrix Analysis

**Top 5 Best Classified Species**:
1. [Species name] - F1: 0.XXX
2. [Species name] - F1: 0.XXX
3. [Species name] - F1: 0.XXX
4. [Species name] - F1: 0.XXX
5. [Species name] - F1: 0.XXX

**Top 5 Worst Classified Species**:
1. [Species name] - F1: 0.XXX
2. [Species name] - F1: 0.XXX
3. [Species name] - F1: 0.XXX
4. [Species name] - F1: 0.XXX
5. [Species name] - F1: 0.XXX

**Common Confusions** (Top 3):
1. [Species A] confused with [Species B] - N times
2. [Species C] confused with [Species D] - N times
3. [Species E] confused with [Species F] - N times

### Artifacts Generated

- [ ] Training curves (loss, accuracy, F1)
- [ ] Confusion matrix visualization
- [ ] Per-class performance chart
- [ ] Sample predictions (correct + incorrect)
- [ ] Model checkpoint
- [ ] Full evaluation report

**Artifacts Location**: `artifacts/experiments/audio_exp_XXX/`

---

## Analysis

### Key Observations

1. **[Observation 1]**  
   [Detailed description of what happened]

2. **[Observation 2]**  
   [Detailed description]

3. **[Observation 3]**  
   [Detailed description]

### What Worked Well ✅

- [Thing that worked]
- [Another success]
- [...]

### What Didn't Work ❌

- [Thing that failed]
- [Another issue]
- [...]

### Unexpected Results 🤔

- [Surprising finding 1]
- [Surprising finding 2]
- [...]

### Comparison to Hypothesis

**Hypothesis**: [Restate original hypothesis]

**Reality**: [What actually happened]

**Explanation**: [Why did it differ from expectations?]

---

## Insights and Learnings

### Technical Insights

1. **[Insight 1]**  
   [Why this matters, what it teaches us]

2. **[Insight 2]**  
   [Explanation]

3. **[Insight 3]**  
   [Explanation]

### Data Insights

- [What did we learn about the data?]
- [Any patterns discovered?]
- [Quality issues identified?]

### Model Insights

- [What did we learn about the model?]
- [Architecture strengths/weaknesses?]
- [Training behavior patterns?]

---

## Next Steps

### Immediate Follow-up

**Priority 1 (High)**:
- [ ] [Next experiment to try]
- [ ] [Analysis to perform]
- [ ] [Fix to implement]

**Priority 2 (Medium)**:
- [ ] [Another task]
- [ ] [Another task]

**Priority 3 (Low)**:
- [ ] [Nice-to-have task]
- [ ] [Another optional task]

### Ideas for Future Experiments

1. **[Idea 1]**  
   Rationale: [Why this might help]  
   Expected impact: [+X%]  
   Effort: [Low / Medium / High]

2. **[Idea 2]**  
   Rationale: [...]  
   Expected impact: [...]  
   Effort: [...]

3. **[Idea 3]**  
   Rationale: [...]  
   Expected impact: [...]  
   Effort: [...]

### Questions Raised

1. [Question that needs investigation]
2. [Another question]
3. [...]

---

## Decision

### Outcome

- [ ] ✅ **SUCCESS** - Proceed with this approach
- [ ] ⚠️ **PARTIAL SUCCESS** - Modify and retry
- [ ] ❌ **FAILURE** - Abandon this approach
- [ ] 🔄 **NEEDS MORE DATA** - Inconclusive, run more experiments

### Justification

[Explain the decision based on results]

### Action Items

- [ ] [Action 1 based on results]
- [ ] [Action 2]
- [ ] [Action 3]

---

## References

### Related Experiments
- [Experiment ID] - [Brief description and outcome]
- [Experiment ID] - [Brief description and outcome]

### Papers/Resources Consulted
- [Paper title] - [Key insight used]
- [Blog post / tutorial] - [What was helpful]

### Code References
- [GitHub repo / paper repo] - [What was adapted]

---

## Reproducibility

### Environment
```bash
# Python version
python --version

# Key packages
torch==X.X.X
librosa==X.X.X
# ... etc
```

### Random Seeds
```python
random_seed: 42
torch.manual_seed(42)
np.random.seed(42)
```

### Exact Command to Reproduce
```bash
# Full command with all arguments
python scripts/03_train_audio.py \
  --model AudioCNN \
  --use-class-weights \
  --normalize-features \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --seed 42 \
  --save-dir artifacts/experiments/audio_exp_XXX
```

### Known Issues / Caveats
- [Any issues that might affect reproducibility]
- [Platform-specific notes]
- [Resource requirements]

---

## Sign-off

**Reviewed By**: [Name]  
**Review Date**: YYYY-MM-DD  
**Approved for**: [Production / Further experimentation / Archived]

**Notes**:
[Any final comments or recommendations]

---

## Metadata

**Template Version**: 1.0  
**Last Updated**: December 4, 2025  
**Document Status**: ✅ Complete

---

## Checklist Before Closing Experiment

- [ ] All results documented
- [ ] Artifacts saved and backed up
- [ ] Code changes committed to version control
- [ ] Insights summarized in experiment log
- [ ] Next experiments planned
- [ ] Model checkpoint saved (if successful)
- [ ] Confusion matrix analyzed
- [ ] Per-class performance reviewed
- [ ] Training curves visualized
- [ ] Decision documented
- [ ] Team notified of results (if applicable)

---

**Experiment Status**: [🔄 In Progress / ✅ Complete]  
**File Last Updated**: YYYY-MM-DD HH:MM
