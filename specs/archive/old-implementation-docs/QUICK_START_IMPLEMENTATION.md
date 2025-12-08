# Implementation Quick Start

**Goal**: Improve audio model performance from 39.5% → 75-85% accuracy

## 🚀 Start Here (Next 2 Days - Phase 1)

### Critical Fixes to Implement

#### Fix 1: Class-Weighted Loss (2-3 hours) ⭐ HIGHEST PRIORITY

**Problem**: Severe class imbalance (1216:1 ratio) causing low F1-macro (0.109)  
**Solution**: Weight loss by inverse class frequency  
**Expected Impact**: F1-macro 0.109 → 0.25-0.35 (+130-220%)

**Quick Implementation**:
```bash
# 1. Modify src/training/trainer.py
# Add class_weights parameter to __init__ and criterion
```

```python
# In Trainer.__init__:
if class_weights is not None:
    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
```

```bash
# 2. Update scripts/03_train_audio.py
# Add argument and load weights
```

```python
# Add argument:
parser.add_argument('--use-class-weights', action='store_true')

# Load weights from validation artifacts:
if args.use_class_weights:
    import json
    with open('artifacts/validation/recommended_class_weights.json') as f:
        weights_data = json.load(f)
    class_weights_dict = weights_data['balanced']
    # Map to tensor in species order
    weights_list = [class_weights_dict[species] for species in species_list]
    class_weights = torch.tensor(weights_list, dtype=torch.float).to(device)
```

```bash
# 3. Test
python scripts/03_train_audio.py --model AudioCNN --use-class-weights
```

**Validation**: F1-macro should improve from 0.109 to >0.20

---

#### Fix 2: Feature Normalization (1-2 hours) ⭐ SECOND PRIORITY

**Problem**: MFCC features not normalized (mean=-8.8, std=62.5) → slow convergence  
**Solution**: Standardize features per channel  
**Expected Impact**: +5-10% accuracy, faster training

**Quick Implementation**:
```bash
# Modify src/datasets/audio.py
```

```python
# In AudioMFCCDataset.__init__:
self.normalize = normalize  # Add parameter (default True)
if normalize:
    self.mfcc_mean = -8.80
    self.mfcc_std = 62.53
    self.delta_mean = 0.02
    self.delta_std = 1.69

# In __getitem__, after loading features:
if self.normalize:
    features[0] = (features[0] - self.mfcc_mean) / (self.mfcc_std + 1e-8)
    features[1] = (features[1] - self.delta_mean) / (self.delta_std + 1e-8)
    # features[2] (Delta²) already normalized
```

**Validation**: Check batch statistics are mean≈0, std≈1

---

#### Fix 3: Retrain Baseline (2-4 hours training)

```bash
# Retrain AudioCNN with both fixes
python scripts/03_train_audio.py --model AudioCNN --use-class-weights --epochs 50

# Retrain AudioViT  
python scripts/03_train_audio.py --model AudioViT --use-class-weights --epochs 50
```

**Expected Results**:
- AudioCNN: 50-60% accuracy, F1-macro 0.25-0.35
- AudioViT: Similar or better

---

## 📋 Phase 1 Checklist (Days 1-2)

- [ ] Implement class-weighted loss
- [ ] Implement feature normalization  
- [ ] Retrain AudioCNN with fixes
- [ ] Retrain AudioViT with fixes
- [ ] Document baseline v2 results
- [ ] Verify F1-macro >0.25 (success criteria)

**If successful** → Proceed to Phase 2  
**If F1-macro <0.20** → Debug before continuing

---

## 🔬 Phase 2 Preview (Days 3-7)

After Phase 1 fixes are validated:

### Experiment 1: Mel-Spectrograms (2-3 days)
- Extract 224×224 mel-spectrograms  
- Train models on mel-specs
- **Expected**: +10-20% improvement

### Experiment 2: SpecAugment (1-2 days)
- Add time/frequency masking
- **Expected**: +5-10% improvement  

### Experiment 3: Longer Duration (1 day)
- Try 5-7 second audio
- **Expected**: +5-15% improvement

---

## 📊 Success Metrics

| Phase | Target Accuracy | Target F1-Macro | Timeline |
|-------|----------------|-----------------|----------|
| Phase 1 | 50-60% | 0.25-0.35 | 2 days |
| Phase 2 | 65-75% | 0.40-0.50 | 1 week |
| Phase 3 | 70-80% | 0.60-0.70 | 2 weeks |
| Phase 4 | 75-85% | 0.65-0.75 | 4 weeks |

---

## 🎯 Immediate Next Action

**RIGHT NOW** (30 minutes):
1. Review this quick start
2. Read detailed plan: `IMPLEMENTATION_PLAN.md`
3. Review validation results: `VALIDATION_SUMMARY_EXECUTIVE.md`

**TODAY** (2-3 hours):
1. Implement class-weighted loss (Fix 1)
2. Implement feature normalization (Fix 2)
3. Start retraining

**TOMORROW**:
1. Complete retraining
2. Evaluate results
3. Document baseline v2
4. **Phase 1 Checkpoint**: Go/No-Go decision

---

## 📚 Key Documentation

- **This file**: Quick start guide
- `IMPLEMENTATION_PLAN.md`: Detailed 4-phase plan
- `VALIDATION_SUMMARY_EXECUTIVE.md`: Why we're doing this
- `VALIDATION_RESULTS.md`: Detailed findings
- `artifacts/validation/`: Data, plots, recommended weights

---

## ⚠️ Important Notes

1. **Must fix class imbalance first** - It's 60% of the problem
2. **Don't skip to experiments** - Fixes must come before new features
3. **Track everything** - Use experiment tracking template
4. **Validate at checkpoints** - Don't proceed if Phase 1 fails

---

## 🆘 If Things Go Wrong

**Problem**: F1-macro doesn't improve after Fix 1  
**Solution**: 
- Verify weights loaded correctly
- Check species mapping matches training data
- Try different weighting methods (effective, sqrt)

**Problem**: Training unstable after Fix 2  
**Solution**:
- Verify normalization statistics
- Reduce learning rate
- Try batch normalization in model instead

**Problem**: Models still underperform after Phase 1  
**Solution**:
- Review confusion matrices for patterns
- Check rare species performance separately
- Consider data collection for very rare species

---

## 💡 Key Insights from Validation

1. **Class imbalance is PRIMARY cause** (1216:1 ratio)
   - House sparrow: 1216 samples (11% of dataset)
   - Hooded merganser: 1 sample
   - This explains why F1-macro (0.109) << accuracy (0.395)

2. **Image models work well** (92.6% accuracy)
   - Proves task is feasible
   - Data quality is good
   - Audio-specific issues

3. **Data integrity is solid** ✅
   - No data leakage
   - All files valid
   - Cache complete

4. **Fix order matters**
   - Class weights → biggest impact
   - Normalization → faster training
   - Better features → incremental gains

---

**Ready to start?** → Begin with Fix 1 (Class-Weighted Loss)

**Questions?** → Review `IMPLEMENTATION_PLAN.md` for detailed guidance

**Document Version**: 1.0  
**Last Updated**: December 4, 2025
