# Planning Documentation Index

**Project**: SpeckitDLBird Audio Model Improvements  
**Status**: ⚠️ REORGANIZED - Now following SpecKit methodology  
**Date**: December 4, 2025  
**Last Updated**: December 4, 2025

---

## 🔄 Repository Reorganization Notice

**This repository has been reorganized to follow SpecKit best practices!**

All specifications, plans, and feature documentation now live under **`.specify/specs/`**:
- `001-validation-phase/` - Validation & root cause analysis
- `002-phase1-critical-fixes/` - Class weights + normalization implementation

**→ See [REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md) for full details**  
**→ See [.specify/specs/README.md](.specify/specs/README.md) for navigation guide**

---

## 📋 Quick Navigation

### For Immediate Action (START HERE)
- **[QUICK_START_IMPLEMENTATION.md](QUICK_START_IMPLEMENTATION.md)** ⭐ - Start here for immediate next steps
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Detailed 4-phase implementation plan

### For Context and Understanding
- **[VALIDATION_SUMMARY_EXECUTIVE.md](VALIDATION_SUMMARY_EXECUTIVE.md)** - Why we're doing this (executive summary)
- **[VALIDATION_RESULTS.md](VALIDATION_RESULTS.md)** - Detailed validation findings (400+ lines)
- **[VALIDATION_COMPLETE.md](VALIDATION_COMPLETE.md)** - What we learned from validation

### For Reference
- **[VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md)** - Complete quality checklist (100+ items)
- **[artifacts/experiments/EXPERIMENT_TEMPLATE.md](artifacts/experiments/EXPERIMENT_TEMPLATE.md)** - Template for tracking experiments

### Supporting Documentation
- **[README_VALIDATION.md](README_VALIDATION.md)** - How to run validations
- **[QUICK_START_VALIDATION.md](QUICK_START_VALIDATION.md)** - Quick validation guide
- **[QA_SUMMARY.md](QA_SUMMARY.md)** - QA approach overview

---

## 🎯 Current Situation Summary

### Problem
Audio models underperforming dramatically:
- Audio: 39.5% accuracy, F1-macro 0.109
- Image: 92.6% accuracy, F1-macro 0.925
- **Gap**: 2.3x performance difference

### Root Causes (Validated)
1. 🔴 **SEVERE class imbalance** (1216:1 ratio) - 60% of problem
2. 🟡 **Missing feature normalization** - 20% of problem  
3. 🟢 **Suboptimal features** (MFCC limitations) - 20% of problem

### Solution Approach
4-phase implementation plan over ~1 month:
- **Phase 1** (Days 1-2): Fix critical issues → 50-60% accuracy
- **Phase 2** (Days 3-7): Feature engineering → 65-75% accuracy
- **Phase 3** (Weeks 2-3): Architecture optimization → 70-80% accuracy
- **Phase 4** (Weeks 4-5): Advanced techniques → 75-85% accuracy

---

## 📚 Document Overview

### Planning Documents (Created Today)

#### 1. IMPLEMENTATION_PLAN.md
**Purpose**: Comprehensive implementation roadmap  
**Audience**: Technical team, developers  
**Length**: ~1000 lines  
**Content**:
- 4 implementation phases with detailed tasks
- Task-by-task breakdown with time estimates
- Code examples for each fix
- Success criteria and checkpoints
- Risk management strategies
- Resource requirements
- Timeline and milestones

**When to use**: Detailed planning, task assignment, progress tracking

---

#### 2. QUICK_START_IMPLEMENTATION.md
**Purpose**: Immediate action guide  
**Audience**: Developer starting work today  
**Length**: ~250 lines  
**Content**:
- Phase 1 quick start (next 2 days)
- Step-by-step fix implementations
- Copy-paste code snippets
- Validation checks
- Troubleshooting guide

**When to use**: Right now, to start implementing fixes

---

#### 3. EXPERIMENT_TEMPLATE.md
**Purpose**: Standardized experiment tracking  
**Audience**: Anyone running experiments  
**Length**: ~400 lines  
**Content**:
- Complete experiment documentation template
- Sections: objectives, methods, results, analysis
- Reproducibility information
- Decision tracking

**When to use**: For every experiment (copy, fill out, save to `artifacts/experiments/`)

---

### Validation Documents (Created Yesterday)

#### 4. VALIDATION_SUMMARY_EXECUTIVE.md
**Purpose**: Executive summary of validation findings  
**Audience**: Stakeholders, decision makers  
**Length**: ~100 lines  
**Content**:
- Key findings in plain language
- Immediate action items
- Expected improvements
- Resources required

**When to use**: Quick overview, stakeholder briefings

---

#### 5. VALIDATION_RESULTS.md
**Purpose**: Comprehensive validation report  
**Audience**: Technical team  
**Length**: ~400 lines  
**Content**:
- Detailed results for all validation checks
- Critical issues with code fixes
- Validation artifacts list
- Sign-off checklist
- Technical analysis

**When to use**: Deep dive into validation findings, technical reference

---

#### 6. VALIDATION_COMPLETE.md
**Purpose**: Learnings and insights document  
**Audience**: Team, future reference  
**Length**: ~150 lines  
**Content**:
- What we learned from validation
- Why problems occurred
- How to prevent in future
- Key takeaways

**When to use**: Post-mortem review, learning from mistakes

---

#### 7. VALIDATION_CHECKLIST.md
**Purpose**: Comprehensive quality checklist  
**Audience**: QA, validators  
**Length**: ~300 lines, 100+ items  
**Content**:
- Requirements completeness checks
- Clarity validation items
- Consistency verification
- Critical issues prioritized

**When to use**: Running validations, quality assurance

---

### Supporting Documents

#### 8. README_VALIDATION.md
**Purpose**: How-to guide for validation framework  
**Content**: Installation, usage, interpretation of validation scripts

#### 9. QUICK_START_VALIDATION.md  
**Purpose**: Quick reference for running validations  
**Content**: One-command validation execution

#### 10. QA_SUMMARY.md
**Purpose**: QA approach overview  
**Content**: Validation methodology, philosophy, best practices

---

## 🗂️ Validation Artifacts

### Generated During Validation

**Location**: `artifacts/validation/`

**Visual Artifacts** (7 PNG files):
1. `feature_distributions.png` - MFCC distribution (shows non-normalized data)
2. `sample_features_0.png` - Sample visualization
3. `sample_features_1.png` - Sample visualization  
4. `sample_features_2.png` - Sample visualization
5. `class_distribution_audio_dataset.png` - **CRITICAL**: Shows 1216:1 imbalance
6. `class_distribution_image_dataset.png` - Shows balanced image data
7. Individual species plots

**Data Artifacts** (1 JSON file):
- `recommended_class_weights.json` - Pre-computed weights for 3 methods:
  - `balanced`: sklearn compute_class_weight method (RECOMMENDED)
  - `effective`: Effective number of samples method
  - `sqrt`: Square root inverse frequency method

**Usage**: Load directly in training scripts for class-weighted loss

---

## 🚀 Implementation Roadmap

### Phase 1: Critical Fixes (Days 1-2)
**Status**: 📋 Not Started  
**Priority**: 🔴 CRITICAL

#### Tasks:
1. **Implement class-weighted loss** (2-3 hours)
   - Code location: `src/training/trainer.py`, `scripts/03_train_audio.py`
   - Artifacts: Use `artifacts/validation/recommended_class_weights.json`
   - Expected: F1-macro 0.109 → 0.25-0.35

2. **Implement feature normalization** (1-2 hours)
   - Code location: `src/datasets/audio.py`
   - Stats: MFCC mean=-8.80, std=62.53; Delta mean=0.02, std=1.69
   - Expected: +5-10% accuracy

3. **Retrain baseline models** (2-4 hours training)
   - Models: AudioCNN, AudioViT
   - Save as: baseline_v2_balanced_normalized
   - Checkpoint: F1-macro must be >0.25 to proceed

**Documentation**: See [QUICK_START_IMPLEMENTATION.md](QUICK_START_IMPLEMENTATION.md)

---

### Phase 2: Feature Engineering (Days 3-7)
**Status**: 📋 Planned  
**Priority**: 🔴 HIGH

#### Experiments:
1. **Mel-spectrograms** (2-3 days)
   - Extract 224×224 mel-specs
   - Expected: +10-20% improvement
   
2. **SpecAugment** (1-2 days)
   - Time/frequency masking
   - Expected: +5-10% improvement

3. **Longer duration** (1 day)
   - Try 5-7 seconds vs 3 seconds
   - Expected: +5-15% improvement

**Documentation**: See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) Phase 2

---

### Phase 3: Architecture Optimization (Weeks 2-3)
**Status**: 📋 Planned  
**Priority**: 🟡 MEDIUM

#### Focus Areas:
- Audio-pretrained models (PANNs, AudioMAE)
- Architecture search (EfficientNet, ResNet variants)
- Hyperparameter tuning

**Expected**: 70-80% accuracy, F1-macro 0.60-0.70

---

### Phase 4: Advanced Techniques (Weeks 4-5)
**Status**: 📋 Planned  
**Priority**: 🟢 MEDIUM-LOW

#### Approaches:
- Multi-modal fusion (audio + image)
- Ensemble methods
- Final optimization

**Expected**: 75-85% accuracy, F1-macro 0.65-0.75

---

## 📊 Success Metrics

### Current Baseline (v1)
| Model | Accuracy | F1-Macro | F1-Weighted |
|-------|----------|----------|-------------|
| AudioCNN | 39.5% | 0.109 | 0.332 |
| AudioViT | 34.4% | 0.161 | 0.317 |
| ImageResNet18 | 92.6% | 0.925 | 0.926 |
| ImageViT | 92.0% | 0.918 | 0.920 |

### Target Metrics by Phase
| Phase | Target Accuracy | Target F1-Macro | Timeline |
|-------|----------------|-----------------|----------|
| Phase 1 | 50-60% | 0.25-0.35 | Day 2 |
| Phase 2 | 65-75% | 0.40-0.50 | Day 7 |
| Phase 3 | 70-80% | 0.60-0.70 | Day 20 |
| Phase 4 | 75-85% | 0.65-0.75 | Day 30 |

### Go/No-Go Checkpoints
- **Phase 1**: Must achieve F1-macro >0.25 or investigate further
- **Phase 2**: Must achieve accuracy >0.60 or re-evaluate approach
- **Phase 3**: Must achieve accuracy >0.70 or skip Phase 4

---

## 🛠️ Tools and Scripts

### Validation Scripts (In `scripts/`)
- `validate_data.py` - Data integrity checks ✅
- `validate_features.py` - Feature quality analysis ✅
- `validate_class_balance.py` - Class distribution analysis ✅
- `run_all_validations.py` - Master orchestration script ✅

**Status**: All scripts tested and working

### Training Scripts (In `scripts/`)
- `03_train_audio.py` - Audio model training ⚠️ Needs updates for Phase 1
- `04_train_image.py` - Image model training ✅ Working
- `05_evaluate.py` - Model evaluation ✅ Working

**Updates Needed**:
- Add `--use-class-weights` argument
- Add `--normalize-features` argument
- Implement class weight loading logic

---

## 📝 Experiment Tracking

### System
Use experiment tracking template for all experiments:
1. Copy `artifacts/experiments/EXPERIMENT_TEMPLATE.md`
2. Rename to `exp_XXX_description.md`
3. Fill out all sections
4. Save results and artifacts
5. Update experiment log

### Experiment Log
**Location**: `artifacts/experiments/experiment_log.md`  
**Format**: Chronological list with summaries

**Entry Template**:
```
### Experiment XXX: [Name]
- Date: YYYY-MM-DD
- Phase: X
- Change: [Brief description]
- Result: [Success/Failure with metrics]
- Next: [What to try next]
```

---

## 🎓 Key Insights

### From Validation Process

1. **Class imbalance is PRIMARY cause**
   - 1216:1 ratio explains F1-macro << accuracy
   - House sparrow (1216 samples) vs hooded merganser (1 sample)
   - Must fix before trying other improvements

2. **Image models prove task is feasible**
   - 92.6% accuracy shows data quality is good
   - Problem is audio-specific, not fundamental
   - Audio can likely reach 75-85% with proper fixes

3. **Feature normalization matters**
   - MFCC mean=-8.80, std=62.53 (not normalized)
   - Can slow training and affect convergence
   - Simple fix with big impact

4. **Data integrity is solid**
   - No data leakage detected
   - All caches valid and complete
   - Can trust evaluation results

### Planning Principles

1. **Fix root causes first** - Don't waste time on experiments without baseline fixes
2. **Validate at checkpoints** - Go/No-Go decisions prevent wasted effort
3. **Track everything** - Systematic tracking enables learning and iteration
4. **Incremental approach** - Build on successes, fail fast on failures
5. **Realistic expectations** - Data limitations will cap maximum performance

---

## ⚠️ Important Warnings

### Before Starting Implementation

1. **DO NOT skip Phase 1** - Critical fixes must come first
2. **DO NOT try experiments before baseline** - Will not show true impact
3. **DO validate after each phase** - Prevents compounding issues
4. **DO track all experiments** - Enables systematic learning
5. **DO review checkpoints** - Go/No-Go decisions save time

### Common Pitfalls to Avoid

1. Implementing wrong class weight mapping (verify species order!)
2. Forgetting to normalize test data the same as training data
3. Not saving baseline results before trying improvements
4. Trying too many changes at once (can't identify what worked)
5. Ignoring validation failures (check confusion matrices!)

---

## 📞 Quick Reference

### File Locations

**Documentation**:
- Planning: Root directory (`IMPLEMENTATION_PLAN.md`, `QUICK_START_IMPLEMENTATION.md`)
- Validation: Root directory (`VALIDATION_*.md`)
- Templates: `artifacts/experiments/EXPERIMENT_TEMPLATE.md`

**Code**:
- Source: `src/` (models, datasets, training, evaluation)
- Scripts: `scripts/` (training, evaluation, validation)

**Data**:
- Raw data: `data/` (CUB-200, Xeno-Canto)
- Features: `artifacts/audio_mfcc_cache/`
- Splits: `artifacts/splits/`

**Results**:
- Models: `artifacts/models/`
- Checkpoints: `artifacts/checkpoints/`
- Results: `artifacts/results/`
- Experiments: `artifacts/experiments/`
- Validation: `artifacts/validation/`

### Key Commands

**Validation**:
```bash
python scripts/run_all_validations.py
```

**Training (current)**:
```bash
python scripts/03_train_audio.py --model AudioCNN
```

**Training (Phase 1 - after fixes)**:
```bash
python scripts/03_train_audio.py --model AudioCNN --use-class-weights --epochs 50
```

**Evaluation**:
```bash
python scripts/05_evaluate.py
```

---

## 🎯 Next Actions

### Immediate (Today - 30 minutes)
1. [ ] Review this index document
2. [ ] Read [QUICK_START_IMPLEMENTATION.md](QUICK_START_IMPLEMENTATION.md)
3. [ ] Review [VALIDATION_SUMMARY_EXECUTIVE.md](VALIDATION_SUMMARY_EXECUTIVE.md)
4. [ ] Prepare development environment

### Short-term (Today - 2-3 hours)
1. [ ] Implement class-weighted loss (Fix 1)
2. [ ] Implement feature normalization (Fix 2)
3. [ ] Start retraining baseline models

### Medium-term (Tomorrow)
1. [ ] Complete baseline retraining
2. [ ] Evaluate Phase 1 results
3. [ ] Phase 1 checkpoint: Go/No-Go decision
4. [ ] Document baseline v2 results

### Long-term (Next 4 weeks)
1. [ ] Execute Phases 2-4 per implementation plan
2. [ ] Track all experiments systematically
3. [ ] Document learnings and insights
4. [ ] Achieve 75-85% accuracy target

---

## 📖 Reading Order Recommendation

### For Quick Start (30 minutes)
1. This document (INDEX_PLANNING.md) ← You are here
2. [QUICK_START_IMPLEMENTATION.md](QUICK_START_IMPLEMENTATION.md)
3. Start implementing Fix 1

### For Comprehensive Understanding (2 hours)
1. [VALIDATION_SUMMARY_EXECUTIVE.md](VALIDATION_SUMMARY_EXECUTIVE.md)
2. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
3. [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md)
4. [VALIDATION_COMPLETE.md](VALIDATION_COMPLETE.md)

### For Reference (as needed)
- [VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md)
- [EXPERIMENT_TEMPLATE.md](artifacts/experiments/EXPERIMENT_TEMPLATE.md)
- Other supporting documents

---

## 🆘 Getting Help

### If Stuck on Implementation
1. Review relevant section in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
2. Check code examples provided
3. Review validation results for context
4. Check troubleshooting section in [QUICK_START_IMPLEMENTATION.md](QUICK_START_IMPLEMENTATION.md)

### If Results Don't Match Expectations
1. Review success criteria for the phase
2. Check experiment template for analysis guidance
3. Review validation artifacts for data issues
4. Consult risk management section in implementation plan

### If Validation Fails
1. Review [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md) for fix examples
2. Check that fixes were implemented correctly
3. Run individual validation scripts for debugging
4. Review [README_VALIDATION.md](README_VALIDATION.md) for interpretation

---

## 📅 Timeline Summary

**Today (Dec 4)**: Planning complete, ready to start  
**Dec 5-6**: Phase 1 (Critical Fixes)  
**Dec 7-11**: Phase 2 (Feature Engineering)  
**Dec 12-25**: Phase 3 (Architecture Optimization)  
**Dec 26 - Jan 5**: Phase 4 (Advanced Techniques)  
**Jan 5**: Final evaluation and documentation

---

## ✅ Documentation Status

- [x] Validation complete
- [x] Root causes identified
- [x] Implementation plan created
- [x] Quick start guide created
- [x] Experiment template created
- [x] Index/navigation created
- [ ] Ready to start implementation ← **YOU ARE HERE**

---

**Status**: 📋 Planning Complete - Ready for Implementation  
**Next Step**: Begin Phase 1, Task 1.1 (Class-Weighted Loss)  
**Document Version**: 1.0  
**Last Updated**: December 4, 2025
