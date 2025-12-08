# Where to Find Things

**Last Updated**: 2025-12-08

Quick reference for navigating the reorganized repository.

---

## 📋 Current Work

### Phase 1 Implementation (In Progress)

- **What to build**: `specs/002-phase1-critical-fixes/spec.md`
- **How to build it**: `specs/002-phase1-critical-fixes/plan.md`
- **Save results to**: `specs/002-phase1-critical-fixes/artifacts/`

---

## 📚 Documentation

### Essential Docs (Root Directory)

- `README.md` - Project overview
- `QUICKSTART.md` - Getting started guide
- `STATUS.md` - Current project status
- `WHERE_TO_FIND_THINGS.md` - This file

### Specifications & Plans

- `specs/README.md` - **Start here** for all specs navigation
- `specs/001-validation-phase/` - Completed validation phase
- `specs/002-phase1-critical-fixes/` - Current implementation phase
- `specs/REORGANIZATION_SUMMARY.md` - Why things changed

### Old Documents (Archived)

- `specs/archive/` - All old CAPS-LOCK .md files moved here
  - `old-validation-docs/` - Pre-SpecKit validation docs
  - `old-planning-docs/` - Pre-SpecKit planning docs
  - `old-implementation-docs/` - Various summaries
  - `README.md` - Guide to archived files

### Constitution & Templates

- `.specify/memory/constitution.md` - Project principles and rules
- `.specify/templates/` - Templates for specs, plans, tasks

---

## 📊 Data & Artifacts

### Current Experiments

- `artifacts/experiments/` - Experimental results
- `artifacts/checkpoints/` - Model checkpoints
- `artifacts/metrics/` - Training metrics
- `artifacts/results/` - Final results

### Datasets

- `data/` - Raw datasets (not in git)
- `artifacts/splits/` - Train/val/test splits
- `artifacts/audio_mfcc_cache/` - Cached MFCC features

### Validation Outputs

- `artifacts/validation/` - Validation phase outputs
  - `recommended_class_weights.json` - Class weights for training
  - `class_distribution_*.png` - Class distribution plots
  - `feature_*.png` - Feature analysis plots

---

## 💻 Code

### Source Code

- `src/` - Main source code
  - `data/` - Dataset loaders
  - `datasets/` - PyTorch datasets
  - `features/` - Feature extraction
  - `models/` - Model definitions
  - `training/` - Training logic
  - `evaluation/` - Evaluation metrics
  - `utils/` - Utilities

### Scripts

- `scripts/` - Executable scripts
  - `verify_datasets.py` - Verify dataset paths
  - `validate_data.py` - Validate data integrity
  - `01_run_indexing.py` - Index datasets
  - `02_splits_and_features.py` - Create splits and extract features
  - `03_train_audio.py` - Train audio models
  - `04_train_image.py` - Train image models
  - `05_evaluate.py` - Evaluate models

### Tests

- `tests/` - Unit and integration tests
  - `test_implementation.py` - Verify implementation
  - `test_normalization.py` - Verify feature normalization
  - `test_phase1_implementation.py` - Test Phase 1 changes

### Notebooks

- `notebooks/` - Jupyter notebooks
  - `00_env_setup.ipynb` - Environment setup
  - `01_intersection.ipynb` - Dataset intersection
  - `02_audio_features.ipynb` - Audio feature exploration
  - `03_image_models.ipynb` - Image model experiments
  - `04_training_compare.ipynb` - Training comparison
  - `05_evaluate.ipynb` - Evaluation notebook

---

## 🔍 Looking for Something?

### "Where are the validation results?"

- **New**: `specs/001-validation-phase/spec.md` (consolidated)
- **Old**: `specs/archive/old-validation-docs/` (reference only)

### "Where is the implementation plan?"

- **Current Phase 1**: `specs/002-phase1-critical-fixes/plan.md`
- **Old plans**: `specs/archive/old-implementation-docs/`

### "Where do I save my experiment results?"

- `specs/002-phase1-critical-fixes/artifacts/`
- Or `artifacts/experiments/` for general experiments

### "Where are the old CAPS-LOCK .md files?"

- All moved to `specs/archive/`
- Organized by category (validation, planning, implementation)
- See `specs/archive/README.md` for guide

### "How do I create a new spec for Phase 2?"

1. Read `specs/README.md` for workflow
2. Create `003-phase2-feature-engineering/` directory
3. Copy template from `.specify/templates/spec-template.md`
4. Follow SpecKit workflow: specify → clarify → plan → tasks

---

## 🚀 Quick Commands

```bash
# View current work spec
cat specs/002-phase1-critical-fixes/spec.md

# View implementation plan
cat specs/002-phase1-critical-fixes/plan.md

# Navigate all specs
cat specs/README.md

# Check what's archived
ls specs/archive/
cat specs/archive/README.md

# View project constitution
cat .specify/memory/constitution.md
```
