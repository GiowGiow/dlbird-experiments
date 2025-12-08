# Feature Specification: Initial Model Validation

**Spec ID**: 001-validation-phase  
**Status**: ✅ Complete  
**Created**: 2025-12-03  
**Last Updated**: 2025-12-04  
**Phase**: Validation & Root Cause Analysis

---

## Executive Summary

Comprehensive validation of baseline audio and image models for bird species classification to identify performance gaps and root causes of poor audio model performance.

**Problem**: Audio models (AudioCNN, AudioViT) significantly underperform image models (ImageResNet18, ImageViT) with a 2.3x accuracy gap (39.5% vs 92.6%) and severely degraded F1-macro score (0.109 vs 0.925).

**Goal**: Systematically validate models, datasets, and features to identify root causes and provide data-driven recommendations for improvement.

---

## User Stories

### US1: Dataset Validation
**As a** ML researcher  
**I want** comprehensive validation of audio (Xeno-Canto) and image (CUB-200) datasets  
**So that** I can confirm data quality and identify distribution issues

**Acceptance Criteria**:
- ✅ Species distribution analysis with visual plots
- ✅ Class imbalance quantification (ratios, Gini coefficient)
- ✅ Sample count statistics per species
- ✅ Missing data identification
- ✅ Dataset integrity validation

### US2: Feature Analysis
**As a** ML researcher  
**I want** statistical analysis of extracted MFCC features  
**So that** I can identify feature quality issues

**Acceptance Criteria**:
- ✅ Per-channel feature statistics (mean, std, min, max)
- ✅ Feature distribution visualization
- ✅ Normalization status check
- ✅ Outlier detection
- ✅ Feature correlation analysis

### US3: Model Performance Validation
**As a** ML researcher  
**I want** detailed performance metrics for all models  
**So that** I can quantify the performance gap and identify patterns

**Acceptance Criteria**:
- ✅ Accuracy and F1-macro scores per model
- ✅ Per-class precision, recall, F1 scores
- ✅ Confusion matrices with visualization
- ✅ Performance comparison across modalities
- ✅ Identification of consistently misclassified species

### US4: Root Cause Analysis
**As a** ML researcher  
**I want** systematic analysis to identify root causes of poor audio performance  
**So that** I can create a data-driven improvement plan

**Acceptance Criteria**:
- ✅ Class imbalance impact quantification
- ✅ Feature quality assessment
- ✅ Architecture suitability evaluation
- ✅ Prioritized list of root causes with evidence
- ✅ Recommended fixes with expected impact

### US5: Validation Artifacts & Documentation
**As a** ML researcher  
**I want** reproducible validation scripts and comprehensive documentation  
**So that** validation can be re-run and results can be trusted

**Acceptance Criteria**:
- ✅ Automated validation scripts (`scripts/validate_*.py`)
- ✅ Generated plots and analysis files in `artifacts/validation/`
- ✅ Executive summary document
- ✅ Detailed validation results document
- ✅ Quick-start validation guide

---

## Functional Requirements

### FR1: Dataset Validation Script
- **ID**: FR1.1  
  **Requirement**: Analyze species distribution across train/val/test splits  
  **Priority**: MUST HAVE

- **ID**: FR1.2  
  **Requirement**: Calculate class imbalance metrics (max/min ratio, Gini)  
  **Priority**: MUST HAVE

- **ID**: FR1.3  
  **Requirement**: Generate distribution plots (bar charts, histograms)  
  **Priority**: MUST HAVE

- **ID**: FR1.4  
  **Requirement**: Validate data integrity (missing files, corrupted samples)  
  **Priority**: SHOULD HAVE

### FR2: Feature Validation Script
- **ID**: FR2.1  
  **Requirement**: Compute per-channel statistics for MFCC features  
  **Priority**: MUST HAVE

- **ID**: FR2.2  
  **Requirement**: Check normalization status (mean≈0, std≈1)  
  **Priority**: MUST HAVE

- **ID**: FR2.3  
  **Requirement**: Visualize feature distributions (box plots, histograms)  
  **Priority**: MUST HAVE

- **ID**: FR2.4  
  **Requirement**: Detect outliers using statistical methods  
  **Priority**: SHOULD HAVE

### FR3: Class Balance Analysis Script
- **ID**: FR3.1  
  **Requirement**: Generate class weight recommendations (balanced, effective, sqrt)  
  **Priority**: MUST HAVE

- **ID**: FR3.2  
  **Requirement**: Calculate expected F1-macro improvement with weighting  
  **Priority**: SHOULD HAVE

- **ID**: FR3.3  
  **Requirement**: Visualize class imbalance severity  
  **Priority**: MUST HAVE

- **ID**: FR3.4  
  **Requirement**: Save weights to JSON for training script integration  
  **Priority**: MUST HAVE

### FR4: Orchestration & Reporting
- **ID**: FR4.1  
  **Requirement**: Master script to run all validation steps  
  **Priority**: MUST HAVE

- **ID**: FR4.2  
  **Requirement**: Generate executive summary with key findings  
  **Priority**: MUST HAVE

- **ID**: FR4.3  
  **Requirement**: Create detailed validation results report  
  **Priority**: MUST HAVE

- **ID**: FR4.4  
  **Requirement**: Produce index document linking all validation artifacts  
  **Priority**: SHOULD HAVE

---

## Key Findings (Post-Validation)

### Critical Issues Identified

1. **SEVERE Class Imbalance** (PRIMARY - 60% of problem)
   - Maximum ratio: 1216:1 (House Sparrow: 1216 samples, Hooded Merganser: 1 sample)
   - 23 species with ≤18 samples
   - Gini coefficient: 0.52 (high inequality)
   - **Impact**: Model ignores rare species, explains low F1-macro (0.109)

2. **Missing Feature Normalization** (SECONDARY - 20% of problem)
   - MFCC channel: mean=-8.80, std=62.53 (not normalized)
   - Delta channel: mean=0.02, std=1.69 (acceptable)
   - Delta² channel: already normalized
   - **Impact**: Slow convergence, gradient instability, suboptimal learning

3. **Suboptimal Feature Representation** (TERTIARY - 20% of problem)
   - MFCC compresses to 40 coefficients (information loss)
   - 3-second audio duration may be too short
   - No augmentation applied
   - **Impact**: Limited feature expressiveness

### Validated Metrics

**Audio Models**:
- AudioCNN: 39.5% accuracy, 0.109 F1-macro
- AudioViT: 39.0% accuracy, 0.108 F1-macro

**Image Models**:
- ImageResNet18: 92.6% accuracy, 0.925 F1-macro
- ImageViT: 91.8% accuracy, 0.918 F1-macro

**Performance Gap**: 2.3x in accuracy, 8.5x in F1-macro

---

## Out of Scope

- Implementation of fixes (covered in spec 002-phase1-critical-fixes)
- Model architecture changes
- New feature extraction pipelines
- Production deployment considerations
- Real-time inference optimization

---

## Success Criteria

- ✅ All validation scripts executable and produce artifacts
- ✅ Root causes identified with quantitative evidence
- ✅ Class imbalance severity documented (1216:1 ratio)
- ✅ Feature normalization issues detected
- ✅ Recommended class weights generated
- ✅ Validation documentation complete and accessible
- ✅ Artifacts saved in `artifacts/validation/`

---

## Dependencies

- Existing trained models in `artifacts/models/`
- Dataset splits in `artifacts/splits/`
- Cached MFCC features in `artifacts/audio_mfcc_cache/`
- Python environment with: pandas, numpy, matplotlib, seaborn, scikit-learn

---

## References

- VALIDATION_SUMMARY_EXECUTIVE.md
- VALIDATION_RESULTS.md
- INDEX_VALIDATION.md
- artifacts/validation/class_distribution_audio_dataset.png
- artifacts/validation/recommended_class_weights.json
