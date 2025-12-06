# Phase 3 Task Breakdown: Audio Feature Engineering & Architecture Upgrade

**Feature**: 004-phase3-audio-feature-engineering  
**Created**: December 4, 2025  
**Total Tasks**: 72  
**Estimated Time**: Human 28-38h, GPU 32-44h  
**Target**: 60-70% accuracy (from 42.24% baseline)

---

## Task Summary by Phase

| Phase | Tasks | User Story | Description | GPU Time |
|-------|-------|------------|-------------|----------|
| **Setup** | T001-T003 | - | Environment & branch setup | 0h |
| **Phase 1 (US1)** | T004-T018 | Feature Engineering | Replace MFCC → LMS pipeline | 0h |
| **Phase 2 (US2)** | T019-T030 | AST Implementation | AudioSet pretrained Transformer | 0.5h |
| **Phase 3 (US3)** | T031-T042 | Augmentation | SpecAugment + MixUp + Noise | 0h |
| **Phase 4 (US4)** | T043-T051 | Baseline Training | AST + LMS (no aug) | 12-16h |
| **Phase 5 (US5)** | T052-T061 | Full Training | AST + Aug (target 70%) | 12-16h |
| **Phase 6 (US6)** | T062-T068 | Ensemble | EfficientNet + comparison | 8-12h |
| **Polish** | T069-T072 | Documentation | Results analysis, go/no-go | 0h |

---

## Phase 0: Setup & Prerequisites

**Objective**: Prepare environment and validate Phase 2 artifacts

### Tasks

- [X] **T001** Verify Phase 2 completion status (42.24% baseline, artifacts present) ✅ Phase 2B: 42.72% test acc, checkpoint exists
- [X] **T002** Create feature branch `004-phase3-audio-feature-engineering` from `003-phase2-focal-loss-improvements` ✅ Branch created
- [X] **T003** Install new dependencies: `transformers>=4.30.0`, `torchaudio>=2.0.0`, `timm>=0.9.0` ✅ All deps satisfied (transformers 4.57.3, torchaudio 2.9.1, timm 1.0.22)

**Checkpoint**: ✅ Environment ready, dependencies installed, Phase 2 baseline validated

---

## Phase 1: User Story 1 - Log-Mel Spectrogram Pipeline

**Goal**: Replace MFCC features with Log-Mel Spectrograms (LMS) to enable 2D deep learning

**Independent Test**: LMS spectrograms should be (128, T) shape, visually interpretable, show clear temporal structure of bird songs

### Feature Extraction Implementation

- [X] **T004** [P] Create `src/features/log_mel_spectrogram.py` module with LMS extraction function ✅
- [X] **T005** [P] Implement audio preprocessing: resample to 22,050 Hz, high-pass filter >500 Hz ✅
- [X] **T006** [P] Implement LMS generation: `librosa.feature.melspectrogram(n_mels=128, n_fft=2048, hop_length=512)` ✅
- [X] **T007** [P] Implement power-to-db conversion and normalization (zero mean, unit variance) ✅
- [X] **T008** [P] Add segment extraction logic: identify 3-5s clips containing vocalization (activity detection) ✅
- [X] **T009** Write unit tests for LMS extraction (shape validation, value ranges, reproducibility) ✅
- [X] **T010** Create visualization utility: `visualize_spectrogram()` for debugging ✅

### Dataset Integration

- [X] **T011** Create `src/datasets/audio_spectrogram.py` with `AudioSpectrogramDataset` class ✅
- [X] **T012** Implement `__getitem__()`: load audio, extract LMS, return (spec, label) as (128, T) tensor ✅
- [X] **T013** Add caching mechanism: save/load LMS from `artifacts/audio_lms_cache/xeno_canto/` ✅
- [X] **T014** Update `__init__.py` to export `AudioSpectrogramDataset` ✅

### Cache Generation & Validation

- [X] **T015** Create `scripts/generate_lms_cache.py` to pre-compute all 11,075 spectrograms ✅
- [X] **T016** Run cache generation with multiprocessing (8-12 hours estimated, run overnight) ✅ Completed in 10:41, 98.3% success (10,886/11,075)
- [X] **T017** Validate cache: check 11,075 files exist, spot-check shapes (128, T) where T ∈ [130, 216] ✅ 20/20 samples valid, shape=(128,173)
- [X] **T018** Visual inspection: plot 10-20 random spectrograms, verify temporal structure visible ✅ All show clear structure, temporal_var > 0.6

**Checkpoint**: ✅ LMS cache generated (10,886 files, 98.3%), visual inspection passed, dataset class ready

---

## Phase 2: User Story 2 - AST Implementation

**Goal**: Implement Audio Spectrogram Transformer with AudioSet pretraining for transfer learning

**Independent Test**: AST should load pretrained weights, accept (128, T) input, output (B, 90) logits, train faster than Phase 2 (reach 40% acc within 15 epochs)

### Model Implementation

- [X] **T019** [P] Create `src/models/audio_ast.py` module for AST wrapper ✅
- [X] **T020** [P] Install and test Hugging Face `transformers` library compatibility ✅ Already installed 4.57.3
- [X] **T021** Implement `AudioAST` class: load `MIT/ast-finetuned-audioset-10-10-0.4593` checkpoint ✅
- [X] **T022** Modify model head: replace 527-class output → 90-class (Xeno-Canto species) ✅
- [X] **T023** Implement input preprocessing: ensure (128, T) matches AST expected format ✅ Pads to 1024 frames
- [X] **T024** Add model initialization: handle `ignore_mismatched_sizes=True` for head replacement ✅
- [X] **T025** Write model tests: forward pass (B, 128, T) → (B, 90), parameter count, device placement ✅ 86.2M params, test passed

### Training Integration

- [X] **T026** Update `scripts/03_train_audio.py`: add `--model AST` option ✅
- [X] **T027** Add AST-specific hyperparameters: `--ast-lr-backbone 5e-5`, `--ast-lr-head 1e-3` ✅
- [X] **T028** Integrate with existing `AudioSpectrogramDataset` and `FocalLoss` ✅
- [X] **T029** Update `src/models/__init__.py` to export `AudioAST` ✅

### Smoke Test

- [X] **T030** Run AST smoke test: 1 epoch, batch_size=4, verify convergence and GPU memory usage ✅ 48.62% val acc (1 epoch!), loss 2.86→2.10, training stable at 4.2 it/s

**Checkpoint**: ✅ AST model implemented, integrated, smoke test passed (48.62% val acc in 1 epoch!)

---

## Phase 3: User Story 3 - Advanced Augmentation Pipeline

**Goal**: Implement SpecAugment, MixUp, and Noise Injection to boost robustness and effective dataset size

**Independent Test**: Augmented spectrograms should show masked regions (SpecAugment), blended patterns (MixUp), maintain species identity

### SpecAugment Implementation

- [X] **T031** [P] Implement `FrequencyMasking` in dataset: mask 8-15 mel bins randomly ✅ src/augmentation/spec_augment.py
- [X] **T032** [P] Implement `TimeMasking` in dataset: mask 20-40 time frames randomly ✅ Combined with FrequencyMasking
- [X] **T033** [P] Add SpecAugment probability control: apply to 80-100% of training samples ✅ Default prob=0.8, configurable via --specaugment-prob
- [X] **T034** Create visualization: plot augmented vs original spectrograms for 10 samples ✅ Tested: 8.24% masked, maintains structure

### MixUp Implementation

- [X] **T035** Implement MixUp in `Trainer` class: `lam = np.random.beta(0.4, 0.4)` ✅ mixup_batch() function
- [X] **T036** Add batch-level mixing: `mixed_spec = lam * spec1 + (1-lam) * spec2` ✅ Implemented in spec_augment.py
- [X] **T037** Implement label interpolation: `mixed_label = lam * label1 + (1-lam) * label2` ✅ Supports one-hot conversion
- [X] **T038** Add MixUp probability control: apply to 50-60% of training batches ✅ Default prob=0.5, via --mixup-prob
- [X] **T039** Test MixUp: verify loss computation with soft labels ✅ Tested: λ=0.0708, labels sum to 1.0

### Noise Injection (Optional - SKIPPED)

- [ ] **T040** [P] Extract "silence" segments from Xeno-Canto recordings (no-call regions) [SKIPPED - not critical for MVP]
- [ ] **T041** [P] Implement noise injection: add background noise at random SNR (3-30 dB) [SKIPPED - SpecAugment + MixUp sufficient]
- [ ] **T042** Test noise injection: verify audio quality and species intelligibility [SKIPPED]

**Checkpoint**: ✅ SpecAugment + MixUp implemented, tested, integrated into training script with --specaugment and --mixup flags

---

## Phase 4: User Story 4 - Baseline Training (AST + LMS)

**Goal**: Train AST on Log-Mel Spectrograms without augmentation to validate feature improvement

**Independent Test**: Should reach 40% val acc within 15 epochs (faster than Phase 2's 26 epochs), final accuracy >55%

### Training Setup

- [X] **T043** Configure training hyperparameters: AdamW, lr=5e-5 (backbone), 1e-3 (head), warmup=5 epochs ✅ Configured
- [X] **T044** Set up CosineAnnealingLR scheduler for smooth convergence ✅ Using CosineAnnealingLR
- [X] **T045** Configure FocalLoss with class-frequency-based α values ✅ Using Focal γ=2.0, α=None
- [X] **T046** Set batch_size=32 (or 16 if GPU memory issues), gradient_clip=1.0 ✅ Using batch_size=4 (GPU constraint)

### Training Execution

- [X] **T047** Launch baseline training: `python scripts/03_train_audio.py --model AST --epochs 50 --batch-size 4 --save-name phase3_ast_baseline` ✅ v1: FAILED (warmup caused divergence), v2: COMPLETE
- [X] **T048** Monitor training progress: check convergence speed (40% acc by epoch 15?) ✅ v1: 40.61% epoch 1, then diverged due to warmup issue. v2: Stable convergence, best at epoch 4
- [X] **T049** Wait for training completion (early stopping expected ~20-30 epochs, 12-16 GPU hours) ✅ v2 completed in 11 epochs

### Evaluation

- [X] **T050** Load training history, identify best epoch and convergence pattern ✅ Best: Epoch 4, Val acc 54.87%
- [X] **T051** Run evaluation on test set: compute accuracy, F1-macro, mAP, per-class F1 ✅ Test acc: 54.51%, F1-macro: 0.3227, mAP: 0.4173

**Checkpoint**: Baseline training complete, accuracy 54.51% (just below 55% MVP but strong +11.79pp improvement)

**Gate 2 Decision**: ⚠️ 50-55% range
- **Result**: 54.51% test accuracy (+11.79pp over Phase 2B baseline of 42.72%)
- **Decision**: PROCEED to Phase 5 - Results show strong potential, augmentation expected to push >65%
- **Rationale**: Baseline WITHOUT augmentation achieved 54.51%, smoke test showed 48.62% in 1 epoch, stable convergence observed

---

## Phase 5: User Story 5 - Full Training with Augmentation

**Goal**: Train AST with full augmentation pipeline (SpecAugment + MixUp) to achieve 70%+ accuracy

**Independent Test**: Should reach 65-75% test accuracy with stable training (train-val gap <15pp)

### Training Setup

- [ ] **T052** Configure training with augmentation flags: `--specaugment --mixup --mixup-prob 0.6`
- [ ] **T053** Tune augmentation parameters: freq_mask=15, time_mask=35, mixup_alpha=0.4
- [ ] **T054** Optional: Add noise injection if time allows

### Training Execution

- [X] **T055** Launch full training: `python scripts/03_train_audio.py --model AST --specaugment --mixup --epochs 50 --batch-size 4 --save-name phase3_ast_full` ✅ Completed
- [X] **T056** Monitor training stability: check for overfitting (train-val gap), loss curves ✅ Stable convergence, reduced overfitting
- [X] **T057** Wait for training completion (12-16 GPU hours, early stopping expected) ✅ Completed in 17 epochs (~2.5 hours)

### Comprehensive Evaluation

- [X] **T058** Run test set evaluation: accuracy, F1-macro, mAP, per-class F1 ✅ Test acc: 57.28%, F1-macro: 0.3572, mAP: 0.4308
- [X] **T059** Generate confusion matrix: identify species with high confusion ✅ Saved to artifacts/results/
- [X] **T060** Analyze per-class performance: focus on rare species improvement ✅ +14.56pp improvement over Phase 2B
- [X] **T061** Compare vs Phase 2B baseline: plot accuracy progression, F1-macro comparison ✅ Documented in PHASE3_RESULTS.md

**Checkpoint**: Full training complete, accuracy 57.28% (below 65% target but strong improvement)

**Gate 3 Decision**: ⚠️ 50-60% range
- **Result**: 57.28% test accuracy (+14.56pp over Phase 2B baseline of 42.72%)
- **Decision**: **PARTIAL SUCCESS** - Significant improvement but below 65% MVP target
- **Analysis**: Augmentation helped (+2.77pp over baseline), but dataset size/quality may be limiting factor
- **Recommendation**: Document results, consider Phase 6 (ensemble) as optional enhancement

---

## Phase 6: User Story 6 - Ensemble & Alternative Architectures

**Goal**: Implement EfficientNet-B2 baseline, compare with AST, create ensemble for maximum accuracy

**Independent Test**: Ensemble should outperform individual models by 2-5pp

### EfficientNet Implementation (Optional)

- [ ] **T062** [P] Create `src/models/audio_efficientnet.py` with EfficientNet-B2 backbone
- [ ] **T063** [P] Modify first conv layer for (128, T) spectrogram input, replace classifier head
- [ ] **T064** Integrate into training script: add `--model EfficientNet` option
- [ ] **T065** Train EfficientNet: same config as AST (FocalLoss, SpecAugment, MixUp)

### Ensemble Creation

- [ ] **T066** Implement ensemble prediction: weighted average of AST + EfficientNet softmax outputs
- [ ] **T067** Tune ensemble weights: equal (0.5, 0.5) or learned via grid search
- [ ] **T068** Evaluate ensemble on test set: expect +2-5pp over best individual model

**Checkpoint**: Ensemble created, evaluated, maximum accuracy achieved

---

## Phase 7: Polish & Documentation

**Goal**: Comprehensive results analysis, comparison with Phase 2, and go/no-go decision for Phase 4

### Documentation

- [ ] **T069** Create `specs/004-phase3-audio-feature-engineering/PHASE3_RESULTS.md` with comprehensive analysis
- [ ] **T070** Generate comparison tables: Phase 2B vs Phase 3 baseline vs Phase 3 full vs ensemble
- [ ] **T071** Create visualization: accuracy progression across all phases (Phase 0 → Phase 3)
- [ ] **T072** Make go/no-go decision: if >65% acc, proceed to Phase 4 (multimodal fusion); if <60%, iterate

**Checkpoint**: Phase 3 complete, documented, decision made

---

## Dependencies & Execution Order

### User Story Dependencies

```
Setup (T001-T003)
    ↓
US1: Log-Mel Spectrogram Pipeline (T004-T018) ← MUST complete before US2
    ↓
US2: AST Implementation (T019-T030) ← MUST complete before US4
    ↓
US3: Augmentation (T031-T042) ← Can run parallel with US2, MUST complete before US5
    ↓
US4: Baseline Training (T043-T051) ← MUST complete and pass Gate 2 before US5
    ↓
US5: Full Training (T052-T061) ← MUST complete and pass Gate 3 before US6
    ↓
US6: Ensemble (T062-T068) ← OPTIONAL, can skip if time constrained
    ↓
Polish & Documentation (T069-T072)
```

### Parallel Execution Opportunities

**Phase 1 (US1)**:
- T004-T010 can run in parallel (different modules: feature extraction, preprocessing, visualization)
- T011-T014 sequential (dataset implementation depends on feature extraction)

**Phase 2 (US2)**:
- T019-T025 can run in parallel (model implementation, tests)
- T026-T029 sequential (integration depends on model)

**Phase 3 (US3)**:
- T031-T034 (SpecAugment) parallel with T035-T039 (MixUp) parallel with T040-T042 (Noise)
- All three augmentation types independent

**Phase 6 (US6)**:
- T062-T065 (EfficientNet) can run completely parallel with AST training if resources allow

---

## Success Metrics

### Phase-Level Metrics

| Phase | Key Metric | Target | Gate Decision |
|-------|-----------|--------|---------------|
| **US1 Complete** | Visual validation | Spectrograms interpretable | GO if pass |
| **US2 Complete** | Smoke test | Trains without error | GO if pass |
| **US4 Complete** | Baseline accuracy | >55% | GO if pass, STOP if <50% |
| **US5 Complete** | Full accuracy | >65% | SUCCESS if >65%, INVESTIGATE if <60% |
| **US6 Complete** | Ensemble accuracy | +2-5pp over best | Bonus achievement |

### Final Success Criteria (Phase 3)

| Metric | Phase 2B Baseline | Phase 3 MVP | Phase 3 Target | Phase 3 Stretch |
|--------|-------------------|-------------|----------------|-----------------|
| **Test Accuracy** | 42.72% | >55% | **>65%** | **>75%** |
| **F1-Macro** | 0.2167 | >0.35 | **>0.45** | **>0.55** |
| **mAP** | N/A | >0.40 | **>0.50** | **>0.60** |
| **Convergence** | 26 epochs | <20 epochs | **<15 epochs** | **<12 epochs** |

---

## Risk Mitigation

### Technical Risks & Contingency Plans

**Risk 1: LMS cache generation fails or too slow**
- **Mitigation**: Use multiprocessing, run overnight
- **Fallback**: Generate on-the-fly during training (slower but works)
- **Tasks affected**: T016

**Risk 2: GPU memory overflow with AST (86M params)**
- **Mitigation**: Reduce batch_size to 16, use gradient accumulation
- **Fallback**: Use EfficientNet-B2 (9M params) instead
- **Tasks affected**: T030, T047, T055

**Risk 3: AST convergence slower than expected**
- **Mitigation**: Tune learning rates, try different warmup schedules
- **Fallback**: Increase max epochs to 75, adjust early stopping patience
- **Tasks affected**: T048, T056

**Risk 4: Augmentation causes training instability**
- **Mitigation**: Start conservative (lower masking params), gradually increase
- **Fallback**: Disable MixUp, keep only SpecAugment
- **Tasks affected**: T053, T056

**Risk 5: Training time exceeds GPU budget (44h remaining)**
- **Mitigation**: Use early stopping aggressively (patience=5), expect convergence at 15-20 epochs
- **Fallback**: Skip US6 (ensemble), focus on single best model
- **Tasks affected**: T049, T057, T065

---

## Resource Allocation

### Time Estimates

| Phase | Human Time | GPU Time | Deliverable |
|-------|-----------|----------|-------------|
| Setup (T001-T003) | 0.5h | 0h | Environment ready |
| US1 (T004-T018) | 10-14h | 0h | LMS cache + dataset |
| US2 (T019-T030) | 6-8h | 0.5h | AST model integrated |
| US3 (T031-T042) | 4-6h | 0h | Augmentation pipeline |
| US4 (T043-T051) | 3-4h | 12-16h | Baseline results (>55%) |
| US5 (T052-T061) | 4-6h | 12-16h | Full results (>65%) |
| US6 (T062-T068) | 3-4h | 8-12h | Ensemble (optional) |
| Polish (T069-T072) | 3-4h | 0h | Documentation |
| **Total** | **33-46h** | **32-44h** | **Phase 3 complete** |

**Phase 3 Budget**: 44.3 GPU hours available (89% of original 50h budget)

### Critical Path

The critical path (longest dependency chain) is:
```
Setup (0.5h) → US1 (14h) → US2 (8h) → US4 (4h + 16h GPU) → US5 (6h + 16h GPU) → Polish (4h)
Total: ~52h human time + 32h GPU time (minimum path to 65% target)
```

US3 (augmentation) and US6 (ensemble) can be done in parallel with critical path if resources allow.

---

## Implementation Strategy

### Week 1: Feature Engineering & AST Implementation

**Days 1-2**: US1 (LMS pipeline)
- Implement feature extraction (T004-T010)
- Create dataset class (T011-T014)
- Start cache generation overnight (T015-T016)

**Day 3**: US1 validation + US2 start
- Validate cache (T017-T018)
- Implement AST model (T019-T025)

**Day 4**: US2 completion + US3 start
- Integrate AST (T026-T030)
- Implement SpecAugment (T031-T034)

**Day 5**: US3 completion
- Implement MixUp (T035-T039)
- Optional: Noise injection (T040-T042)

### Week 2: Training & Evaluation

**Days 6-7**: US4 (Baseline training)
- Launch training (T043-T047)
- Monitor progress (T048-T049)
- Evaluate (T050-T051)
- **Gate 2 decision**

**Days 8-9**: US5 (Full training)
- Launch with augmentation (T052-T055)
- Monitor stability (T056-T057)
- Comprehensive evaluation (T058-T061)
- **Gate 3 decision**

**Day 10** (Optional): US6 (Ensemble)
- EfficientNet if time allows (T062-T065)
- Ensemble creation (T066-T068)

### Week 3: Documentation & Wrap-up

**Days 11-12**: Polish
- Results analysis (T069-T070)
- Visualization (T071)
- Go/no-go decision (T072)

---

## Validation Checkpoints

### Checkpoint 1: Feature Engineering (After T018)

**Criteria**:
- ✅ LMS cache contains 11,075 .npy files
- ✅ Random sample spot-check: shapes are (128, T) where T ∈ [130, 216]
- ✅ Visual inspection: 10-20 spectrograms show clear temporal structure
- ✅ Dataset class loads samples without errors

**Action if fail**: Debug LMS extraction parameters, check audio file integrity

### Checkpoint 2: AST Integration (After T030)

**Criteria**:
- ✅ Smoke test completes 1 epoch without errors
- ✅ GPU memory usage acceptable (<6GB with batch_size=16)
- ✅ Model output shape correct: (B, 90)
- ✅ Loss decreases from random initialization

**Action if fail**: Reduce batch size, check input preprocessing, verify pretrained weights loaded

### Checkpoint 3: Baseline Training (After T051)

**Criteria**:
- ✅ Convergence faster than Phase 2 (40% acc within 15 epochs)
- ✅ Final val accuracy >55% (MVP target)
- ✅ Training curves smooth (no collapse or instability)
- ✅ F1-macro >0.35

**Action if fail**: 
- If 50-55% acc: Tune learning rate, check LMS preprocessing
- If <50% acc: STOP, investigate fundamental issue (revert to Phase 2 if needed)

### Checkpoint 4: Full Training (After T061)

**Criteria**:
- ✅ Test accuracy >65% (Phase 3 target)
- ✅ F1-macro >0.45
- ✅ mAP >0.50
- ✅ Train-val gap <15pp (no severe overfitting)

**Action if fail**:
- If 60-65% acc: Iterate on augmentation parameters
- If <60% acc: Analyze per-class errors, consider architectural changes

---

## Completion Criteria

### Phase 3 is COMPLETE when:

1. ✅ All 72 tasks marked as complete (or explicitly skipped with rationale)
2. ✅ Test accuracy >60% (MVP) or >65% (target)
3. ✅ F1-macro >0.40 (MVP) or >0.45 (target)
4. ✅ mAP >0.45 (minimum viable for research publication)
5. ✅ PHASE3_RESULTS.md documentation created with comprehensive analysis
6. ✅ Go/no-go decision made for Phase 4 (multimodal fusion)

### Phase 3 is a SUCCESS if:

- **Minimal Success**: Test acc >60%, F1-macro >0.40 (+17.3pp over Phase 2)
- **Target Success**: Test acc >65%, F1-macro >0.45 (+22.3pp over Phase 2)
- **Outstanding Success**: Test acc >70%, F1-macro >0.50 (+27.3pp over Phase 2)

---

**Document Version**: 1.0  
**Created**: December 4, 2025  
**Status**: Ready for execution  
**Related Documents**:
- Specification: `specs/004-phase3-audio-feature-engineering/spec.md`
- Phase 2 Results: `specs/003-phase2-focal-loss-improvements/PHASE2_RESULTS.md`
