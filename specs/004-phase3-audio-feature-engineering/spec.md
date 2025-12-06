# Phase 3: Audio Feature Engineering & Architecture Upgrade

**Feature ID**: 004-phase3-audio-feature-engineering  
**Created**: December 4, 2025  
**Status**: Planning  
**Priority**: Critical  
**Based on**: Deep research findings + Phase 2 results (42.24% acc → Target: 70%+ acc)

---

## Executive Summary

Phase 2 achieved 42.24% accuracy using FocalLoss + AudioCNNv2 (4.2M params) on MFCC features. However, deep research reveals that **MFCCs are fundamentally incompatible with modern 2D deep learning** (CNNs, Transformers). The 42% ceiling represents an architectural mismatch, not a capacity limit.

**Root Cause**: MFCCs discard fine-grained spectro-temporal details required for 90-class bird species discrimination. CNNs/ViT require dense, image-like inputs with spatial correlation—MFCCs are sparse, compressed representations optimized for classical ML (HMMs).

**Solution**: Replace MFCCs with **Log-Mel Spectrograms (LMS)** + implement **AudioSet-pretrained models** (AST, PANNs) + **advanced augmentation** (SpecAugment, MixUp). This transforms audio → image processing, enabling transfer learning from ImageNet/AudioSet.

**Target**: 70%+ accuracy (state-of-the-art bioacoustics benchmark), F1-macro >0.50

---

## Problem Statement

### Current State (Phase 2 Results)

**Achievements**:
- ✅ 42.24% val acc, 42.72% test acc
- ✅ F1-macro 0.2167 (2x improvement over baseline)
- ✅ Stable training (FocalLoss + capacity solved imbalance)

**Fundamental Limitation**:
- ❌ **38-42% accuracy ceiling** regardless of architecture/capacity
- ❌ **MFCC feature bottleneck**: Compressed spectral envelope lacks spatial detail
- ❌ **No transfer learning**: Training from scratch on 7,751 samples
- ❌ **Architectural mismatch**: 2D CNNs/ViT can't exploit MFCC structure

### Research-Backed Analysis

**Key Finding #1: MFCCs Fail for Fine-Grained Classification**
> "MFCCs were traditionally implemented for classical machine learning and statistical models (HMMs)... They inherently discard fine-grained spectro-temporal details crucial for distinguishing between calls/songs of closely related bird species."

**Key Finding #2: LMS Enables Transfer Learning**
> "Log-Mel Spectrograms convert audio processing into image processing, which is precisely why pre-trained CNNs (ResNet) and ViTs excel. Using LMS allows the network to utilize inductive biases learned during pretraining on large visual datasets."

**Key Finding #3: AudioSet Pretraining is Mandatory**
> "The small intersecting dataset of 90 species from Xeno-Canto... makes training deep networks from random initialization highly prone to overfitting. The most successful methodologies rely on transfer learning from AudioSet (5,000+ hours, 527 sound classes)."

### Performance Gap Analysis

| Approach | Features | Pretrained | Val Acc | Gap to SOTA |
|----------|----------|-----------|---------|-------------|
| Phase 0 Baseline | MFCC | ❌ | 39.5% | -30.5pp |
| Phase 2B (Current) | MFCC | ❌ | 42.24% | -27.8pp |
| **BirdCLEF Winners** | **LMS** | ✅ AudioSet | **70-85%** | **Baseline** |

The 27.8 percentage point gap is attributed to:
1. **Feature representation**: MFCC vs LMS (~15-20pp)
2. **Transfer learning**: Scratch vs AudioSet (~10-12pp)
3. **Advanced augmentation**: None vs SpecAugment/MixUp (~5-8pp)

---

## Goals & Success Criteria

### Primary Objectives

1. **Feature Engineering Overhaul**
   - Replace MFCC pipeline with Log-Mel Spectrogram generation
   - Target: 128 mel bins, 3-5s segments, 22.05-32kHz sample rate
   - Validation: Spectrograms visually interpretable, show temporal structure

2. **Architecture Upgrade (Transfer Learning)**
   - Implement AudioSet-pretrained models: AST (Transformer), PANNs (CNN)
   - Target: Leverage 5,000+ hours of generic audio knowledge
   - Validation: Convergence faster than Phase 2 (< 15 epochs to 40% acc)

3. **Advanced Augmentation Pipeline**
   - SpecAugment (time/freq masking), MixUp (β=0.4-0.6), Noise injection
   - Target: 3-5x effective training data through augmentation
   - Validation: No overfitting (train-val gap < 15pp)

### Success Metrics

**Must-Have (Critical Success)**:
| Metric | Phase 2B Baseline | Phase 3 Target | Stretch Goal |
|--------|-------------------|----------------|--------------|
| **Val Accuracy** | 42.24% | **>60%** | **>70%** |
| **Test Accuracy** | 42.72% | **>60%** | **>70%** |
| **F1-Macro** | 0.2167 | **>0.40** | **>0.50** |
| **Training Stability** | Stable | Stable | Stable |
| **Convergence Speed** | 26 epochs | **<20 epochs** | **<15 epochs** |

**Should-Have (High Value)**:
- mAP (Mean Average Precision) >0.50 (better metric for imbalanced data)
- Per-class F1 >0.20 for rare species (currently unmeasured)
- Train-val gap <15pp (Phase 2B: 52pp overfitting)

**Nice-to-Have (Bonus)**:
- 75%+ accuracy (competitive with BirdCLEF top-10)
- F1-macro >0.60 (rare species well-handled)
- Ensemble accuracy >80% (combining AST + PANNs)

### Non-Goals (Out of Scope)

- ❌ Multimodal fusion (audio + image) - defer to Phase 4
- ❌ Self-supervised pretraining on unlabeled Xeno-Canto - Phase 5+
- ❌ Real-time inference optimization - production concern
- ❌ New dataset collection - work with existing 11,075 recordings

---

## Technical Context

### Dataset Characteristics

**Xeno-Canto Corpus (90 species intersection)**:
- Total: 11,075 recordings
- Train: 7,751 (70%)
- Val: 1,662 (15%)
- Test: 1,662 (15%)
- Imbalance: 1216:1 (max:min ratio)
- Quality: Field recordings (variable noise, duration, sample rate)

**Audio Characteristics**:
- Sample rates: 16-96 kHz (variable)
- Durations: 3-300 seconds (highly variable)
- Noise: Wind, traffic, other species, microphone artifacts
- Target: Bird vocalizations typically 2-12 kHz frequency band

### Current Pipeline (Phase 2 - To Be Replaced)

```python
# Current MFCC extraction (audio.py)
mfcc_static = librosa.feature.mfcc(
    y=audio, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512
)
mfcc_delta = librosa.feature.delta(mfcc_static)
mfcc_delta2 = librosa.feature.delta(mfcc_static, order=2)
mfcc_combined = np.stack([mfcc_static, mfcc_delta, mfcc_delta2])
# Shape: (3, 20, ~500 time frames) - Compressed, sparse
```

**Problems**:
- Only 20 MFCC coefficients (severe dimensionality reduction)
- Discards phase information and fine spectral details
- Output shape (3, 20, 500) too sparse for CNNs to learn spatial features
- No visual interpretability (can't debug/understand what model sees)

### Proposed Pipeline (Phase 3 - Research-Backed)

```python
# Log-Mel Spectrogram extraction
mel_spec = librosa.feature.melspectrogram(
    y=audio, 
    sr=22050,           # Standardize sample rate
    n_fft=2048,         # Frequency resolution
    hop_length=512,     # Temporal resolution
    n_mels=128,         # Vertical dimension (frequency bins)
    fmin=500,           # High-pass filter (remove low-freq noise)
    fmax=11025          # Nyquist frequency
)
log_mel = librosa.power_to_db(mel_spec, ref=np.max)
# Shape: (128, ~130 time frames for 3s) - Dense, image-like
```

**Advantages**:
- 128 mel bins (vs 20 MFCC) = 6.4x richer frequency resolution
- Dense 2D representation suitable for CNNs/Transformers
- Visually interpretable (can see song structure)
- Compatible with ImageNet/AudioSet pretrained models
- Standard in BirdCLEF competitions (proven effective)

### Research-Backed Parameter Calibration

**Critical Parameters** (from research):

| Parameter | Recommended Value | Rationale |
|-----------|------------------|-----------|
| `sample_rate` | 22,050 or 32,000 Hz | Captures bird vocalizations (most <16 kHz) while optimizing compute |
| `n_mels` | 128 | Standard for deep learning, balances resolution vs compute |
| `n_fft` | 1024 or 2048 | Determines frequency resolution (2048 better for detail) |
| `hop_length` | 256 or 512 | Controls temporal granularity (512 is standard) |
| `fmin` | 500-1000 Hz | High-pass filter to remove wind/traffic noise |
| `fmax` | 11,025 Hz (for 22.05kHz) | Nyquist frequency (sr/2) |
| `segment_duration` | 3-5 seconds | Sufficient context for song phrase analysis |

**Preprocessing Pipeline**:
1. **Standardize**: Resample all audio to 22,050 Hz
2. **Filter**: High-pass >500 Hz (remove low-freq environmental noise)
3. **Segment**: Extract 3-5s clips containing vocalization (activity detection)
4. **Transform**: Generate Log-Mel Spectrogram (128, T) where T ≈ 130-216 frames
5. **Normalize**: Per-clip normalization (zero mean, unit variance)

---

## Architecture & Model Selection

### Research Consensus: Two Architectures Required

**Recommendation**: Implement both AST (Transformer) and PANNs (CNN) to compare and ensemble.

### 1. Audio Spectrogram Transformer (AST) [Primary Focus]

**Why AST**:
> "AST is a purely attention-based model... It captures long-range global context and non-local dependencies, even in its lowest layers. This capacity is crucial for species identification where the overall pattern or syntax of the song—the relationship between segments separated by silence—is often more informative than features of an isolated call."

**Architecture**:
- **Input**: Log-Mel Spectrogram (128, T) treated as 2D image
- **Patch embedding**: Break spectrogram into 16×16 patches
- **Transformer encoder**: 12 layers, 768 hidden dim (ViT-Base equivalent)
- **Pretraining**: ImageNet → AudioSet (transfer learning path)
- **Output**: Global average pooling → FC(90 classes)

**Implementation**:
```python
# Pseudocode
from transformers import ASTForAudioClassification

model = ASTForAudioClassification.from_pretrained(
    'MIT/ast-finetuned-audioset-10-10-0.4593',  # AudioSet checkpoint
    num_labels=90,
    ignore_mismatched_sizes=True
)
# Fine-tune on Xeno-Canto LMS
```

**Expected Performance**: 65-75% accuracy (based on literature)

### 2. Pretrained Audio Neural Networks (PANNs) [CNN Baseline]

**Why PANNs**:
> "PANNs, developed from the AudioSet corpus, provide robust benchmark for CNN-based audio classification. The most effective architecture is Wavegram-Logmel-CNN... PANN-based models have proven highly reliable in large-scale bioacoustics competitions like BirdCLEF."

**Architecture Options**:
1. **Wavegram-Logmel-CNN**: Dual-input (LMS + waveform), fused features
2. **EfficientNet-B0/B2**: Proven in BirdCLEF, efficient, good performance
3. **ResNet-like PANNs**: CNN14, Wavegram-Logmel variants

**Implementation**:
```python
# PANNs library or torchvision EfficientNet
import torchaudio.models as models

# Option 1: Use PANNs directly
model = models.wav2vec2_model(...)  # Replace with PANNs

# Option 2: EfficientNet (proven in competitions)
from efficientnet_pytorch import EfficientNet
backbone = EfficientNet.from_pretrained('efficientnet-b2')
# Modify first conv for (128, T) input, replace classifier
```

**Expected Performance**: 60-70% accuracy (CNN baseline)

### Comparison Matrix

| Model | Type | Params | Pretraining | Strength | Expected Acc |
|-------|------|--------|-------------|----------|--------------|
| **AST** | Transformer | 86M | ImageNet→AudioSet | Long-range temporal context, global attention | **65-75%** |
| **PANNs (Wavegram)** | CNN (dual-input) | ~81M | AudioSet | Local spectral texture, waveform fusion | **60-70%** |
| **EfficientNet-B2** | CNN | ~9M | ImageNet→Custom | Efficient, proven in BirdCLEF, compact | **60-68%** |
| AudioCNNv2 (Phase 2) | CNN (scratch) | 4.2M | None | Baseline | 42% |

**Decision**: Prioritize AST (best for bird songs with temporal structure), use PANNs/EfficientNet as ensemble members.

---

## Data Augmentation Strategy

### Critical Insight from Research

> "Data augmentation is not merely a method to increase sample size but a mandatory technique to enforce model robustness and invariance to environmental noise, pitch shifts, and timing variations."

### 1. SpecAugment (Mandatory)

**What**: Randomly mask blocks on time/frequency axes of spectrogram

**Implementation**:
```python
# torchaugio.transforms.FrequencyMasking, TimeMasking
freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

# Apply during training
spec_augmented = time_mask(freq_mask(log_mel_spec))
```

**Parameters**:
- `freq_mask_param`: 8-15 bins (out of 128)
- `time_mask_param`: 20-40 frames (out of ~130)
- Probability: 0.8-1.0 (apply to most samples)

**Effect**: Forces network to learn from partial information, robust to frequency-specific noise and temporal dropouts.

### 2. MixUp (High Priority)

**What**: Linear interpolation of spectrograms and labels

**Implementation**:
```python
# During training batch
lam = np.random.beta(0.4, 0.4)  # Beta distribution
mixed_spec = lam * spec1 + (1 - lam) * spec2
mixed_label = lam * label1 + (1 - lam) * label2
```

**Parameters**:
- Beta distribution: α=0.4, β=0.4 (cited as optimal)
- Probability: 0.5-0.6 (apply to 50-60% of batches)

**Effect**: Smooths decision boundaries, powerful regularization, especially for rare classes.

### 3. Random Noise Injection (Important)

**What**: Add environmental noise at random SNR

**Implementation**:
```python
# Load noise clips (wind, rain, traffic) or Xeno-Canto "silence" segments
noise = load_random_noise()
snr_db = np.random.uniform(3, 30)  # 3-30 dB SNR range
noisy_audio = add_noise_at_snr(audio, noise, snr_db)
```

**Effect**: Prevents overfitting to clean recordings, trains robustness to field conditions.

### 4. Pitch Shifting & Time Stretching (Optional)

**What**: Augment audio in time/frequency domain

**Parameters**:
- Pitch shift: ±2 semitones (preserves species identity)
- Time stretch: 0.9-1.1x speed

**Effect**: Enforces invariance to recording quality and individual variation.

### Augmentation Priority

| Technique | Priority | Expected Impact | Implementation Effort |
|-----------|----------|-----------------|----------------------|
| **SpecAugment** | **CRITICAL** | **+5-8pp acc** | Low (torchaudio built-in) |
| **MixUp** | **HIGH** | **+3-5pp acc** | Medium (batch-level) |
| **Noise Injection** | **HIGH** | **+2-4pp acc** | Medium (requires noise corpus) |
| Pitch/Time Shift | Medium | +1-2pp acc | Medium (librosa/torchaudio) |

**Phase 3 MVP**: SpecAugment + MixUp (mandatory), Noise injection (if time allows)

---

## Loss Function & Training Strategy

### Focal Loss (Retain from Phase 2)

**Validated in Phase 2**: FocalLoss (γ=2.0) worked well, 2x F1-macro improvement.

**Recommendation**: Keep FocalLoss, but tune α based on class frequencies.

**Implementation**:
```python
# Compute α as inverse frequency
class_counts = np.bincount(train_labels)
alpha = 1.0 / (class_counts / class_counts.sum())
alpha = alpha / alpha.sum() * num_classes  # Normalize

focal_loss = FocalLoss(gamma=2.0, alpha=torch.FloatTensor(alpha))
```

**Advanced (Research-backed)**: Average BCE + Focal Loss for stability.
```python
loss = 0.5 * bce_loss(outputs, targets) + 0.5 * focal_loss(outputs, targets)
```

### Training Hyperparameters (AST)

Based on AudioSet fine-tuning literature:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | AdamW | Standard for Transformers |
| **Base LR** | 5e-5 (backbone), 1e-3 (head) | Lower lr for pretrained layers |
| **Warmup** | 5 epochs linear | Prevent early instability |
| **Scheduler** | CosineAnnealing | Smooth convergence |
| **Batch Size** | 32-64 | Balance memory and stability |
| **Weight Decay** | 0.01 | Transformer regularization |
| **Gradient Clip** | 1.0 | Prevent explosion |
| **Max Epochs** | 50 | Early stopping likely ~20-30 |
| **Early Stopping** | 7 epochs patience | Same as Phase 2 |

### Multi-Stage Fine-Tuning (Optional Advanced)

**If using multimodal fusion (Phase 4 preview)**:
1. **Stage 1**: Fine-tune audio encoder (AST) on Xeno-Canto alone
2. **Stage 2**: Freeze encoder, train fusion layers
3. **Stage 3**: Unfreeze all, end-to-end fine-tuning (lr=1e-5)

**Phase 3**: Single-stage fine-tuning sufficient (unimodal audio only).

---

## Implementation Plan

### Phase 3 User Stories

**US1**: Replace MFCC with Log-Mel Spectrogram pipeline  
**US2**: Implement AST (Audio Spectrogram Transformer) with AudioSet pretraining  
**US3**: Implement augmentation pipeline (SpecAugment, MixUp)  
**US4**: Baseline training AST on LMS features (target: 60% acc)  
**US5**: Advanced training with full augmentation (target: 70% acc)  
**US6**: Implement PANNs/EfficientNet for ensemble comparison  

### Task Breakdown (High-Level)

**Setup & Feature Engineering (US1)** - 8-12 hours
- Implement LMS extraction in `src/features/audio.py`
- Update `AudioMFCCDataset` → `AudioSpectrogramDataset`
- Generate LMS cache for all 11,075 recordings
- Validate: Visual inspection of spectrograms, check shapes

**AST Implementation (US2)** - 6-10 hours
- Install transformers library, download AST checkpoint
- Create `src/models/audio_ast.py` wrapper
- Integrate into `scripts/03_train_audio.py`
- Smoke test: 1 epoch training

**Augmentation (US3)** - 4-6 hours
- Implement SpecAugment in dataset class
- Implement MixUp in training loop
- Optional: Noise injection pipeline
- Validate: Visual inspection of augmented spectrograms

**Baseline Training (US4)** - 12-16 GPU hours
- Train AST + FocalLoss on LMS (no augmentation)
- Monitor convergence speed (<15 epochs to 40% target)
- Evaluate: Expect 55-65% accuracy

**Full Training (US5)** - 12-16 GPU hours
- Train AST + FocalLoss + SpecAugment + MixUp
- Target: 70%+ accuracy
- Evaluate: mAP, F1-macro, per-class F1

**Ensemble (US6 - Optional)** - 8-12 GPU hours
- Implement EfficientNet baseline
- Train and evaluate
- Ensemble AST + EfficientNet (weighted average)

**Total Estimated**:
- Human time: 25-35 hours
- GPU time: 32-44 hours (within 44-hour Phase 3 budget)

---

## Evaluation Metrics

### Primary Metrics (Prioritized)

1. **mAP (Mean Average Precision)** - Better than accuracy for imbalance
2. **F1-Macro** - Equal weight to all species (current: 0.2167)
3. **Test Accuracy** - Overall correct predictions (current: 42.72%)
4. **Per-Class F1** - Identify worst-performing species

### Why mAP is Critical

> "mAP and Macro-F1 score are required to accurately reflect the model's generalized performance across all 90 species, especially minority classes."

**Implementation**:
```python
from sklearn.metrics import average_precision_score

# Per-class AP
aps = []
for class_idx in range(90):
    ap = average_precision_score(
        y_true_binary[:, class_idx], 
        y_pred_proba[:, class_idx]
    )
    aps.append(ap)

mAP = np.mean(aps)  # Target: >0.50
```

### Success Thresholds

| Metric | Phase 2B | Phase 3 MVP | Phase 3 Target | Phase 3 Stretch |
|--------|----------|-------------|----------------|-----------------|
| **Test Accuracy** | 42.72% | >55% | **>65%** | **>75%** |
| **F1-Macro** | 0.2167 | >0.35 | **>0.45** | **>0.55** |
| **mAP** | N/A | >0.40 | **>0.50** | **>0.60** |
| **Min Per-Class F1** | Unknown | >0.10 | **>0.20** | **>0.30** |

---

## Risks & Mitigation

### Technical Risks

**Risk 1: Pretrained model compatibility**
- **Issue**: AST expects specific input format, may not match our LMS generation
- **Mitigation**: Follow exact AudioSet preprocessing (torchaudio.transforms), validate input shapes
- **Fallback**: Fine-tune from ImageNet ViT directly on our LMS

**Risk 2: GPU memory overflow**
- **Issue**: AST (86M params) + batch_size=64 may exceed 6GB VRAM
- **Mitigation**: Reduce batch_size to 16-32, use gradient accumulation
- **Fallback**: Use EfficientNet-B0 (9M params) instead

**Risk 3: Augmentation instability**
- **Issue**: Aggressive MixUp/SpecAugment may hurt convergence
- **Mitigation**: Start conservative (lower masking params), gradually increase
- **Fallback**: Disable MixUp, keep only SpecAugment

**Risk 4: Training time exceeds budget**
- **Issue**: 50 epochs × 2 models × 0.5h/epoch = 50 GPU hours (exceeds 44h budget)
- **Mitigation**: Use early stopping aggressively (patience=5), expect convergence at 20 epochs
- **Fallback**: Train only AST, skip PANNs comparison

### Data Risks

**Risk 5: Cache generation time**
- **Issue**: Generating LMS for 11,075 recordings may take 6-12 hours
- **Mitigation**: Parallelize with multiprocessing, cache incrementally
- **Fallback**: Generate on-the-fly (slower training but no upfront cost)

**Risk 6: Noise corpus unavailable**
- **Issue**: Need environmental noise samples for augmentation
- **Mitigation**: Extract "silence" segments from Xeno-Canto itself
- **Fallback**: Skip noise injection, rely on SpecAugment + MixUp

---

## Dependencies & Prerequisites

### Software Requirements

**New Libraries**:
- `transformers>=4.30.0` (Hugging Face, for AST)
- `torchaudio>=2.0.0` (audio transforms, SpecAugment)
- `timm>=0.9.0` (optional, for EfficientNet)

**Existing**:
- PyTorch 2.9.1, librosa 0.11.0, scikit-learn 1.7.2

### Hardware Requirements

- **GPU**: NVIDIA RTX 3060 (6GB) - adequate with batch_size=16-32
- **Storage**: +5GB for LMS cache (128×130 float32 per sample)
- **Compute**: 32-44 GPU hours budget (Phase 3 allocation)

### Data Requirements

- **Xeno-Canto**: 11,075 recordings (already downloaded)
- **Preprocessing**: Resample to 22,050 Hz, extract 3-5s segments
- **Optional**: Environmental noise corpus (ESC-50 or custom)

---

## Success Validation

### Gate 1: Feature Engineering (After US1)

**Criteria**:
- ✅ LMS cache generated for all 11,075 samples
- ✅ Visual inspection: spectrograms show clear temporal structure
- ✅ Shape validation: (128, T) where T ∈ [130, 216] for 3-5s
- ✅ Smoke test: Load LMS in DataLoader, no errors

**Go/No-Go**: If visualization unclear or shape mismatch → Debug before training

### Gate 2: AST Baseline (After US4)

**Criteria**:
- ✅ Convergence speed: Reach 40% val acc within 15 epochs (Phase 2: 26 epochs)
- ✅ No training collapse (loss decreases smoothly)
- ✅ Accuracy >50% (minimum viable improvement over Phase 2)

**Go/No-Go**: 
- If >50% acc → Proceed to US5 (augmentation)
- If 42-50% acc → Debug (check LMS params, learning rate)
- If <42% acc → STOP, fundamental issue (revert to Phase 2, investigate)

### Gate 3: Full Training (After US5)

**Criteria**:
- ✅ Test accuracy >60% (Phase 3 MVP)
- ✅ F1-macro >0.40
- ✅ mAP >0.45
- ✅ Training stable (no overfitting collapse)

**Go/No-Go**:
- If >65% acc → **SUCCESS**, proceed to Phase 4 (multimodal fusion)
- If 60-65% acc → **PARTIAL SUCCESS**, iterate on augmentation
- If <60% acc → **INVESTIGATE**, analyze per-class errors, consider architecture changes

---

## Open Questions

1. **CQT (Constant-Q Transform) fusion**: Research mentions CQT for pitch-based features. Should we generate both LMS + CQT and fuse early in the network? (Deferred to Phase 3 advanced work)

2. **Segment length**: 3s vs 5s clips? Longer = more context, but fewer training samples. (Start with 5s, ablate if needed)

3. **Sample rate**: 22.05 kHz vs 32 kHz? Trade-off between detail and compute. (Start with 22.05 kHz, standard in literature)

4. **Ensemble weights**: If implementing AST + EfficientNet ensemble, how to weight? Equal average or learned? (Start with equal, optimize if time allows)

5. **Class balancing**: Continue with FocalLoss α or add explicit upsampling? (Keep FocalLoss with tuned α, proven in Phase 2)

---

## References & Research Sources

1. **Feature Engineering**: "MFCCs discard fine-grained spectro-temporal details" - Research Section I.1
2. **Log-Mel Spectrograms**: "Convert audio processing into image processing" - Research Section I.2
3. **AudioSet Transfer Learning**: "5,000+ hours, 527 classes" - Research Section II.1
4. **AST Architecture**: "Purely attention-based, captures long-range context" - Research Section II.3
5. **PANNs**: "Wavegram-Logmel-CNN, proven in BirdCLEF" - Research Section II.2
6. **SpecAugment**: "Masking of time and frequency blocks" - Research Section III.2
7. **MixUp**: "Linear interpolation of inputs and labels" - Research Section III.2
8. **Focal Loss**: "Prioritizes hard, rare examples" - Research Section III.1
9. **Evaluation**: "mAP and Macro-F1 required for imbalanced data" - Research Section V.2

---

## Approval & Sign-Off

**Prepared by**: AI Agent (GitHub Copilot) based on deep research findings  
**Date**: December 4, 2025  
**Status**: Awaiting approval to proceed to technical planning (plan.md)

**Stakeholder Review**:
- [ ] Technical feasibility validated (architecture, dependencies)
- [ ] Resource allocation approved (32-44 GPU hours, 25-35 human hours)
- [ ] Success criteria agreed upon (60%+ MVP, 70%+ target)
- [ ] Risk mitigation acceptable (fallbacks defined)

**Next Steps**:
1. Review and approve this specification
2. Generate detailed technical plan (plan.md) with architecture diagrams
3. Break down into task list (tasks.md) with 50-80 atomic tasks
4. Begin implementation starting with US1 (feature engineering)

---

**Document Version**: 1.0  
**Last Updated**: December 4, 2025  
**Related Documents**: 
- Phase 2 Results: `specs/003-phase2-focal-loss-improvements/PHASE2_RESULTS.md`
- Research Report: (Provided in planning session)
