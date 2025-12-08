# Data Model: Paper Audio Integration

**Date**: 2025-12-08

## Document Structure (LaTeX Sections)

### 1. Methodology

*   **Subsection: Audio Feature Engineering**
    *   **Content**: Critique of MFCCs (38% acc), introduction of Log-Mel Spectrograms (LMS).
    *   **Key Parameters**: $F_s=22.05$ kHz, $n\_fft=2048$, $n\_mels=128$, Hop=512.
*   **Subsection: Network Architectures**
    *   **Content**: Description of AudioCNN (PANNs-inspired) and Audio Spectrogram Transformer (AST).
    *   **Focus**: Transfer learning from AudioSet.
*   **Subsection: Class Imbalance Mitigation**
    *   **Content**: Focal Loss ($\gamma=2.0$) implementation.
    *   **Content**: Data Augmentation (SpecAugment, MixUp).

### 2. Experiments

*   **Subsection: Datasets**
    *   **Entity**: CUB-200-2011
        *   **Scope**: Image models only.
        *   **Classes**: 200.
    *   **Entity**: Xeno-Canto Intersection
        *   **Scope**: Audio models only.
        *   **Classes**: 90 (intersection with CUB).
        *   **Constraint**: Explicitly state this difference to avoid confusion.

### 3. Results

*   **Table: Audio Model Performance**
    *   **Columns**: Model, Features, Loss, Accuracy, F1-Macro.
    *   **Rows**:
        *   Baseline (MFCC + CNN): ~39%
        *   Phase 2B (MFCC + CNNv2 + Focal): 42.24%
        *   Phase 3 (LMS + AST + Aug): 57.28%

### 4. Discussion / Future Work

*   **Subsection: Multimodal Fusion**
    *   **Content**: Propose Intermediate Feature Fusion of ViT (Image) and AST (Audio) as the next step.
    *   **Rationale**: Leveraging complementary information from both modalities.

## Bibliography Entities

*   **New References**:
    *   AST (Gong et al., 2021)
    *   PANNs (Kong et al., 2020)
    *   Focal Loss (Lin et al., 2017)
    *   SpecAugment (Park et al., 2019)
    *   MixUp (Zhang et al., 2017)
