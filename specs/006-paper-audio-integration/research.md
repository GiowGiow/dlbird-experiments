# Research Findings: Paper Audio Integration

**Date**: 2025-12-08
**Source**: Technical Report "Deep Learning Methodologies for Avian Species Identification"

## 1. Audio Feature Engineering

*   **Decision**: Replace MFCCs with Log-Mel Spectrograms (LMS).
*   **Rationale**:
    *   MFCCs yielded only 38% accuracy, proving incompatible with 2D deep learning architectures (CNN/ViT).
    *   LMS provides a dense, image-like representation that preserves spectro-temporal details, enabling effective transfer learning from ImageNet/AudioSet.
    *   **Parameters**: $F_s=22.05$ kHz, $n\_fft=2048$, $n\_mels=128$, Hop Length=512.

## 2. Network Architectures

*   **Decision**: Adopt PANNs (AudioCNN) and AST (Audio Spectrogram Transformer).
*   **Rationale**:
    *   **PANNs (AudioCNN)**: Serves as the robust convolutional benchmark, leveraging Wavegram-Logmel fusion concepts.
    *   **AST**: Provides a pure attention-based model analogous to ViT, capturing global context crucial for bird song syntax. Outperforms CNNs in sound event detection.

## 3. Class Imbalance Mitigation

*   **Decision**: Implement Focal Loss and Advanced Augmentation.
*   **Rationale**:
    *   **Focal Loss**: $\gamma=2.0$ focuses training on "hard" examples, countering the 1216:1 class imbalance.
    *   **SpecAugment**: Time/Frequency masking forces robustness against signal dropouts.
    *   **MixUp**: Linear interpolation smooths decision boundaries for rare classes.

## 4. Dataset Clarification

*   **Decision**: Explicitly distinguish between CUB-200 and Intersection datasets.
*   **Rationale**:
    *   Image experiments in the literature (and potentially some baselines) use the full CUB-200 (200 classes).
    *   Our Audio/Fusion experiments use the Intersection of CUB-200 and Xeno-Canto (90 classes).
    *   **Constraint**: Results must be reported separately. Specifically, we must report our Image Classification results on the 90-species intersection to provide a fair baseline for our Audio models, distinct from the full CUB-200 benchmarks.

## 5. Multimodal Fusion (Future Work)

*   **Decision**: Recommend Intermediate Feature Fusion as a future direction.
*   **Rationale**: Concatenating feature vectors from specialized encoders (ViT + AST) allows learning higher-order correlations, superior to late fusion.
