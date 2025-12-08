# Feature Specification: Paper Audio Integration

**Feature Branch**: `006-paper-audio-integration`
**Created**: 2025-12-08
**Status**: Draft
**Input**: User description: "Review and polish paper, add audio experiments/methodology, address dataset intersection, validate pages/references."

## User Scenarios & Testing

### User Story 1 - Audio Methodology Integration (Priority: P1)

As a reader, I need to understand the specific audio processing techniques and architectures used so that I can assess the scientific validity of the audio classification approach.

**Why this priority**: The paper currently lacks the "Audio" half of the "Audiovisual" title. This is critical for the paper's completeness.

**Independent Test**: The "Methodology" section contains detailed subsections on "Audio Feature Engineering" (LMS vs MFCC) and "Audio Architectures" (AudioCNN, AST).

**Acceptance Scenarios**:
1. **Given** the Methodology section, **When** read, **Then** it explains why MFCCs were replaced by Log-Mel Spectrograms.
2. **Given** the Methodology section, **When** read, **Then** it describes the AudioCNN and AST architectures.
3. **Given** the Methodology section, **When** read, **Then** it details the strategies for class imbalance (Focal Loss, Augmentation).

---

### User Story 2 - Audio Results & Dataset Clarification (Priority: P1)

As a reader, I need to see the audio experiment results and clearly understand the dataset constraints (200 vs 90 species) to avoid confusion about model comparability.

**Why this priority**: Misunderstanding the dataset split (full CUB vs intersection) would invalidate the results in the reader's eyes.

**Independent Test**: The "Experiments" section explicitly states the dataset difference and presents audio results for the 90-species subset.

**Acceptance Scenarios**:
1. **Given** the Experiments section, **When** read, **Then** it explicitly states that Image models used CUB-200 (200 classes) while Audio models used the Xeno-Canto intersection (90 classes).
2. **Given** the Experiments section, **When** read, **Then** it presents Image Classification results specifically for the 90-species intersection dataset, separate from the full CUB-200 results.
3. **Given** the Results section, **When** read, **Then** it presents the progression of Audio model performance from Phase 0 (Baseline) to Phase 3 (AST+LMS).

---

### User Story 3 - Bibliography & Formatting (Priority: P2)

As a reviewer, I expect a well-formatted bibliography and adherence to page limits.

**Why this priority**: Essential for acceptance at ICML.

**Independent Test**: The paper compiles with `pdflatex` and the bibliography contains all cited works.

**Acceptance Scenarios**:
1. **Given** the `example_paper.tex` file, **When** compiled with `pdflatex`, **Then** it produces a PDF with no errors.
2. **Given** the PDF, **When** checked, **Then** it is between 8 and 10 pages long.
3. **Given** the Bibliography, **When** checked, **Then** it includes the references provided in the technical report.

## Functional Requirements

1.  **Update `example_paper.tex` Methodology**:
    *   Insert "Audio Feature Engineering" section: Explain MFCC limitations (38% acc) vs LMS advantages.
    *   Insert "Audio Architectures" section: Describe AudioCNN (PANNs-inspired) and AST (Transformer).
    *   Insert "Class Imbalance Mitigation" section: Describe Focal Loss ($\gamma=2.0$) and Augmentation (SpecAugment, MixUp).

2.  **Update `example_paper.tex` Experiments**:
    *   Clarify Dataset: Explicitly distinguish CUB-200 (200 classes) vs Intersection (90 classes).
    *   Insert Image Results on Intersection: Report performance of image models on the 90-species subset.
    *   Insert Audio Results Table:
        *   Phase 0 (MFCC+CNN/ViT): ~39%
        *   Phase 1 (Weights): Failed
        *   Phase 2 (Focal+Capacity): 42.24%
        *   Phase 3 (LMS+AST): 57.28%

3.  **Update `example_paper.bib`**:
    *   Add BibTeX entries for all references listed in the prompt (AST, PANNs, Focal Loss, SpecAugment, etc.).

4.  **Validation**:
    *   Ensure `pdflatex` compilation.
    *   Check page count (8-10 pages).

## Success Criteria

*   **Compilation**: `pdflatex` runs successfully.
*   **Completeness**: Audio methodology and results are fully integrated.
*   **Clarity**: Dataset distinction (200 vs 90) is unambiguous.
*   **Length**: Final PDF is 8-10 pages.

## Assumptions

*   The existing Image sections in `example_paper.tex` are accurate and do not need major rewrites, only integration.
*   The provided technical report text is the source of truth for the Audio sections.
