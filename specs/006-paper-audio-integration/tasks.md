# Tasks: Paper Audio Integration

**Feature**: Paper Audio Integration
**Status**: In Progress

## Phase 1: Setup
**Goal**: Prepare the environment and secure existing work.

- [x] T001 Verify LaTeX environment (pdflatex, bibtex) availability
- [x] T002 Backup `paper/example_paper.tex` and `paper/example_paper.bib` to `paper/backups/`

## Phase 2: Foundational
**Goal**: Establish bibliographic foundation for citations.

- [x] T003 Add AST reference (Gong et al., 2021) to `paper/example_paper.bib`
- [x] T004 Add PANNs reference (Kong et al., 2020) to `paper/example_paper.bib`
- [x] T005 Add Focal Loss reference (Lin et al., 2017) to `paper/example_paper.bib`
- [x] T006 Add SpecAugment reference (Park et al., 2019) to `paper/example_paper.bib`
- [x] T007 Add MixUp reference (Zhang et al., 2017) to `paper/example_paper.bib`

## Phase 3: Audio Methodology Integration (User Story 1)
**Goal**: Document the specific audio processing techniques and architectures.
**Story**: [US1] Audio Methodology Integration

- [x] T008 [US1] Insert "Audio Feature Engineering" section (LMS vs MFCC) in `paper/example_paper.tex`
- [x] T009 [US1] Insert "Audio Architectures" subsection describing AudioCNN/PANNs in `paper/example_paper.tex`
- [x] T010 [US1] Insert "Audio Architectures" subsection describing AST in `paper/example_paper.tex`
- [x] T011 [US1] Insert "Class Imbalance Mitigation" subsection describing Focal Loss in `paper/example_paper.tex`
- [x] T012 [US1] Insert "Class Imbalance Mitigation" subsection describing SpecAugment/MixUp in `paper/example_paper.tex`

## Phase 4: Audio Results & Dataset Clarification (User Story 2)
**Goal**: Present results clearly with correct dataset context.
**Story**: [US2] Audio Results & Dataset Clarification

- [x] T013 [US2] Update "Datasets" section to explicitly distinguish CUB-200 (200 classes) vs Intersection (90 classes) in `paper/example_paper.tex`
- [x] T013b [US2] Report Image Classification results on Intersection Dataset (90 classes) in `paper/example_paper.tex`
- [x] T014 [US2] Create "Audio Results" table structure in `paper/example_paper.tex`
- [x] T015 [US2] Populate "Audio Results" table with Phase 0-3 data (39% -> 57.28%) in `paper/example_paper.tex`
- [x] T016 [US2] Add analysis text describing the performance progression and impact of Focal Loss/AST in `paper/example_paper.tex`
- [x] T016b [US2] Add "Future Work" section proposing Intermediate Feature Fusion in `paper/example_paper.tex`

## Phase 5: Bibliography & Formatting (User Story 3)
**Goal**: Ensure academic compliance and formatting.
**Story**: [US3] Bibliography & Formatting

- [x] T017 [US3] Compile paper with `pdflatex` and `bibtex` to resolve all citations in `paper/example_paper.tex`
- [x] T018 [US3] Verify page count is between 8-10 pages and adjust spacing if needed in `paper/example_paper.tex`
- [x] T019 [US3] Final proofread for consistency between Image and Audio sections in `paper/example_paper.tex`

## Dependencies

1.  **Phase 2 (Bibliography)** must be completed before **Phase 3** and **Phase 4** to avoid undefined citation errors during intermediate compilations.
2.  **Phase 3 (Methodology)** and **Phase 4 (Results)** can be executed in parallel, but it is recommended to do Methodology first to establish context for the Results.

## Implementation Strategy

1.  **MVP**: Complete Phase 1, 2, and T013 (Dataset Clarification) + T015 (Results Table). This ensures the core scientific claims are accurate.
2.  **Full Feature**: Complete all Methodology sections (Phase 3) to provide the necessary scientific backing.
3.  **Polish**: Phase 5 ensures the paper is submission-ready.
