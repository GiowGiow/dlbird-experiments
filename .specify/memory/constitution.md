# SpeckitDLBird Constitution
## Core Principles


### I. Reproducibility-First

All experiments must be fully reproducible: deterministic seeds, pinned dependencies managed with UV, versioned datasets/artifacts, and executable notebooks. Each result corresponds to a specific notebook and commit.

<!-- Deterministic runs, environment pinning, and data versioning are mandatory. -->

### II. Clean, Concise Code with Documentation

Code and notebooks must be minimal, readable, and documented inline. Each deep learning architecture includes a brief design rationale and clear hyperparameters. Experiments are described in notebook markdown cells.

<!-- Clarity over cleverness; comments explain "why" not just "what". -->

### III. Data Ethics and Compliance

We only use bird datasets (audio and image) with appropriate licenses; no personally identifiable information. Attributions and licenses are recorded; dataset splits avoid leakage.

<!-- Ethics and proper attribution are foundational to this research. -->

### IV. Results Traceability

Each experiment result corresponds to a notebook section and saved artifact path. Figures and metrics are generated programmatically and saved with unique, descriptive filenames.

<!-- Notebooks produce results deterministically with timestamps and seeds. -->

### V. Simplicity and Baselines

Start with strong classical and simple baselines before complex models. Experiment with both audio (CNN, ViT) and image (ResNet, ViT) modalities for bird species classification.

<!-- YAGNI: include only what improves clarity and insight. -->

## Development Workflow, Experiments, and Notebooks

We use multiple Jupyter notebooks, each focused and self-contained. UV manages Python packages and lockfiles.

Notebook plan (names and purposes):

- 00_env_setup.ipynb: verify environment setup and dependencies.
- 01_intersection.ipynb: identify common species between CUB-200 (images) and Xeno-Canto (audio) datasets; create stratified splits.
- 02_audio_features.ipynb: extract and cache audio features (MFCCs); prepare for audio model training.
- 03_image_models.ipynb: train and evaluate image classification models (ResNet-18, ViT).
- 04_training_compare.ipynb: train and compare audio models (CNN, ViT); consolidate results across modalities.

Environment management (UV):

- Dependencies managed via `pyproject.toml` and locked with UV.
- Key packages: torch, torchvision, torchaudio, timm, librosa, scikit-learn, numpy, pandas, matplotlib, seaborn, jupyter, ipykernel, rich.
- Lock all deps; record Python version; create kernel with the env name.
- Environment setup documented in `00_env_setup.ipynb`.

Architecture documentation:

- For each DL model, include a markdown cell with: input size, backbone, number of params (estimate), optimizer, lr schedule, regularization, augmentation, epochs.
- Provide a short "why this design" rationale emphasizing simplicity and efficiency.
- Models implemented: Audio CNN, Audio ViT, Image ResNet-18, Image ViT.

Experiment metadata:

- Set seeds (e.g., 42) for numpy, torch, and dataloaders; enable cudnn deterministic where applicable.
- Save artifacts under `artifacts/` with run identifiers; log config JSON next to checkpoints.
- Results saved as JSON files in `artifacts/results/` for easy comparison and analysis.

## Project Structure

The project follows SpecKit organizational patterns with clear separation between specifications, implementation, and experimental results.

### Core Implementation Structure

- `src/`: Core implementation modules
  - `data/`: Dataset loaders for CUB-200 and Xeno-Canto
  - `datasets/`: PyTorch dataset classes for audio and image data
  - `features/`: Audio feature extraction (MFCC caching)
  - `models/`: Model implementations (Audio CNN/ViT, Image ResNet/ViT)
  - `training/`: Training loops and utilities
  - `evaluation/`: Metrics and result aggregation
  - `utils/`: Species mapping and data splitting utilities

- `notebooks/`: Jupyter notebooks for experiments (see Notebook plan above)
- `scripts/`: Standalone Python scripts for running experiments
- `artifacts/`: Saved models, results, splits, and cached features
- `data/`: Raw dataset directories (not tracked in git)

### SpecKit Documentation Structure

Following SpecKit methodology, all specifications, plans, and feature documentation are organized under `.specify/`:

```
.specify/
├── memory/
│   ├── constitution.md          # This file - project governing principles
│   ├── technical_plan.md         # High-level technical roadmap
│   └── tasks.md                  # Global task tracking
├── scripts/
│   └── [utility scripts for spec management]
├── templates/
│   ├── spec-template.md          # Template for feature specifications
│   ├── plan-template.md          # Template for implementation plans
│   ├── tasks-template.md         # Template for task breakdowns
│   └── checklist-template.md     # Template for validation checklists
└── specs/
    ├── 001-validation-phase/
    │   ├── spec.md               # Feature specification (what to build)
    │   ├── plan.md               # Implementation plan (how to build)
    │   ├── quickstart.md         # Quick start guide
    │   ├── research.md           # Technical research and decisions
    │   └── artifacts/            # Validation outputs, plots, analysis
    ├── 002-phase1-critical-fixes/
    │   ├── spec.md               # Class imbalance & normalization fixes
    │   ├── plan.md               # Detailed implementation steps
    │   ├── tasks.md              # Actionable task breakdown
    │   ├── quickstart.md         # Immediate action guide
    │   └── artifacts/            # Experiment results, checkpoints
    └── 003-future-feature/
        └── ...
```

### Specification Numbering Convention

Features and experimental phases are numbered sequentially with descriptive names:
- `001-validation-phase`: Initial model validation and root cause analysis
- `002-phase1-critical-fixes`: Class weighting and feature normalization
- `003-phase2-feature-engineering`: Mel-spectrograms and augmentation
- `004-phase3-architecture-optimization`: Pretrained models and architecture search
- `005-phase4-advanced-techniques`: Multi-modal fusion and ensembles

Each specification directory contains:
- **spec.md**: Functional requirements and user stories (tech-stack agnostic)
- **plan.md**: Technical implementation plan with architecture decisions
- **tasks.md**: Ordered task breakdown with dependencies and parallel execution markers
- **quickstart.md**: Immediate action steps for starting implementation
- **research.md**: Technical research, library versions, and design rationale
- **artifacts/**: Experimental outputs, plots, checkpoints, analysis files

### Document Organization Rules

1. **Specifications First**: Write `spec.md` before `plan.md` - define what to build before how to build it
2. **Clarify Before Planning**: Use clarification workflow to refine specs before creating technical plans
3. **Tasks from Plans**: Generate `tasks.md` from validated `plan.md` with clear dependencies
4. **Artifacts Co-located**: Keep experimental results with their corresponding spec for traceability
5. **Version Control**: All spec documents are versioned; amendments documented in spec history section

## Governance

Constitution supersedes all other practices. Amendments require documentation and thoughtful consideration of impact on reproducibility and project structure.

This constitution supersedes ad-hoc practices for this project. Amendments require:

1. A short proposal documenting the change and rationale.
2. Review by project maintainers.
3. Migration plan for affected notebooks, figures, and artifacts.

Compliance rules:
- Complexity must be justified within the 4-page constraint.
- Use `GUIDANCE.md` for runtime guidance and `README.md` for quickstart.

**Version**: 0.1.0 | **Ratified**: 2025-12-03 | **Last Amended**: 2025-12-03
<!-- Initial version aligned to ICML 2025 short paper requirements -->
