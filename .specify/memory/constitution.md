# SpeckitDLBird Constitution
<!-- Short ICML-style bird classification project constitution -->

## Core Principles


### I. Reproducibility-First

All experiments must be fully reproducible: deterministic seeds, pinned dependencies managed with UV, versioned datasets/artifacts, and executable notebooks. Each result in the paper maps to a specific notebook and commit.

<!-- Deterministic runs, environment pinning, and data versioning are mandatory. -->

### II. Clean, Concise Code with Documentation

Code and notebooks must be minimal, readable, and documented inline. Each deep learning architecture includes a brief design rationale and clear hyperparameters. Experiments are described in notebook markdown cells.

<!-- Clarity over cleverness; comments explain "why" not just "what". -->

### III. Data Ethics and Compliance

We only use bird image datasets with appropriate licenses; no personally identifiable information. Attributions and licenses are recorded; dataset splits avoid leakage.

<!-- Ethics statement is included per ICML 2025 requirements. -->

### IV. Results Traceability

Each table/figure in the ICML paper corresponds to a notebook section and saved artifact path. Figures are generated programmatically and saved with unique, descriptive filenames.

<!-- Notebooks produce figures/tables deterministically with timestamps and seeds. -->

### V. Simplicity and Baselines

Start with strong classical and simple CNN baselines before complex models. Report ablations sparingly to fit a short 4-page limit.

<!-- YAGNI: include only what improves clarity and insight. -->

## Paper Requirements (ICML 2025 Short, 4 pages)

The article follows the ICML 2025 template (neurips-like style if applicable) with the following minimal sections:

1. Title and Authors: concise, descriptive title; affiliations; contact.
1. Abstract (≤150 words): task, dataset(s), methods (baseline + DL), key results.
1. Introduction (½ page): motivation for bird classification; contributions in bullet form.
1. Related Work (½ page): brief coverage of bird classification and lightweight CNNs/transformers.

1. Method (1 page):

- Dataset description and splits.
- Baseline: linear SVM on frozen features (e.g., ResNet-18 pretrained) or simple k-NN.
- Deep Learning: small CNN and/or MobileNetV3/ResNet-18 fine-tuning; architecture summary and training setup.
- Losses, metrics, and evaluation protocol.


1. Experiments (1 page):

- Training details: UV environment, seeds, batch size, lr schedule.
- Results: accuracy, macro-F1; tables created from notebook outputs.
- Ablations: minimal (e.g., input resolution, augmentation on/off).


1. Discussion/Conclusion (½ page): key insights and limitations.
1. Ethics Statement: data licensing, animal welfare considerations.
1. Reproducibility Checklist: environment, data sources, code release, seeds.
1. References: concise, relevant citations.

Formatting constraints:

- Main text ≤4 pages; references may exceed. Figures and tables counted in the 4-page limit.
- Use the ICML 2025 LaTeX template; no font or spacing changes.
- Provide source of figures and tables and ensure legibility.

## Development Workflow, Experiments, and Notebooks

We use multiple Jupyter notebooks, each focused and self-contained. UV manages Python packages and lockfiles.

Notebook plan (names and purposes):

- 01_data_prep.ipynb: download/verify dataset licensing; create stratified splits; basic EDA and class distribution.
- 02_features_baseline.ipynb: extract pretrained features; train SVM/k-NN baseline; report metrics.
- 03_cnn_finetune.ipynb: fine-tune lightweight CNN (ResNet-18/MobileNetV3); log training curves and best metrics.
- 04_ablations.ipynb: small ablations (image size, augmentation); summarize impact.
- 05_figures_tables.ipynb: consolidate results, generate paper-ready tables and figures.

Environment management (UV):

- uv init; uv add torch torchvision torchaudio timm scikit-learn numpy pandas matplotlib seaborn jupyter ipykernel rich
- Lock all deps; record Python version; create kernel with the env name.
- Store exact uv commands and versions in 00_env_setup.md and notebooks’ first cell.

Architecture documentation:

- For each DL model, include a markdown cell with: input size, backbone, number of params (estimate), optimizer, lr schedule, regularization, augmentation, epochs.
- Provide a short “why this design” rationale emphasizing simplicity and efficiency.

Experiment metadata:

- Set seeds (e.g., 42) for numpy, torch, and dataloaders; enable cudnn deterministic where applicable.
- Save artifacts under `artifacts/` with run identifiers; log config JSON next to checkpoints.

## Governance
<!-- Example: Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->

This constitution supersedes ad-hoc practices for this project. Amendments require:

1. A short proposal documenting the change and rationale.
2. Review by project maintainers.
3. Migration plan for affected notebooks, figures, and artifacts.

Compliance rules:

- All PRs verify paper section completeness and notebook-to-result mapping.
- Complexity must be justified within the 4-page constraint.
- Use `GUIDANCE.md` for runtime guidance and `README.md` for quickstart.

**Version**: 0.1.0 | **Ratified**: 2025-12-03 | **Last Amended**: 2025-12-03
<!-- Initial version aligned to ICML 2025 short paper requirements -->
