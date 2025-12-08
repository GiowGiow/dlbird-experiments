# SpecKit DLBird Constitution

<!-- Sync Impact Report
Version: 1.0.0 -> 2.0.0
Modified Principles:
- Reproducibility-First -> Academic Rigor (Refocused)
- Clean, Concise Code -> LaTeX Integrity (Refocused)
- Data Ethics -> Cohesion (Refocused)
- Results Traceability -> Constraint Management (Refocused)
- Simplicity -> Reproducibility & Transparency (Refocused)
Added Sections:
- Governance
- Paper Production Workflow
Removed Sections:
- Development Workflow (Replaced by Paper Workflow)
Templates requiring updates:
- .specify/templates/plan-template.md (✅ updated implicitly by constitution reference)
- .specify/templates/spec-template.md (✅ generic)
- .specify/templates/tasks-template.md (✅ generic)
Follow-up TODOs:
- None
-->

## Core Principles

### I. Academic Rigor

The primary output is a scientific paper (`example_paper.tex`). All content must adhere to formal scientific standards: passive voice, objective analysis, and precise terminology. Claims must be backed by data or citations.

### II. LaTeX Integrity

The paper must always compile with `pdflatex` without errors. Use standard ICML 2025 packages. Bibliography (`example_paper.bib`) must be valid and complete.

### III. Cohesion

The paper must read as a single, unified study. Audio and Image sections must be integrated, not just concatenated. Discussion and Conclusion must synthesize findings from both modalities.

### IV. Constraint Management

Strict adherence to the ICML page limit (8-10 pages). Content must be concise and high-value. Space usage is a critical resource to be managed.

### V. Reproducibility & Transparency

Explicitly address the dataset constraint: Image experiments on full CUB-200 vs. Image/Audio experiments on the 90-species intersection. This distinction must be clear to the reader to avoid confusion.

## Governance

**Ratification Date**: 2025-12-03
**Last Amended Date**: 2025-12-08
**Constitution Version**: 2.0.0

### Amendment Procedure

Changes to this constitution require consensus from both Image and Audio teams.

### Versioning

Follows Semantic Versioning.

- MAJOR: Phase shifts (e.g., Dev -> Paper).
- MINOR: New principles or section additions.
- PATCH: Clarifications and fixes.

### Compliance

- Automated: `pdflatex` compilation checks.
- Manual: Peer review for tone, cohesion, and page limits.

## Paper Production Workflow

We are in the "Academic Paper Production" phase.

**Primary Artifact**: `paper/example_paper.tex`

**Team Roles**:

- **Image Team**: Introduction, Related Work, Image Methodology/Results.
- **Audio Team**: Audio Methodology, Audio Results, Future Fusion Strategy.

**Key Context**:

- **Title**: "Fine-Grained Audiovisual Categorization of Birds Using Popular Datasets"
- **Template**: ICML 2025 LaTeX
- **Constraint**: 8-10 pages

**Execution Strategy**:

1. **Drafting**: Fill missing sections (Audio/Fusion) using the provided technical reports.
2. **Integration**: Merge sections ensuring consistent terminology and flow.
3. **Validation**: Compile, check page count, review content against principles.
4. **Refinement**: Polish language, improve figures, trim excess to fit limits.

## Project Structure

- `paper/`: The core workspace.
  - `example_paper.tex`: The manuscript.
  - `example_paper.bib`: References.
  - `icml2025.sty`: Style file.
- `artifacts/`: Source of truth for results (figures, tables).
- `src/` & `notebooks/`: Reference implementation (frozen for paper consistency). Very important for network architecture
- `specs/001` through `specs/004` have .md RESULTS which detail 