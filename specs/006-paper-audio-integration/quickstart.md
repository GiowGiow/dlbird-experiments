# Quickstart: Paper Compilation

**Date**: 2025-12-08

## Prerequisites

*   **LaTeX Distribution**: TeX Live 2024 (or compatible)
*   **Packages**: `icml2025`, `natbib`, `graphicx`, `amsmath`, `amssymb`

## Build Instructions

1.  **Navigate to paper directory**:
    ```bash
    cd paper
    ```

2.  **Compile LaTeX**:
    ```bash
    pdflatex example_paper.tex
    bibtex example_paper
    pdflatex example_paper.tex
    pdflatex example_paper.tex
    ```

3.  **Verify Output**:
    *   Open `example_paper.pdf`.
    *   Check that the document is 8-10 pages long.
    *   Verify that the Bibliography is present and formatted correctly.

## Troubleshooting

*   **Missing .sty files**: Ensure `icml2025.sty` is in the `paper/` directory.
*   **BibTeX errors**: Check `example_paper.blg` for citation keys not found.
