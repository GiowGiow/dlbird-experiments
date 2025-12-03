# Quick Start Guide

## SpeckitDLBird - Bird Species Classification

### Prerequisites

- Python 3.12
- UV package manager
- CUDA-capable GPU (recommended)
- External drive with datasets mounted

### 1. Verify Installation

```bash
# Check if environment is set up
uv run python test_implementation.py
```

Expected output: All tests should pass ✓

### 2. Verify Dataset Access

```bash
# Check if datasets are accessible
uv run python verify_datasets.py
```

If datasets are not accessible:
- Mount your external drive
- Update paths in notebooks to match your setup
- Run verify again

### 3. Run Experiments

Launch Jupyter:
```bash
jupyter notebook notebooks/
```

Select kernel: **Python (speckitdlbird)**

Run notebooks in order:
1. `00_env_setup.ipynb` - Verify environment
2. `01_intersection.ipynb` - Index datasets and compute intersection
3. `02_audio_features.ipynb` - Extract and cache MFCC features
4. `03_image_models.ipynb` - Prepare image datasets
5. `04_training_compare.ipynb` - Train all models
6. `05_results_paper.ipynb` - Generate results and paper

### 4. Monitor Progress

Artifacts are saved to:
- `artifacts/checkpoints/` - Model checkpoints
- `artifacts/metrics/` - Evaluation metrics
- `artifacts/splits/` - Train/val/test splits
- `artifacts/audio_mfcc_cache/` - Cached audio features

### 5. Access Results

After running all notebooks:
- Training curves: `paper/figures/training_curves.pdf`
- Confusion matrices: `paper/figures/confusion_*.pdf`
- Results table: `paper/tables/results_summary.tex`
- Final paper: `paper/output/icml2025_bird_classification.pdf`

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in training config
- Use smaller models first (CNN before ViT)
- Clear cache between runs

### Dataset Not Found
- Check external drive is mounted
- Update paths in notebook cells
- Verify with `verify_datasets.py`

### Slow Feature Extraction
- Audio MFCC extraction is CPU-intensive
- Features are cached for reuse
- First run will be slow, subsequent runs fast

### Import Errors
- Ensure kernel is `Python (speckitdlbird)`
- Restart kernel if needed
- Check `uv sync` completed successfully

## Key Commands

```bash
# Run tests
uv run python test_implementation.py

# Verify datasets
uv run python verify_datasets.py

# Start Jupyter
jupyter notebook notebooks/

# Train a specific model (from notebook)
# See 04_training_compare.ipynb for examples

# Generate paper
# Run 05_results_paper.ipynb
```

## Expected Timeline

- Environment setup: 5-10 minutes
- Dataset indexing: 10-30 minutes
- MFCC extraction: 1-4 hours (depends on dataset size)
- Model training: 2-6 hours per model (GPU)
- Results generation: 10-20 minutes

## Support

For detailed information:
- See `README.md` for full documentation
- See `IMPLEMENTATION_SUMMARY.md` for technical details
- Check inline documentation in source files
- Review notebook markdown cells for guidance

## Success Indicators

✓ All tests pass
✓ Datasets accessible
✓ Features cached successfully
✓ Models train without errors
✓ Metrics saved properly
✓ Paper compiles successfully

---

**Ready to start? Run the verification scripts first!**
