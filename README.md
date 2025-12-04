# SpeckitDLBird: Bird Species Classification

A comprehensive bird species classification system using both images and audio (MFCC features) with CNN and Vision Transformer architectures.

## Project Structure

```
speckitdlbird/
├── src/                    # Source code
│   ├── data/               # Dataset indexing modules
│   │   ├── xeno_canto.py   # Xeno-Canto audio indexing
│   │   ├── cub.py          # CUB-200-2011 image indexing
│   ├── features/           # Feature extraction
│   │   └── audio.py        # MFCC extraction and caching
│   ├── datasets/           # PyTorch datasets
│   │   ├── audio.py        # Audio MFCC dataset
│   │   └── image.py        # Image dataset
│   ├── models/             # Model architectures
│   │   ├── audio_cnn.py    # Compact CNN for audio
│   │   ├── audio_vit.py    # ViT for audio
│   │   ├── image_resnet.py # ResNet for images
│   │   └── image_vit.py    # ViT for images
│   ├── training/           # Training utilities
│   │   └── trainer.py      # Unified trainer with AMP
│   ├── evaluation/         # Evaluation utilities
│   │   ├── metrics.py      # Metrics computation
│   │   └── aggregate.py    # Results aggregation
│   └── utils/              # Utility functions
│       ├── species.py      # Species name normalization
│       └── splits.py       # Stratified split generation
├── notebooks/              # Jupyter notebooks
│   ├── 00_env_setup.ipynb
│   ├── 01_intersection.ipynb
│   ├── 02_audio_features.ipynb
│   ├── 03_image_models.ipynb
│   ├── 04_training_compare.ipynb
│   └── 05_results_paper.ipynb
├── artifacts/              # Generated artifacts
│   ├── checkpoints/        # Model checkpoints
│   ├── splits/             # Train/val/test splits
│   ├── audio_mfcc_cache/   # Cached MFCC features
│   ├── metrics/            # Evaluation metrics
│   └── env.json            # Environment versions
├── data/                   # Working data directory
├── paper/                  # LaTeX paper
│   ├── figures/            # Publication figures
│   ├── tables/             # LaTeX tables
│   └── output/             # Compiled PDF
└── README.md

## Setup

### 1. Environment Setup

This project uses UV for package management. Install UV if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Initialize the project and install dependencies:

```bash
cd speckitdlbird
uv sync
```

Create Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name speckitdlbird --display-name "Python (speckitdlbird)"
```

### 2. Dataset Paths

Update the dataset paths in notebooks according to your local setup:

- **Xeno-Canto A-M v11**: `/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m/versions/11`
- **Xeno-Canto N-Z v11**: `/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/rohanrao/xeno-canto-bird-recordings-extended-n-z/versions/11`
- **CUB-200-2011 v7**: `/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/wenewone/cub2002011/versions/7`
- **SSW60 tarball**: `/media/giovanni/TOSHIBA EXT/dlbird/datasets/mixed/ssw60.tar.gz`

## Workflow

### 1. Environment Verification

Run `00_env_setup.ipynb` to verify installation and save environment info.

### 2. Dataset Indexing and Intersection

Run `01_intersection.ipynb` to:
- Index Xeno-Canto audio metadata
- Index CUB-200-2011 image metadata
- Extract and index SSW60 dataset
- Normalize species names
- Compute species intersection
- Filter datasets to intersection species

### 3. Audio Feature Extraction

Run `02_audio_features.ipynb` to:
- Extract MFCC static, delta, and delta-delta features
- Cache features as numpy arrays
- Create stratified train/val/test splits

### 4. Image Dataset Preparation

Run `03_image_models.ipynb` to:
- Create stratified splits for images
- Generate image manifests
- Set up preprocessing transforms

### 5. Model Training

Run `04_training_compare.ipynb` to:
- Train Audio CNN on MFCC features
- Train Audio ViT on MFCC features
- Train Image ResNet on CUB/SSW60
- Train Image ViT on CUB/SSW60
- Evaluate all models and save metrics

### 6. Results and Paper Generation

Run `05_results_paper.ipynb` to:
- Aggregate metrics from all experiments
- Generate comparison tables and figures
- Fill LaTeX template with results
- Compile ICML 2025 short paper

## Models

### Audio Models

1. **Audio CNN**: Compact convolutional network (~1-5M parameters)
   - Input: (3, H, W) MFCC stack
   - Architecture: 3 conv blocks + adaptive pooling + FC layers
   - Optimizer: Adam (lr=3e-4)

2. **Audio ViT**: Vision Transformer adapted for audio
   - Input: MFCC resized to 224x224
   - Pretrained: ViT-B/16 from Google
   - Fine-tuning: Classification head + last blocks
   - Optimizer: Adam (lr=3e-5)

### Image Models

1. **Image ResNet**: ResNet-18/50 for images
   - Pretrained on ImageNet
   - Fine-tune last block + FC head
   - Optimizer: Adam (lr=3e-4)

2. **Image ViT**: Vision Transformer for images
   - Pretrained: ViT-B/16 from Google
   - Fine-tune classifier + last N blocks
   - Optimizer: Adam (lr=3e-5)

## Reproducibility

- Fixed random seeds (42) for all experiments
- Deterministic CUDA operations enabled
- All dependencies locked with UV
- Environment versions saved in `artifacts/env.json`
- Stratified splits ensure balanced class distribution
- AMP for consistent training speed

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score (Macro)**: Unweighted mean F1 across classes
- **F1 Score (Weighted)**: Sample-weighted mean F1
- **Per-class Precision/Recall/F1**: Detailed per-species metrics
- **Confusion Matrix**: Full confusion matrix visualization

## Paper

The project generates a 4-page ICML 2025 short paper with:
- Methodology description
- Results tables and figures
- Ablation studies (audio vs image vs fusion)
- Ethics statement and reproducibility checklist

Compile the paper:

```bash
cd paper
pdflatex icml2025_bird_classification.tex
```

## License

This project follows the licenses of the constituent datasets:
- Xeno-Canto: CC-BY-NC-SA 4.0
- CUB-200-2011: Research use
- SSW60: Check source repository

## Citation

```bibtex
@misc{speckitdlbird2025,
  title={Bird Species Classification using Images and Audio with CNNs and Vision Transformers},
  author={[Your Name]},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].
