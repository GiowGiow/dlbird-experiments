# Technical Implementation Plan: Bird Species Classification (Images + Audio)

This plan follows the chosen stack: Python, UV for package management, PyTorch ecosystem (torch/torchvision/torchaudio), transformers/timm for ViT, librosa for audio features, scikit-learn for metrics, and Jupyter notebooks for experiments. It implements CNN and ViT models for both audio (MFCC stacks) and images, using the intersection of Xeno-Canto and CUB-200-2011 datasets.

## Contract

- Inputs:
  - Dataset roots (paths provided below) and metadata files.
  - Configs for preprocessing and training (JSON/YAML or inline dicts).

- Outputs:
  - Artifacts: cached MFCC tensors, preprocessed image datasets, checkpoints, logs.
  - Metrics: accuracy, macro-F1, per-class stats, confusion matrices, plots.
  - Results analysis and summaries.

- Error modes:
  - Missing files/metadata, species mismatches, invalid audio/image files, CUDA unavailability.

- Success criteria:
  - Reproducible runs (fixed seeds), valid comparisons across datasets and modalities, clear metrics tables and figures.

## Dataset Paths

- Xeno-Canto part 1 (A–M v11): `/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m/versions/11`
- Xeno-Canto part 2 (N–Z v11): `/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/rohanrao/xeno-canto-bird-recordings-extended-n-z/versions/11`
- CUB-200-2011 v7: `/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/wenewone/cub2002011/versions/7`
- Working directories: `data/`, `artifacts/`

## UV Environment

- UV commands (run in terminal):
  - `uv init`
  - `uv add torch torchvision torchaudio timm transformers scikit-learn numpy pandas matplotlib seaborn jupyter ipykernel librosa rich`
  - `uv lock`
  - `python -m ipykernel install --user --name speckitdlbird --display-name "Python (speckitdlbird)"`

- Record exact versions in `notebooks/00_env_setup.ipynb` and `artifacts/env.json`.

## Notebooks Overview

- `00_env_setup.ipynb`: environment checks, seeds, versions.
- `01_intersection.ipynb`: index/normalize species and compute Xeno-Canto ∩ CUB intersection.
- `02_audio_features.ipynb`: MFCC static/delta/delta-delta extraction and caching.
- `03_image_models.ipynb`: image preprocessing and dataset construction.
- `04_training_compare.ipynb`: train CNN/ViT on audio and images; checkpoint and evaluate.
- `05_results_paper.ipynb`: aggregate results and generate analysis.

## Data Processing

### Metadata ingestion

- Xeno-Canto v11: read metadata CSV/JSON (fields: species/common/scientific name, file path, sampling rate, duration). Map to local audio files.
- CUB-200-2011: read `images.txt`, `classes.txt`, `image_class_labels.txt`. Map class id -> species name; build absolute image paths.

### Species normalization

- Normalize names: lowercase; remove authorship strings; strip punctuation; collapse whitespace; unify hyphen/space.
- Build mapping dict: {normalized_name -> canonical/scientific name} per dataset.
- Compute intersection between Xeno-Canto and CUB-200 species; keep only species present in both for intersection experiments.

### Splits

- For each dataset/modality:
  - Stratified train/val/test splits per species (e.g., 70/15/15). Ensure no speaker/recording leakage for audio; no image duplication across splits.
  - Persist splits to `artifacts/splits/*.json`.

## Audio Feature Extraction

- Load audio with torchaudio or librosa at target sample rate (e.g., 22.05 kHz).
- Compute MFCCs: `librosa.feature.mfcc` or `torchaudio.transforms.MFCC`.
- Compute Delta and Delta-Delta: `librosa.feature.delta` (order=1,2) or `torchaudio.functional.compute_deltas`.
- Stack channels: `(H, W, 3)` = [MFCC, Delta, Delta-Delta].
- Length normalization: center-crop or pad to fixed time (e.g., 3s window). Optionally log-mel + MFCC comparison.
- Save numpy arrays (float32) under `artifacts/audio_mfcc_cache/{dataset}/{species}/{record_id}.npy`.

## Image Preprocessing

- Use torchvision transforms:
  - Train: RandomResizedCrop(224), HorizontalFlip, ColorJitter(light), Normalize(ImageNet stats).
  - Val/Test: Resize(256), CenterCrop(224), Normalize(ImageNet stats).
- Persist a manifest for each image sample: path, species, split.

## PyTorch DataModules

- Datasets:
  - AudioMFCCDataset: loads cached `(H,W,3)` tensors, returns `torch.Tensor` shaped `(3,H,W)`.
  - ImageDataset: wraps image paths with transforms.

- DataLoaders: batch sizes tuned per modality (audio 32, image 64 as starting points); num_workers with deterministic seeds; collate handles variable lengths (but we normalize lengths upstream).

## Models

### CNN for Audio

- Architecture: small ConvNet for 3-channel input.
  - Example: [Conv(32,3x3)-BN-ReLU] x2 -> MaxPool -> [Conv(64)] x2 -> MaxPool -> [Conv(128)] x2 -> AdaptivePool -> FC(256) -> Dropout -> FC(num_classes).
  - Params: ~1–5M to keep training fast.
  - Loss: CrossEntropyLoss.
  - Optimizer: Adam (lr=3e-4), weight decay=1e-4.
  - Scheduler: CosineAnnealingLR or StepLR.

### ViT for Audio

- Approach: treat `(H,W,3)` MFCC stack as image input.
- Model: ViT-B/16 (transformers or timm) with 224x224 input; upsample MFCC to 224x224; normalize channels.
- Fine-tuning: unfreeze final layers, replace head with `num_classes` linear; consider lower lr (1e-5 to 3e-5).

### CNN for Images

- Model: torchvision ResNet-18 or 50 pretrained; replace final FC.
- Strategy: fine-tune last block + FC for speed; optionally full fine-tune if resources permit.

### ViT for Images

- Model: ViT-B/16 pretrained; fine-tune classification head; optionally unfreeze last N blocks.

## Training

- Common:
  - Fixed seeds; AMP (`torch.cuda.amp`); gradient clipping; checkpoint best val accuracy; early stopping.
  - Logging: tqdm progress; save `metrics.json`; `events.csv`; plots.

- Hyperparameters (initial):
  - Epochs: 20–30; Batch sizes: audio 32, image 64; LR: CNN 3e-4, ViT 3e-5; Weight decay 1e-4.
  - Augmentations: as above.

- Checkpoints under `artifacts/checkpoints/{modality}_{model}/{dataset}/`.

## Evaluation

- Metrics: accuracy, macro-F1; per-class precision/recall; confusion matrix.
- Optional: ROC-AUC one-vs-rest for image models if probabilities available.
- Save per-class metrics CSV and confusion matrix PNG.

## Cross-Modal Experiments

- Intersection (Xeno-Canto ∩ CUB-200): run audio-only and image-only classifiers.
- Ensure comparable preprocessing and hyperparameters.

## Ablations

- Audio vs Image vs Late Fusion (logits weighted sum or simple MLP on concatenated heads).
- Input resolution effects; augmentation on/off.

## Results Aggregation and Analysis

- Aggregate metrics across models/datasets into a single DataFrame and export tables.
- Generate figures (training curves, confusion matrices) programmatically.
- Analyze model performance and generate comprehensive summaries.

## Risks and Mitigations

- Class imbalance: use weighted loss or balanced sampling.
- Audio quality variability: clip-level selection based on SNR/duration; robust preprocessing.
- Compute limits: start with smaller models/resolutions; use mixed precision.

## Milestones

- M1: Species intersection and dataset indexing.
- M2: MFCC cache and image manifests.
- M3: Baseline CNN/ResNet runs.
- M4: ViT runs for audio and images.
- M5: Results aggregation and analysis.
