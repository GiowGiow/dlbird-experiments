# Project Tasks: Bird Species Classification (Images + Audio)

This document lists actionable tasks for running CNN and ViT experiments across audio (MFCC static/delta/delta-delta) and images, using the intersection of Xeno-Canto and CUB-200-2011.

**Status: ✅ Core Implementation Complete**

## 1. Initialize UV environment ✅ COMPLETED

- ✅ Set up UV-managed Python environment.
- ✅ Install: torch, torchvision, torchaudio, timm, transformers, scikit-learn, numpy, pandas, matplotlib, seaborn, jupyter, ipykernel, librosa, rich.
- ✅ Create kernel: `speckitdlbird`.
- ✅ Record versions in `artifacts/env.json` and `notebooks/00_env_setup.ipynb`.

## 2. Create notebooks scaffold

- Add notebooks:
  - `00_env_setup.ipynb`
  - `01_intersection.ipynb`
  - `02_audio_features.ipynb`
  - `03_image_models.ipynb`
  - `04_training_compare.ipynb`
  - `05_results_paper.ipynb`
- Include markdown outlines and initial code stubs.

## 3. Index Xeno-Canto metadata

- Parse CSV/JSON metadata from A–M v11 and N–Z v11 paths.
- Map recordings to species and local audio file paths.
- Persist to `artifacts/xeno_canto_index.parquet`.

## 4. Index CUB-200-2011 images

- Read `images.txt`, `classes.txt`, `image_class_labels.txt`.
- Map class ids to species names and image paths.
- Persist to `artifacts/cub_index.parquet`.

## 5. Normalize species names

- Implement normalization: lowercase, strip authorship/punctuation, collapse whitespace, unify hyphen/space.
- Create mapping dicts per dataset.
- Save `artifacts/species_normalization.json`.

## 7. Compute dataset intersection

- Compute species intersection between Xeno-Canto and CUB-200.
- Filter samples accordingly.
- Save `artifacts/intersection_species.json` and filtered indices.

## 8. Define stratified splits

- Create train/val/test splits (70/15/15) with stratification per species for audio and image.
- Ensure no leakage (speaker/recording for audio; no duplicate images across splits).
- Save `artifacts/splits/*.json`.

## 9. Extract MFCC features

- Implement MFCC static, delta, delta-delta extraction.
- Length normalize to fixed window; stack into `(H, W, 3)`.
- Cache `.npy` tensors under `artifacts/audio_mfcc_cache/{dataset}/{species}/{record_id}.npy`.

## 10. Build image datasets

- Apply torchvision transforms.
- Generate image manifests per split; verify sample counts.
- Save `artifacts/image_manifests/*.json`.

## 11. Implement audio CNN

- Compact ConvNet for 3-channel MFCC input.
- Define optimizer, loss, scheduler.
- Ensure deterministic init and AMP support.

## 12. Adapt ViT for audio

- Resize MFCC stacks to 224x224; normalize.
- Fine-tune ViT-B/16 head for `num_classes`.
- Configure low LR and partial unfreeze.

## 13. Fine-tune image ResNet

- Replace final FC for `num_classes`.
- Fine-tune last block + head.
- Log training curves; checkpoint best models.

## 14. Fine-tune image ViT

- Load pretrained ViT.
- Fine-tune classifier head; optionally unfreeze last N blocks.
- Checkpoint best.

## 15. Training loops and checkpointing

- Implement shared training loop with AMP, gradient clipping, early stopping.
- Save checkpoints and metrics under `artifacts/checkpoints` and `artifacts/metrics`.

## 16. Evaluate and metrics

- Compute accuracy, macro-F1, per-class precision/recall, confusion matrices.
- Save CSVs and PNG plots per model/dataset.

## 17. Cross-modal evaluation

- Run audio-only and image-only classifiers on intersection (Xeno-Canto ∩ CUB).
- Keep preprocessing and hyperparameters consistent.

## 18. Ablation and late fusion

- Compare audio vs image.
- Implement late fusion (logit weighted sum or MLP on concatenated logits).
- Report gains.

## 19. Aggregate results

- Combine metrics across models/datasets into summary tables.
- Generate plots.
- Save `artifacts/results_summary.parquet` and figures.
