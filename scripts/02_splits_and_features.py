"""Create stratified splits and extract MFCC features - Tasks 8-9"""

import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.splits import create_stratified_splits
from src.features.audio import cache_audio_features

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
SPLITS_DIR = ARTIFACTS / "splits"
SPLITS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ARTIFACTS / "audio_mfcc_cache"

print("=" * 80)
print("TASK 8: CREATE STRATIFIED SPLITS")
print("=" * 80)

# Load filtered datasets
xc_df = pd.read_parquet(ARTIFACTS / "xeno_canto_filtered.parquet")
cub_df = pd.read_parquet(ARTIFACTS / "cub_filtered.parquet")

print(
    f"\nXeno-Canto: {len(xc_df)} recordings, {xc_df['species_normalized'].nunique()} species"
)
print(f"CUB: {len(cub_df)} images, {cub_df['species_normalized'].nunique()} species")

# Filter out species with too few samples (need at least 2 for stratification)
xc_counts = xc_df["species_normalized"].value_counts()
species_to_keep = xc_counts[xc_counts >= 2].index
xc_df = xc_df[xc_df["species_normalized"].isin(species_to_keep)].copy()
print(
    f"After filtering species with <2 samples: {len(xc_df)} recordings, {xc_df['species_normalized'].nunique()} species"
)

cub_counts = cub_df["species_normalized"].value_counts()
species_to_keep = cub_counts[cub_counts >= 2].index
cub_df = cub_df[cub_df["species_normalized"].isin(species_to_keep)].copy()
print(
    f"After filtering species with <2 samples: {len(cub_df)} images, {cub_df['species_normalized'].nunique()} species"
)

# Create audio splits
print("\nCreating Xeno-Canto audio splits (70/15/15)...")
xc_splits = create_stratified_splits(xc_df, "species_normalized", 0.7, 0.15, 0.15, 42)
print(
    f"Train: {len(xc_splits['train'])}, Val: {len(xc_splits['val'])}, Test: {len(xc_splits['test'])}"
)

with open(SPLITS_DIR / "xeno_canto_audio_splits.json", "w") as f:
    json.dump({k: [int(x) for x in v] for k, v in xc_splits.items()}, f)

# Create image splits
print("\nCreating CUB image splits (70/15/15)...")
cub_splits = create_stratified_splits(cub_df, "species_normalized", 0.7, 0.15, 0.15, 42)
print(
    f"Train: {len(cub_splits['train'])}, Val: {len(cub_splits['val'])}, Test: {len(cub_splits['test'])}"
)

with open(SPLITS_DIR / "cub_image_splits.json", "w") as f:
    json.dump({k: [int(x) for x in v] for k, v in cub_splits.items()}, f)

print("\n✓ Saved splits to", SPLITS_DIR)

print("\n" + "=" * 80)
print("TASK 9: EXTRACT MFCC FEATURES")
print("=" * 80)

print("\n⚠️  WARNING: This will take 1-4 hours depending on CPU speed")
print("Features will be cached for reuse. Press Ctrl+C to skip.\n")

import time

time.sleep(3)

print("Extracting MFCC features for Xeno-Canto (n=11,076 files)...")
print("Parameters: 40 MFCC coefficients, 3s duration, 22.05kHz sampling rate")
print("Output: (H, W, 3) stacked static + delta + delta-delta\n")

success_count = cache_audio_features(
    df=xc_df,
    cache_dir=CACHE_DIR / "xeno_canto",
    dataset_name="Xeno-Canto",
    n_mfcc=40,
    hop_length=512,
    n_fft=2048,
    target_sr=22050,
    duration=3.0,
)

print(f"\n✓ Successfully cached {success_count}/{len(xc_df)} audio features")
print(f"✓ Cache location: {CACHE_DIR / 'xeno_canto'}")

print("\n" + "=" * 80)
print("✓ TASKS 8-9 COMPLETE")
print("=" * 80)
