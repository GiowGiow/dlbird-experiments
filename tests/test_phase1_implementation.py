"""Test Phase 1 implementation - T007, T008, T027"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Import modified components
# Import load_class_weights from the training script
import importlib.util

script_path = Path(__file__).parent.parent / "scripts/03_train_audio.py"
spec = importlib.util.spec_from_file_location(
    "train_audio", str(script_path)
)
train_audio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_audio)
load_class_weights = train_audio.load_class_weights

from src.datasets.audio import AudioMFCCDataset

print("=" * 80)
print("PHASE 1 IMPLEMENTATION VALIDATION")
print("=" * 80)

# Test T007/T008: load_class_weights function
print("\n[T007-T008] Testing load_class_weights()...")

# Load species list
ARTIFACTS = Path("artifacts")
xc_df = pd.read_parquet(ARTIFACTS / "xeno_canto_filtered.parquet")
xc_counts = xc_df["species_normalized"].value_counts()
species_to_keep = xc_counts[xc_counts >= 2].index
xc_df = xc_df[xc_df["species_normalized"].isin(species_to_keep)].copy()
species_list = sorted(xc_df["species_normalized"].unique())

# Test load function
weights = load_class_weights(species_list, method="balanced")

print(f"✅ Function executed successfully")
print(f"✅ Weights shape: {weights.shape} (expected: (90,))")
assert weights.shape[0] == 90, f"Expected 90 species, got {weights.shape[0]}"

print(f"✅ Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
print(f"✅ Weight statistics: mean={weights.mean():.4f}, std={weights.std():.4f}")
assert weights.min() > 0, "Weights should be positive"
assert weights.max() < 100, "Weights seem unreasonably large"
assert torch.all(torch.isfinite(weights)), "Weights contain NaN or inf"

print("✅ All class weights tests passed!")

# Test T027: Normalization validation
print("\n[T027] Testing feature normalization...")

with open(ARTIFACTS / "splits" / "xeno_canto_audio_splits.json", "r") as f:
    splits = json.load(f)

species_to_idx = {sp: i for i, sp in enumerate(species_list)}
cache_dir = ARTIFACTS / "audio_mfcc_cache" / "xeno_canto"

# Create dataset with normalization
dataset = AudioMFCCDataset(
    df=xc_df,
    cache_dir=cache_dir,
    indices=splits["train"][:1000],  # Sample 1000 for speed
    species_to_idx=species_to_idx,
    transform=None,
    normalize=True,
)

# Load a batch
loader = DataLoader(dataset, batch_size=32, shuffle=False)
batch_features, batch_labels = next(iter(loader))

print(f"✅ Batch shape: {batch_features.shape}")

# Check channel-wise statistics
mfcc_mean = batch_features[:, 0].mean().item()
mfcc_std = batch_features[:, 0].std().item()
delta_mean = batch_features[:, 1].mean().item()
delta_std = batch_features[:, 1].std().item()

print(f"\nNormalized statistics:")
print(f"  MFCC  - mean: {mfcc_mean:+.4f}, std: {mfcc_std:.4f}")
print(f"  Delta - mean: {delta_mean:+.4f}, std: {delta_std:.4f}")

# Validation criteria (allow some variance due to batch sampling)
assert abs(mfcc_mean) < 0.2, f"MFCC mean should be near 0, got {mfcc_mean}"
assert 0.8 < mfcc_std < 1.2, f"MFCC std should be near 1, got {mfcc_std}"
assert abs(delta_mean) < 0.2, f"Delta mean should be near 0, got {delta_mean}"
assert 0.8 < delta_std < 1.2, f"Delta std should be near 1, got {delta_std}"

print("✅ Normalization statistics are within expected range!")
print("✅ Features normalized to mean≈0, std≈1")

# Test without normalization for comparison
dataset_no_norm = AudioMFCCDataset(
    df=xc_df,
    cache_dir=cache_dir,
    indices=splits["train"][:100],
    species_to_idx=species_to_idx,
    transform=None,
    normalize=False,
)

loader_no_norm = DataLoader(dataset_no_norm, batch_size=32, shuffle=False)
batch_no_norm, _ = next(iter(loader_no_norm))

print(f"\nWithout normalization (for comparison):")
print(
    f"  MFCC  - mean: {batch_no_norm[:, 0].mean().item():+.4f}, std: {batch_no_norm[:, 0].std().item():.4f}"
)
print(
    f"  Delta - mean: {batch_no_norm[:, 1].mean().item():+.4f}, std: {batch_no_norm[:, 1].std().item():.4f}"
)

print("\n" + "=" * 80)
print("✅ ALL VALIDATION TESTS PASSED")
print("=" * 80)
print("\nPhase 1 implementation is ready for training!")
print("\nNext steps:")
print(
    "  1. Run 1-epoch smoke test: python scripts/03_train_audio.py --model AudioCNN --use-class-weights --epochs 1 --batch-size 32"
)
print("  2. If successful, run full training for 50 epochs")
