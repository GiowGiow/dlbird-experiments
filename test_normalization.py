#!/usr/bin/env python
"""
Test script for Phase 1 - Feature Normalization Validation
Tests: T026, T027, T028

Validates that MFCC features are properly normalized with:
- MFCC channel: mean ≈ 0, std ≈ 1
- Delta channel: mean ≈ 0, std ≈ 1
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.audio import AudioMFCCDataset
from torch.utils.data import DataLoader
import pandas as pd

# Paths
ARTIFACTS_DIR = Path("artifacts")
SPLITS_FILE = ARTIFACTS_DIR / "splits" / "xeno_canto_audio_splits.json"
CACHE_DIR = ARTIFACTS_DIR / "audio_mfcc_cache" / "xeno_canto"

def test_normalization():
    """Test that normalized features have mean≈0, std≈1"""
    print("=" * 80)
    print("PHASE 1 - FEATURE NORMALIZATION VALIDATION")
    print("=" * 80)
    
    # Load data (same approach as training script)
    print("\n1. Loading dataset with normalization enabled...")
    xc_df = pd.read_parquet(ARTIFACTS_DIR / "xeno_canto_filtered.parquet")
    
    # Filter to species with >=2 samples
    xc_counts = xc_df["species_normalized"].value_counts()
    species_to_keep = xc_counts[xc_counts >= 2].index
    xc_df = xc_df[xc_df["species_normalized"].isin(species_to_keep)].copy()
    
    species_list = sorted(xc_df['species_normalized'].unique())
    species_to_idx = {sp: idx for idx, sp in enumerate(species_list)}
    
    # Load splits
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    
    # Create dataset with normalization
    train_dataset = AudioMFCCDataset(
        df=xc_df,
        cache_dir=CACHE_DIR,
        indices=splits["train"],
        species_to_idx=species_to_idx,
        normalize=True
    )
    
    print(f"   Dataset size: {len(train_dataset)} samples")
    print(f"   Normalization: ENABLED")
    
    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Collect statistics from first 1000 samples
    print("\n2. Computing statistics on 1000 samples...")
    mfcc_stats = []
    delta_stats = []
    
    sample_count = 0
    for features, _ in train_loader:
        # features shape: (batch, 3, time, mfcc_dim)
        # Channel 0: MFCC
        # Channel 1: Delta
        # Channel 2: Delta²
        
        mfcc_channel = features[:, 0, :, :].numpy()  # (batch, time, dim)
        delta_channel = features[:, 1, :, :].numpy()
        
        mfcc_stats.append({
            'mean': mfcc_channel.mean(),
            'std': mfcc_channel.std()
        })
        delta_stats.append({
            'mean': delta_channel.mean(),
            'std': delta_channel.std()
        })
        
        sample_count += features.shape[0]
        if sample_count >= 1000:
            break
    
    # Compute overall statistics
    mfcc_mean = np.mean([s['mean'] for s in mfcc_stats])
    mfcc_std = np.mean([s['std'] for s in mfcc_stats])
    delta_mean = np.mean([s['mean'] for s in delta_stats])
    delta_std = np.mean([s['std'] for s in delta_stats])
    
    print("\n3. Normalization Statistics:")
    print(f"   MFCC Channel:")
    print(f"      Mean: {mfcc_mean:.4f} (target: ~0.00)")
    print(f"      Std:  {mfcc_std:.4f} (target: ~1.00)")
    
    print(f"\n   Delta Channel:")
    print(f"      Mean: {delta_mean:.4f} (target: ~0.00)")
    print(f"      Std:  {delta_std:.4f} (target: ~1.00)")
    
    # Validation
    print("\n4. Validation Results:")
    mfcc_mean_ok = abs(mfcc_mean) < 0.1
    mfcc_std_ok = 0.8 < mfcc_std < 1.2
    delta_mean_ok = abs(delta_mean) < 0.1
    delta_std_ok = 0.8 < delta_std < 1.2
    
    print(f"   MFCC mean ≈ 0:   {'✓ PASS' if mfcc_mean_ok else '✗ FAIL'}")
    print(f"   MFCC std ≈ 1:    {'✓ PASS' if mfcc_std_ok else '✗ FAIL'}")
    print(f"   Delta mean ≈ 0:  {'✓ PASS' if delta_mean_ok else '✗ FAIL'}")
    print(f"   Delta std ≈ 1:   {'✓ PASS' if delta_std_ok else '✗ FAIL'}")
    
    all_pass = mfcc_mean_ok and mfcc_std_ok and delta_mean_ok and delta_std_ok
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ NORMALIZATION VALIDATION PASSED")
        print("Features are properly normalized with mean≈0, std≈1")
    else:
        print("✗ NORMALIZATION VALIDATION FAILED")
        print("Some features do not meet normalization criteria")
    print("=" * 80)
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    exit_code = test_normalization()
    sys.exit(exit_code)
