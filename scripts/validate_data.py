#!/usr/bin/env python3
"""Validate data integrity and completeness.

This script checks:
- Dataset indexing (Xeno-Canto, CUB-200)
- Species intersection
- Data splits integrity
- No data leakage
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_xeno_canto():
    """Validate Xeno-Canto audio dataset."""
    print("\n" + "=" * 80)
    print("VALIDATING XENO-CANTO DATASET")
    print("=" * 80)

    artifacts = project_root / "artifacts"
    xc_file = artifacts / "xeno_canto_filtered.parquet"

    if not xc_file.exists():
        print(f"❌ File not found: {xc_file}")
        return False

    df = pd.read_parquet(xc_file)

    # Check columns
    required_cols = ["record_id", "species", "file_path", "species_normalized"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False

    print(f"✓ All required columns present: {required_cols}")

    # Check data
    print(f"✓ Total records: {len(df):,}")
    print(f"✓ Unique species: {df['species_normalized'].nunique()}")
    print(f"✓ Columns: {df.columns.tolist()}")

    # Check for missing values
    missing = df[required_cols].isnull().sum()
    if missing.any():
        print(f"⚠️ Missing values:\n{missing[missing > 0]}")
    else:
        print("✓ No missing values in required columns")

    # Check file paths exist (sample)
    sample_paths = df["file_path"].sample(min(10, len(df)))
    invalid_paths = []
    for path in sample_paths:
        if not Path(path).exists():
            invalid_paths.append(path)

    if invalid_paths:
        print(f"⚠️ Some file paths don't exist (sample):")
        for path in invalid_paths[:3]:
            print(f"  - {path}")
    else:
        print("✓ File paths valid (sampled)")

    return True


def validate_cub():
    """Validate CUB-200 image dataset."""
    print("\n" + "=" * 80)
    print("VALIDATING CUB-200 DATASET")
    print("=" * 80)

    artifacts = project_root / "artifacts"
    cub_file = artifacts / "cub_filtered.parquet"

    if not cub_file.exists():
        print(f"❌ File not found: {cub_file}")
        return False

    df = pd.read_parquet(cub_file)

    # Check columns (use image_id instead of sample_id)
    required_cols = ["image_id", "species_normalized", "file_path"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False

    print(f"✓ All required columns present: {required_cols}")

    # Check data
    print(f"✓ Total images: {len(df):,}")
    print(f"✓ Unique species: {df['species_normalized'].nunique()}")
    print(f"✓ Columns: {df.columns.tolist()}")

    # Check for missing values
    missing = df[required_cols].isnull().sum()
    if missing.any():
        print(f"⚠️ Missing values:\n{missing[missing > 0]}")
    else:
        print("✓ No missing values in required columns")

    # Check file paths exist (sample)
    sample_paths = df["file_path"].sample(min(10, len(df)))
    invalid_paths = []
    for path in sample_paths:
        if not Path(path).exists():
            invalid_paths.append(path)

    if invalid_paths:
        print(f"⚠️ Some file paths don't exist (sample):")
        for path in invalid_paths[:3]:
            print(f"  - {path}")
    else:
        print("✓ File paths valid (sampled)")

    return True


def validate_intersection():
    """Validate species intersection computation."""
    print("\n" + "=" * 80)
    print("VALIDATING SPECIES INTERSECTION")
    print("=" * 80)

    artifacts = project_root / "artifacts"
    meta_file = artifacts / "intersection_metadata.json"

    if not meta_file.exists():
        print(f"❌ File not found: {meta_file}")
        return False

    with open(meta_file, "r") as f:
        meta = json.load(f)

    print("✓ Intersection metadata loaded")
    print(f"✓ Common species: {meta['intersection_count']}")
    print(f"✓ Xeno-Canto species: {meta['xeno_canto_species']}")
    print(f"✓ CUB species: {meta['cub_species']}")

    # Verify intersection
    xc_df = pd.read_parquet(artifacts / "xeno_canto_filtered.parquet")
    cub_df = pd.read_parquet(artifacts / "cub_filtered.parquet")

    xc_species = set(xc_df["species_normalized"].unique())
    cub_species = set(cub_df["species_normalized"].unique())
    common = xc_species & cub_species

    if len(common) == meta["intersection_count"]:
        print(f"✓ Intersection count matches: {len(common)}")
    else:
        print(
            f"⚠️ Intersection mismatch: computed={len(common)}, metadata={meta['intersection_count']}"
        )

    return True


def validate_splits():
    """Validate train/val/test splits."""
    print("\n" + "=" * 80)
    print("VALIDATING DATA SPLITS")
    print("=" * 80)

    artifacts = project_root / "artifacts"
    splits_dir = artifacts / "splits"

    # Audio splits
    audio_splits_file = splits_dir / "xeno_canto_audio_splits.json"
    if not audio_splits_file.exists():
        print(f"❌ Audio splits not found: {audio_splits_file}")
        return False

    with open(audio_splits_file, "r") as f:
        audio_splits = json.load(f)

    print(f"\n✓ Audio splits loaded")
    print(f"  Train: {len(audio_splits['train']):,}")
    print(f"  Val: {len(audio_splits['val']):,}")
    print(f"  Test: {len(audio_splits['test']):,}")
    print(
        f"  Total: {len(audio_splits['train']) + len(audio_splits['val']) + len(audio_splits['test']):,}"
    )

    # Check for overlap (data leakage)
    train_set = set(audio_splits["train"])
    val_set = set(audio_splits["val"])
    test_set = set(audio_splits["test"])

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    if train_val_overlap:
        print(f"❌ Train-Val overlap: {len(train_val_overlap)} samples")
    else:
        print("✓ No Train-Val overlap")

    if train_test_overlap:
        print(f"❌ Train-Test overlap: {len(train_test_overlap)} samples")
    else:
        print("✓ No Train-Test overlap")

    if val_test_overlap:
        print(f"❌ Val-Test overlap: {len(val_test_overlap)} samples")
    else:
        print("✓ No Val-Test overlap")

    # Image splits
    image_splits_file = splits_dir / "cub_image_splits.json"
    if not image_splits_file.exists():
        print(f"⚠️ Image splits not found: {image_splits_file}")
    else:
        with open(image_splits_file, "r") as f:
            image_splits = json.load(f)

        print(f"\n✓ Image splits loaded")
        print(f"  Train: {len(image_splits['train']):,}")
        print(f"  Val: {len(image_splits['val']):,}")
        print(f"  Test: {len(image_splits['test']):,}")

    return not (train_val_overlap or train_test_overlap or val_test_overlap)


def validate_cache():
    """Validate audio feature cache."""
    print("\n" + "=" * 80)
    print("VALIDATING AUDIO FEATURE CACHE")
    print("=" * 80)

    artifacts = project_root / "artifacts"
    cache_dir = artifacts / "audio_mfcc_cache" / "xeno_canto"

    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return False

    # Count cached files
    cached_files = list(cache_dir.glob("*/*.npy"))
    print(f"✓ Cache directory exists: {cache_dir}")
    print(f"✓ Cached feature files: {len(cached_files):,}")

    # Check a sample file
    if cached_files:
        sample_file = cached_files[0]
        try:
            features = np.load(sample_file)
            print(f"✓ Sample feature shape: {features.shape}")

            # Validate shape
            if len(features.shape) != 3:
                print(
                    f"❌ Invalid feature shape: expected 3D, got {len(features.shape)}D"
                )
                return False

            if features.shape[2] != 3:
                print(f"⚠️ Unexpected channel count: {features.shape[2]} (expected 3)")
            else:
                print("✓ Feature channels correct (3: MFCC, Delta, Delta²)")

            # Check data type
            print(f"✓ Feature dtype: {features.dtype}")

        except Exception as e:
            print(f"❌ Error loading sample file: {e}")
            return False
    else:
        print("⚠️ No cached files found")

    return True


def main():
    """Run all validation checks."""
    print("=" * 80)
    print("DATA VALIDATION SUITE")
    print("=" * 80)
    print(f"Project root: {project_root}")

    results = {
        "Xeno-Canto": validate_xeno_canto(),
        "CUB-200": validate_cub(),
        "Intersection": validate_intersection(),
        "Splits": validate_splits(),
        "Cache": validate_cache(),
    }

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status:12} {check}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ All validation checks passed!")
        return 0
    else:
        print("\n⚠️ Some validation checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
