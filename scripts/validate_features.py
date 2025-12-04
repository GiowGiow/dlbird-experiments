#!/usr/bin/env python3
"""Validate audio feature extraction and statistics.

This script checks:
- Feature shape consistency
- Feature statistics (mean, std, range)
- Feature normalization
- Feature quality visualization
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_feature_shapes():
    """Check consistency of cached feature shapes."""
    print("\n" + "=" * 80)
    print("CHECKING FEATURE SHAPES")
    print("=" * 80)

    cache_dir = project_root / "artifacts" / "audio_mfcc_cache" / "xeno_canto"

    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return False

    cached_files = list(cache_dir.glob("*/*.npy"))

    if not cached_files:
        print("❌ No cached files found")
        return False

    print(f"✓ Found {len(cached_files):,} cached feature files")

    # Check sample of files
    sample_size = min(100, len(cached_files))
    sample_files = np.random.choice(cached_files, sample_size, replace=False)

    shapes = []
    for file_path in sample_files:
        try:
            features = np.load(file_path)
            shapes.append(features.shape)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return False

    # Check consistency
    unique_shapes = set(shapes)

    if len(unique_shapes) == 1:
        shape = shapes[0]
        print(f"✓ All features have consistent shape: {shape}")

        # Validate expected shape
        expected_mfcc = 40
        expected_channels = 3

        if shape[0] == expected_mfcc:
            print(f"✓ MFCC coefficients correct: {shape[0]}")
        else:
            print(
                f"⚠️ MFCC coefficients unexpected: {shape[0]} (expected {expected_mfcc})"
            )

        if shape[2] == expected_channels:
            print(f"✓ Channels correct: {shape[2]} (MFCC, Delta, Delta²)")
        else:
            print(f"⚠️ Channels unexpected: {shape[2]} (expected {expected_channels})")

        # Calculate time frames
        time_frames = shape[1]
        duration_estimate = (time_frames * 512) / 22050  # hop_length / sr
        print(f"✓ Time frames: {time_frames} (~{duration_estimate:.2f} seconds)")

    else:
        print(f"⚠️ Inconsistent shapes found: {unique_shapes}")
        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"  {shape}: {count} files")

    return len(unique_shapes) == 1


def check_feature_statistics():
    """Analyze feature statistics across dataset."""
    print("\n" + "=" * 80)
    print("ANALYZING FEATURE STATISTICS")
    print("=" * 80)

    cache_dir = project_root / "artifacts" / "audio_mfcc_cache" / "xeno_canto"
    cached_files = list(cache_dir.glob("*/*.npy"))

    # Load sample of features
    sample_size = min(200, len(cached_files))
    sample_files = np.random.choice(cached_files, sample_size, replace=False)

    print(f"✓ Loading {sample_size} samples for statistical analysis...")

    mfcc_values = []
    delta_values = []
    delta2_values = []

    for file_path in sample_files:
        try:
            features = np.load(file_path)
            mfcc_values.append(features[:, :, 0].flatten())
            delta_values.append(features[:, :, 1].flatten())
            delta2_values.append(features[:, :, 2].flatten())
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            continue

    # Concatenate
    mfcc_all = np.concatenate(mfcc_values)
    delta_all = np.concatenate(delta_values)
    delta2_all = np.concatenate(delta2_values)

    # Compute statistics
    print("\n📊 Feature Statistics:")
    print("-" * 80)
    print(f"{'Channel':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)

    channels = [
        ("MFCC", mfcc_all),
        ("Delta", delta_all),
        ("Delta²", delta2_all),
    ]

    for name, values in channels:
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        print(
            f"{name:<15} {mean:<12.4f} {std:<12.4f} {min_val:<12.4f} {max_val:<12.4f}"
        )

    # Check if normalized
    print("\n🔍 Normalization Check:")
    for name, values in channels:
        mean = np.mean(values)
        std = np.std(values)

        if abs(mean) < 0.1 and abs(std - 1.0) < 0.1:
            print(f"  ✓ {name}: Appears normalized (mean≈0, std≈1)")
        else:
            print(f"  ⚠️ {name}: NOT normalized (mean={mean:.4f}, std={std:.4f})")

    return mfcc_all, delta_all, delta2_all


def visualize_feature_distributions(mfcc_all, delta_all, delta2_all):
    """Create visualizations of feature distributions."""
    print("\n" + "=" * 80)
    print("CREATING FEATURE DISTRIBUTION VISUALIZATIONS")
    print("=" * 80)

    output_dir = project_root / "artifacts" / "validation"
    output_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    channels = [
        ("MFCC", mfcc_all, axes[0]),
        ("Delta", delta_all, axes[1]),
        ("Delta²", delta2_all, axes[2]),
    ]

    for name, values, ax in channels:
        # Sample for faster plotting
        sample = np.random.choice(values, min(10000, len(values)), replace=False)

        ax.hist(sample, bins=100, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title(f"{name} Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Zero")
        ax.legend()

        # Add statistics text
        mean = np.mean(values)
        std = np.std(values)
        ax.text(
            0.02,
            0.98,
            f"Mean: {mean:.3f}\nStd: {std:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    output_file = output_dir / "feature_distributions.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved feature distributions to: {output_file}")

    plt.close()


def visualize_sample_features():
    """Visualize sample MFCC features."""
    print("\n" + "=" * 80)
    print("VISUALIZING SAMPLE FEATURES")
    print("=" * 80)

    cache_dir = project_root / "artifacts" / "audio_mfcc_cache" / "xeno_canto"
    cached_files = list(cache_dir.glob("*/*.npy"))

    # Select random samples from different species
    sample_files = np.random.choice(
        cached_files, min(3, len(cached_files)), replace=False
    )

    output_dir = project_root / "artifacts" / "validation"
    output_dir.mkdir(exist_ok=True)

    for idx, file_path in enumerate(sample_files):
        features = np.load(file_path)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        channel_names = ["MFCC", "Delta", "Delta²"]

        for ch_idx, (ax, name) in enumerate(zip(axes, channel_names)):
            im = ax.imshow(
                features[:, :, ch_idx], aspect="auto", cmap="viridis", origin="lower"
            )
            ax.set_title(f"{name} - {file_path.parent.name}")
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Coefficients")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()

        output_file = output_dir / f"sample_features_{idx}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"✓ Saved sample visualization to: {output_file}")

        plt.close()


def check_feature_quality():
    """Check for potential quality issues in features."""
    print("\n" + "=" * 80)
    print("CHECKING FEATURE QUALITY")
    print("=" * 80)

    cache_dir = project_root / "artifacts" / "audio_mfcc_cache" / "xeno_canto"
    cached_files = list(cache_dir.glob("*/*.npy"))

    sample_size = min(100, len(cached_files))
    sample_files = np.random.choice(cached_files, sample_size, replace=False)

    issues = {
        "all_zeros": 0,
        "all_same": 0,
        "extreme_values": 0,
        "nan_values": 0,
        "inf_values": 0,
    }

    for file_path in sample_files:
        features = np.load(file_path)

        # Check for all zeros
        if np.all(features == 0):
            issues["all_zeros"] += 1

        # Check for all same values
        if np.all(features == features[0, 0, 0]):
            issues["all_same"] += 1

        # Check for extreme values (>1000 or <-1000)
        if np.any(np.abs(features) > 1000):
            issues["extreme_values"] += 1

        # Check for NaN
        if np.any(np.isnan(features)):
            issues["nan_values"] += 1

        # Check for Inf
        if np.any(np.isinf(features)):
            issues["inf_values"] += 1

    print("\n🔍 Quality Issues Found:")
    print("-" * 80)

    total_issues = sum(issues.values())

    for issue, count in issues.items():
        if count > 0:
            print(f"  ⚠️ {issue}: {count}/{sample_size} files")
        else:
            print(f"  ✓ {issue}: None")

    if total_issues == 0:
        print("\n✓ No quality issues detected!")
    else:
        print(
            f"\n⚠️ Found {total_issues} potential quality issues in {sample_size} samples"
        )

    return total_issues == 0


def main():
    """Run all feature validation checks."""
    print("=" * 80)
    print("FEATURE VALIDATION SUITE")
    print("=" * 80)
    print(f"Project root: {project_root}")

    # Run checks
    shape_ok = check_feature_shapes()
    mfcc_all, delta_all, delta2_all = check_feature_statistics()
    visualize_feature_distributions(mfcc_all, delta_all, delta2_all)
    visualize_sample_features()
    quality_ok = check_feature_quality()

    # Summary
    print("\n" + "=" * 80)
    print("FEATURE VALIDATION SUMMARY")
    print("=" * 80)

    if shape_ok and quality_ok:
        print("✓ All feature validation checks passed!")
        print("\n📝 Next Steps:")
        print("  1. Review feature distribution plots in artifacts/validation/")
        print("  2. Check if normalization is needed (see statistics above)")
        print(
            "  3. Consider alternative feature representations if performance is poor"
        )
        return 0
    else:
        print(
            "⚠️ Some feature validation checks failed. Please review the output above."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
