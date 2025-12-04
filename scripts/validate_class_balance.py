#!/usr/bin/env python3
"""Analyze class distribution and imbalance in datasets.

This script checks:
- Class distribution statistics
- Imbalance ratios
- Rare species identification
- Recommended class weights
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_audio_class_balance():
    """Analyze class balance in audio dataset."""
    print("\n" + "=" * 80)
    print("AUDIO DATASET CLASS BALANCE ANALYSIS")
    print("=" * 80)

    artifacts = project_root / "artifacts"
    df = pd.read_parquet(artifacts / "xeno_canto_filtered.parquet")

    counts = df["species_normalized"].value_counts()

    print(f"\n📊 Overall Statistics:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Total species: {len(counts)}")
    print(f"  Min samples per species: {counts.min()}")
    print(f"  Max samples per species: {counts.max()}")
    print(f"  Median samples per species: {counts.median():.0f}")
    print(f"  Mean samples per species: {counts.mean():.1f}")
    print(f"  Std samples per species: {counts.std():.1f}")
    print(f"  Imbalance ratio (max/min): {counts.max() / counts.min():.1f}x")

    # Distribution analysis
    quartiles = counts.quantile([0.25, 0.5, 0.75])
    print(f"\n📈 Distribution Quartiles:")
    print(f"  Q1 (25%): {quartiles[0.25]:.0f} samples")
    print(f"  Q2 (50%): {quartiles[0.5]:.0f} samples")
    print(f"  Q3 (75%): {quartiles[0.75]:.0f} samples")

    # Identify rare and common species
    rare_threshold = counts.quantile(0.25)
    common_threshold = counts.quantile(0.75)

    rare_species = counts[counts <= rare_threshold]
    common_species = counts[counts >= common_threshold]

    print(f"\n🔍 Species Categories:")
    print(
        f"  Rare species (≤{rare_threshold:.0f} samples): {len(rare_species)} species"
    )
    print(
        f"  Common species (≥{common_threshold:.0f} samples): {len(common_species)} species"
    )
    print(
        f"  Medium species: {len(counts) - len(rare_species) - len(common_species)} species"
    )

    # Show extreme cases
    print(f"\n🏆 Top 5 Most Common Species:")
    for species, count in counts.head(5).items():
        print(f"  {species}: {count} samples")

    print(f"\n⚠️ Top 5 Rarest Species:")
    for species, count in counts.tail(5).items():
        print(f"  {species}: {count} samples")

    return df, counts


def analyze_image_class_balance():
    """Analyze class balance in image dataset."""
    print("\n" + "=" * 80)
    print("IMAGE DATASET CLASS BALANCE ANALYSIS")
    print("=" * 80)

    artifacts = project_root / "artifacts"
    df = pd.read_parquet(artifacts / "cub_filtered.parquet")

    counts = df["species_normalized"].value_counts()

    print(f"\n📊 Overall Statistics:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Total species: {len(counts)}")
    print(f"  Min samples per species: {counts.min()}")
    print(f"  Max samples per species: {counts.max()}")
    print(f"  Median samples per species: {counts.median():.0f}")
    print(f"  Mean samples per species: {counts.mean():.1f}")
    print(f"  Imbalance ratio (max/min): {counts.max() / counts.min():.1f}x")

    return df, counts


def analyze_split_balance():
    """Analyze class balance in train/val/test splits."""
    print("\n" + "=" * 80)
    print("SPLIT BALANCE ANALYSIS")
    print("=" * 80)

    artifacts = project_root / "artifacts"

    # Load audio data and splits
    df = pd.read_parquet(artifacts / "xeno_canto_filtered.parquet")

    with open(artifacts / "splits/xeno_canto_audio_splits.json", "r") as f:
        splits = json.load(f)

    print("\n🎯 Audio Dataset Splits:")

    for split_name in ["train", "val", "test"]:
        split_indices = splits[split_name]
        split_df = df.iloc[split_indices]
        split_counts = split_df["species_normalized"].value_counts()

        print(f"\n  {split_name.upper()} Split:")
        print(f"    Samples: {len(split_df):,}")
        print(f"    Species: {len(split_counts)}")
        print(f"    Min per species: {split_counts.min()}")
        print(f"    Max per species: {split_counts.max()}")
        print(f"    Mean per species: {split_counts.mean():.1f}")

        # Check for species with very few samples
        few_samples = split_counts[split_counts < 3]
        if len(few_samples) > 0:
            print(f"    ⚠️ Species with <3 samples: {len(few_samples)}")


def compute_class_weights(counts):
    """Compute recommended class weights."""
    print("\n" + "=" * 80)
    print("COMPUTING CLASS WEIGHTS")
    print("=" * 80)

    # Method 1: Inverse frequency
    total_samples = counts.sum()
    n_classes = len(counts)
    weights = total_samples / (n_classes * counts)

    print(f"\n⚙️ Inverse Frequency Weights:")
    print(f"  Min weight: {weights.min():.4f}")
    print(f"  Max weight: {weights.max():.4f}")
    print(f"  Median weight: {weights.median():.4f}")
    print(f"  Weight ratio (max/min): {weights.max() / weights.min():.2f}x")

    # Method 2: Balanced weights (sklearn style)
    balanced_weights = n_classes / (2 * counts)

    print(f"\n⚙️ Balanced Weights (sklearn style):")
    print(f"  Min weight: {balanced_weights.min():.4f}")
    print(f"  Max weight: {balanced_weights.max():.4f}")
    print(f"  Median weight: {balanced_weights.median():.4f}")

    # Method 3: Square root reweighting (less aggressive)
    sqrt_inv = np.sqrt(total_samples / (n_classes * counts))

    print(f"\n⚙️ Square Root Reweighting:")
    print(f"  Min weight: {sqrt_inv.min():.4f}")
    print(f"  Max weight: {sqrt_inv.max():.4f}")
    print(f"  Median weight: {sqrt_inv.median():.4f}")

    return weights, balanced_weights, sqrt_inv


def visualize_class_distribution(counts, dataset_name):
    """Visualize class distribution."""
    print(f"\n📊 Creating visualizations for {dataset_name}...")

    output_dir = project_root / "artifacts" / "validation"
    output_dir.mkdir(exist_ok=True)

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Histogram of samples per species
    axes[0, 0].hist(counts, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(
        counts.median(),
        color="red",
        linestyle="--",
        label=f"Median: {counts.median():.0f}",
    )
    axes[0, 0].axvline(
        counts.mean(),
        color="orange",
        linestyle="--",
        label=f"Mean: {counts.mean():.1f}",
    )
    axes[0, 0].set_xlabel("Samples per Species")
    axes[0, 0].set_ylabel("Number of Species")
    axes[0, 0].set_title(f"{dataset_name} - Distribution of Samples per Species")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 2. Bar plot of top 20 species
    top_20 = counts.head(20)
    axes[0, 1].barh(range(len(top_20)), top_20.values, color="forestgreen", alpha=0.7)
    axes[0, 1].set_yticks(range(len(top_20)))
    axes[0, 1].set_yticklabels([s[:30] for s in top_20.index], fontsize=8)
    axes[0, 1].set_xlabel("Number of Samples")
    axes[0, 1].set_title(f"{dataset_name} - Top 20 Species by Sample Count")
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis="x", alpha=0.3)

    # 3. Bar plot of bottom 20 species
    bottom_20 = counts.tail(20)
    axes[1, 0].barh(range(len(bottom_20)), bottom_20.values, color="coral", alpha=0.7)
    axes[1, 0].set_yticks(range(len(bottom_20)))
    axes[1, 0].set_yticklabels([s[:30] for s in bottom_20.index], fontsize=8)
    axes[1, 0].set_xlabel("Number of Samples")
    axes[1, 0].set_title(f"{dataset_name} - Bottom 20 Species by Sample Count")
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis="x", alpha=0.3)

    # 4. Cumulative distribution
    sorted_counts = counts.sort_values(ascending=False)
    cumsum = sorted_counts.cumsum()
    cumsum_pct = (cumsum / cumsum.max()) * 100

    axes[1, 1].plot(range(len(cumsum_pct)), cumsum_pct, linewidth=2, color="purple")
    axes[1, 1].axhline(80, color="red", linestyle="--", alpha=0.5, label="80% of data")
    axes[1, 1].set_xlabel("Number of Species (sorted by frequency)")
    axes[1, 1].set_ylabel("Cumulative % of Samples")
    axes[1, 1].set_title(f"{dataset_name} - Cumulative Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Find how many species account for 80% of data
    species_for_80pct = (cumsum_pct <= 80).sum()
    axes[1, 1].axvline(species_for_80pct, color="red", linestyle="--", alpha=0.5)
    axes[1, 1].text(
        species_for_80pct,
        40,
        f"{species_for_80pct} species\n= 80% of data",
        ha="left",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    output_file = (
        output_dir / f"class_distribution_{dataset_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved class distribution to: {output_file}")

    plt.close()


def main():
    """Run all class balance analyses."""
    print("=" * 80)
    print("CLASS BALANCE VALIDATION SUITE")
    print("=" * 80)
    print(f"Project root: {project_root}")

    # Analyze audio dataset
    audio_df, audio_counts = analyze_audio_class_balance()
    visualize_class_distribution(audio_counts, "Audio Dataset")

    # Analyze image dataset
    image_df, image_counts = analyze_image_class_balance()
    visualize_class_distribution(image_counts, "Image Dataset")

    # Analyze splits
    analyze_split_balance()

    # Compute class weights
    weights, balanced_weights, sqrt_weights = compute_class_weights(audio_counts)

    # Save recommended weights
    output_dir = project_root / "artifacts" / "validation"
    output_dir.mkdir(exist_ok=True)

    weights_dict = {
        "inverse_frequency": weights.to_dict(),
        "balanced": balanced_weights.to_dict(),
        "sqrt_reweighting": sqrt_weights.to_dict(),
    }

    weights_file = output_dir / "recommended_class_weights.json"
    with open(weights_file, "w") as f:
        json.dump(weights_dict, f, indent=2)

    print(f"\n✓ Saved recommended class weights to: {weights_file}")

    # Summary
    print("\n" + "=" * 80)
    print("CLASS BALANCE SUMMARY")
    print("=" * 80)

    print("\n🔍 Key Findings:")

    # Audio analysis
    audio_imbalance = audio_counts.max() / audio_counts.min()
    if audio_imbalance > 10:
        print(f"  ⚠️ SEVERE audio class imbalance: {audio_imbalance:.1f}x")
        print("     → Recommendation: Use class-weighted loss or oversampling")
    elif audio_imbalance > 5:
        print(f"  ⚠️ Moderate audio class imbalance: {audio_imbalance:.1f}x")
        print("     → Recommendation: Consider class weighting")
    else:
        print(f"  ✓ Audio class balance acceptable: {audio_imbalance:.1f}x")

    # Image analysis
    image_imbalance = image_counts.max() / image_counts.min()
    if image_imbalance > 10:
        print(f"  ⚠️ SEVERE image class imbalance: {image_imbalance:.1f}x")
    elif image_imbalance > 5:
        print(f"  ⚠️ Moderate image class imbalance: {image_imbalance:.1f}x")
    else:
        print(f"  ✓ Image class balance acceptable: {image_imbalance:.1f}x")

    print("\n📝 Recommended Actions:")
    print("  1. Implement class-weighted loss function")
    print("  2. Use WeightedRandomSampler for training")
    print("  3. Apply oversampling to minority classes")
    print("  4. Consider focal loss for hard examples")
    print("  5. Monitor per-class metrics (not just overall accuracy)")

    print("\n✓ Class balance analysis complete!")
    print(f"  Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
