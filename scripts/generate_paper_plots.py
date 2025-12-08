import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "font.family": "serif"})


def plot_model_comparison():
    # Data
    # Image Classification (CUB-200 Full)
    image_models = ["SVM", "EfficientNetB0", "ViT-Base (1k)", "ViT-Base (21k)"]
    image_scores = [0.47, 0.74, 0.85, 0.83]
    image_colors = [
        "#d62728",
        "#1f77b4",
        "#2ca02c",
        "#2ca02c",
    ]  # Red, Blue, Green, Green

    # Audio & Intersection (Intersection Subset)
    audio_models = [
        "AudioViT (MFCC)",
        "AudioCNNv2 (MFCC)",
        "AST (LMS)",
        "ResNet-18 (Img)",
        "ViT (Img)",
    ]
    audio_scores = [0.3779, 0.4272, 0.5728, 0.8552, 0.9233]
    audio_colors = [
        "#ff7f0e",
        "#ff7f0e",
        "#9467bd",
        "#1f77b4",
        "#2ca02c",
    ]  # Orange, Orange, Purple, Blue, Green

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Image Classification
    bars1 = ax1.bar(image_models, image_scores, color=image_colors, alpha=0.8)
    ax1.set_title("Image Classification (Full CUB-200)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # Rotate x labels
    ax1.set_xticklabels(image_models, rotation=20, ha="right")

    # Plot 2: Audio & Intersection
    bars2 = ax2.bar(audio_models, audio_scores, color=audio_colors, alpha=0.8)
    ax2.set_title("Audio & Image (Intersection Subset)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # Rotate x labels
    ax2.set_xticklabels(audio_models, rotation=20, ha="right")

    # Legend for colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#d62728", label="Traditional"),
        Patch(facecolor="#1f77b4", label="CNN"),
        Patch(facecolor="#2ca02c", label="Transformer (Vision)"),
        Patch(facecolor="#ff7f0e", label="Transformer/CNN (Audio Baseline)"),
        Patch(facecolor="#9467bd", label="Audio Spectrogram Transformer"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=5,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend

    # Save
    if not os.path.exists("paper"):
        os.makedirs("paper")
    plt.savefig("paper/model_comparison_chart.pdf")
    print("Saved paper/model_comparison_chart.pdf")


def plot_class_distribution():
    try:
        # Try to find the parquet file
        parquet_path = "artifacts/xeno_canto_filtered.parquet"
        if not os.path.exists(parquet_path):
            # Fallback to hardcoded values if file not found (e.g. in test env)
            print(
                f"Warning: {parquet_path} not found. Using hardcoded data for demonstration."
            )
            # Using values from Table 1 and some synthetic data to mimic the distribution
            counts_values = (
                [1216, 984, 621, 608, 466]
                + [100] * 20
                + [55] * 20
                + [10] * 20
                + [4, 4, 2, 2]
            )
            counts = pd.Series(counts_values)
            counts = counts.sort_values(ascending=False)
        else:
            df = pd.read_parquet(parquet_path)
            counts = df["species_normalized"].value_counts()
            counts = counts.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))

        # Plot
        plt.bar(
            range(len(counts)), counts.values, color="#1f77b4", width=1.0, alpha=0.8
        )
        plt.yscale("log")
        plt.xlabel("Species Rank (Sorted by Frequency)", fontsize=12)
        plt.ylabel("Number of Samples (Log Scale)", fontsize=12)
        plt.title(
            "Class Distribution: Long-Tail Imbalance", fontsize=14, fontweight="bold"
        )
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # Annotate head and tail
        max_val = counts.iloc[0]
        min_val = counts.iloc[-1]

        plt.annotate(
            f"Most Common: {max_val}",
            xy=(0, max_val),
            xytext=(10, max_val),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=10,
        )

        plt.annotate(
            f"Least Common: {min_val}",
            xy=(len(counts) - 1, min_val),
            xytext=(len(counts) - 30, min_val * 5),
            arrowprops=dict(facecolor="black", shrink=0.05),
            fontsize=10,
        )

        # Add text about imbalance ratio
        ratio = max_val / min_val
        plt.text(
            0.7,
            0.8,
            f"Imbalance Ratio: {ratio:.0f}:1",
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        plt.tight_layout()
        if not os.path.exists("paper"):
            os.makedirs("paper")
        plt.savefig("paper/class_distribution.pdf")
        print("Saved paper/class_distribution.pdf")

    except Exception as e:
        print(f"Could not generate class distribution plot: {e}")


if __name__ == "__main__":
    plot_model_comparison()
    plot_class_distribution()
