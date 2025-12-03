"""Results aggregation utilities."""

from pathlib import Path
from typing import List, Dict
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_results(metrics_dir: Path) -> pd.DataFrame:
    """Aggregate metrics from all experiments into a DataFrame.

    Args:
        metrics_dir: Directory containing metrics JSON files

    Returns:
        DataFrame with aggregated results
    """
    metrics_files = list(metrics_dir.glob("*.json"))

    records = []
    for f in metrics_files:
        with open(f) as fh:
            data = json.load(fh)

            # Parse experiment name (format: dataset_modality_model)
            exp_name = data.get("experiment", f.stem)
            parts = exp_name.split("_")

            record = {
                "experiment": exp_name,
                "dataset": parts[0] if len(parts) > 0 else "unknown",
                "modality": parts[1] if len(parts) > 1 else "unknown",
                "model": parts[2] if len(parts) > 2 else "unknown",
            }

            # Add metrics
            metrics = data.get("metrics", {})
            record.update(
                {
                    "accuracy": metrics.get("accuracy", 0),
                    "f1_macro": metrics.get("f1_macro", 0),
                    "f1_weighted": metrics.get("f1_weighted", 0),
                }
            )

            records.append(record)

    df = pd.DataFrame(records)
    return df


def generate_comparison_table(
    results_df: pd.DataFrame, save_path: Path = None
) -> pd.DataFrame:
    """Generate a comparison table of results.

    Args:
        results_df: DataFrame with aggregated results
        save_path: Path to save LaTeX table

    Returns:
        Pivot table with results
    """
    # Create pivot table
    pivot = results_df.pivot_table(
        index=["modality", "model"],
        columns="dataset",
        values=["accuracy", "f1_macro"],
        aggfunc="mean",
    )

    # Round values
    pivot = pivot.round(4)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as LaTeX
        latex_str = pivot.to_latex(
            float_format="%.4f",
            caption="Model Performance Comparison",
            label="tab:results",
        )

        with open(save_path, "w") as f:
            f.write(latex_str)

        print(f"Saved comparison table to {save_path}")

    return pivot


def generate_figures(results_df: pd.DataFrame, figures_dir: Path):
    """Generate publication-ready figures.

    Args:
        results_df: DataFrame with aggregated results
        figures_dir: Directory to save figures
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150

    # Figure 1: Accuracy comparison by modality and model
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by modality and model
    grouped = results_df.groupby(["modality", "model"])["accuracy"].mean().reset_index()

    # Create grouped bar plot
    modalities = grouped["modality"].unique()
    x = range(len(modalities))
    width = 0.35

    models = grouped["model"].unique()
    for i, model in enumerate(models):
        data = grouped[grouped["model"] == model]
        values = [
            data[data["modality"] == m]["accuracy"].values[0]
            if len(data[data["modality"] == m]) > 0
            else 0
            for m in modalities
        ]
        ax.bar([xi + i * width for xi in x], values, width, label=model)

    ax.set_xlabel("Modality")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance by Modality")
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(modalities)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_by_modality.pdf", bbox_inches="tight")
    plt.close()

    # Figure 2: F1 score comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped_f1 = (
        results_df.groupby(["modality", "model"])["f1_macro"].mean().reset_index()
    )

    for i, model in enumerate(models):
        data = grouped_f1[grouped_f1["model"] == model]
        values = [
            data[data["modality"] == m]["f1_macro"].values[0]
            if len(data[data["modality"] == m]) > 0
            else 0
            for m in modalities
        ]
        ax.bar([xi + i * width for xi in x], values, width, label=model)

    ax.set_xlabel("Modality")
    ax.set_ylabel("F1 Score (Macro)")
    ax.set_title("Model F1 Score by Modality")
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(modalities)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "f1_by_modality.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved figures to {figures_dir}")


def save_summary_parquet(results_df: pd.DataFrame, save_path: Path):
    """Save aggregated results as Parquet file.

    Args:
        results_df: DataFrame with results
        save_path: Path to save file
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(save_path, index=False)
    print(f"Saved results summary to {save_path}")
