"""Evaluate all trained models - Task 12"""

import sys
from pathlib import Path
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.audio_cnn import AudioCNN
from src.models.audio_vit import AudioViT
from src.models.image_resnet import ImageResNet
from src.models.image_vit import ImageViT
from src.datasets.audio import AudioMFCCDataset
from src.datasets.image import ImageDataset, get_image_transforms

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
RESULTS_DIR = ARTIFACTS / "results"
RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 80)
print("TASK 12: EVALUATE ALL MODELS")
print("=" * 80)


def evaluate_model(model, dataloader, device, model_name):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    results = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
    }

    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")

    return results


def plot_confusion_matrix(cm, class_names, model_name, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot with smaller font for many classes
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Normalized Count"},
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix to {save_path}")


# -----------------------------------------------------------
# Evaluate Audio Models
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("Evaluating Audio Models")
print("=" * 80)

# Load data
xc_df = pd.read_parquet(ARTIFACTS / "xeno_canto_filtered.parquet")
xc_counts = xc_df["species_normalized"].value_counts()
species_to_keep = xc_counts[xc_counts >= 2].index
xc_df = xc_df[xc_df["species_normalized"].isin(species_to_keep)].copy()

with open(ARTIFACTS / "splits" / "xeno_canto_audio_splits.json", "r") as f:
    splits = json.load(f)

species_list = sorted(xc_df["species_normalized"].unique())
species_to_idx = {sp: i for i, sp in enumerate(species_list)}
num_classes = len(species_list)

cache_dir = ARTIFACTS / "audio_mfcc_cache" / "xeno_canto"

test_dataset = AudioMFCCDataset(
    df=xc_df,
    cache_dir=cache_dir,
    indices=splits["test"],
    species_to_idx=species_to_idx,
    transform=None,
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# Evaluate AudioCNN
print("\n--- AudioCNN ---")
model = AudioCNN(num_classes=num_classes).to(device)
checkpoint_path = MODELS_DIR / "audio_cnn" / "AudioCNN_best.pth"

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    results_cnn = evaluate_model(model, test_loader, device, "AudioCNN")

    # Save results
    with open(RESULTS_DIR / "audio_cnn_results.json", "w") as f:
        json.dump(results_cnn, f, indent=2)

    # Plot confusion matrix
    cm = confusion_matrix(results_cnn["labels"], results_cnn["predictions"])
    plot_confusion_matrix(
        cm, species_list, "AudioCNN", RESULTS_DIR / "audio_cnn_confusion_matrix.png"
    )
else:
    print(f"  Checkpoint not found at {checkpoint_path}")

# Evaluate AudioViT
print("\n--- AudioViT ---")
model = AudioViT(num_classes=num_classes, pretrained="google/vit-base-patch16-224").to(
    device
)
checkpoint_path = MODELS_DIR / "audio_vit" / "AudioViT_best.pth"

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    results_vit = evaluate_model(model, test_loader, device, "AudioViT")

    # Save results
    with open(RESULTS_DIR / "audio_vit_results.json", "w") as f:
        json.dump(results_vit, f, indent=2)

    # Plot confusion matrix
    cm = confusion_matrix(results_vit["labels"], results_vit["predictions"])
    plot_confusion_matrix(
        cm, species_list, "AudioViT", RESULTS_DIR / "audio_vit_confusion_matrix.png"
    )
else:
    print(f"  Checkpoint not found at {checkpoint_path}")

# -----------------------------------------------------------
# Evaluate Image Models
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("Evaluating Image Models")
print("=" * 80)

# Load data
cub_df = pd.read_parquet(ARTIFACTS / "cub_filtered.parquet")
cub_counts = cub_df["species_normalized"].value_counts()
species_to_keep = cub_counts[cub_counts >= 2].index
cub_df = cub_df[cub_df["species_normalized"].isin(species_to_keep)].copy()

with open(ARTIFACTS / "splits" / "cub_image_splits.json", "r") as f:
    splits = json.load(f)

species_list = sorted(cub_df["species_normalized"].unique())
species_to_idx = {sp: i for i, sp in enumerate(species_list)}
num_classes = len(species_list)

test_dataset = ImageDataset(
    df=cub_df,
    indices=splits["test"],
    species_to_idx=species_to_idx,
    transform=get_image_transforms(train=False, image_size=224),
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# Evaluate ResNet-18
print("\n--- ResNet-18 ---")
model = ImageResNet(num_classes=num_classes, pretrained=False).to(device)
checkpoint_path = MODELS_DIR / "image_resnet18" / "ImageResNet18_best.pth"

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    results_resnet = evaluate_model(model, test_loader, device, "ImageResNet18")

    # Save results
    with open(RESULTS_DIR / "image_resnet18_results.json", "w") as f:
        json.dump(results_resnet, f, indent=2)

    # Plot confusion matrix
    cm = confusion_matrix(results_resnet["labels"], results_resnet["predictions"])
    plot_confusion_matrix(
        cm,
        species_list,
        "ResNet-18",
        RESULTS_DIR / "image_resnet18_confusion_matrix.png",
    )
else:
    print(f"  Checkpoint not found at {checkpoint_path}")

# Evaluate ViT-B/16
print("\n--- ViT-B/16 ---")
model = ImageViT(num_classes=num_classes, pretrained="google/vit-base-patch16-224").to(
    device
)
checkpoint_path = MODELS_DIR / "image_vit" / "ImageViT_best.pth"

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    results_vit = evaluate_model(model, test_loader, device, "ImageViT")

    # Save results
    with open(RESULTS_DIR / "image_vit_results.json", "w") as f:
        json.dump(results_vit, f, indent=2)

    # Plot confusion matrix
    cm = confusion_matrix(results_vit["labels"], results_vit["predictions"])
    plot_confusion_matrix(
        cm, species_list, "ViT-B/16", RESULTS_DIR / "image_vit_confusion_matrix.png"
    )
else:
    print(f"  Checkpoint not found at {checkpoint_path}")

# -----------------------------------------------------------
# Aggregate Results
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("Aggregating Results")
print("=" * 80)

results_summary = {}

for model_name in ["audio_cnn", "audio_vit", "image_resnet18", "image_vit"]:
    results_file = RESULTS_DIR / f"{model_name}_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)
            results_summary[model_name] = {
                "accuracy": data["accuracy"],
                "f1_macro": data["f1_macro"],
                "f1_weighted": data["f1_weighted"],
            }

# Create comparison table
print("\n=== Model Comparison ===")
print(f"{'Model':<20} {'Accuracy':<12} {'F1 (macro)':<12} {'F1 (weighted)':<12}")
print("-" * 56)
for model_name, metrics in results_summary.items():
    print(
        f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f} {metrics['f1_weighted']:<12.4f}"
    )

# Save summary
with open(RESULTS_DIR / "results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# Create comparison plot
if results_summary:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = list(results_summary.keys())
    accuracies = [results_summary[m]["accuracy"] for m in models]
    f1_macros = [results_summary[m]["f1_macro"] for m in models]
    f1_weighteds = [results_summary[m]["f1_weighted"] for m in models]

    x = np.arange(len(models))

    axes[0].bar(x, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Test Accuracy Comparison")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, f1_macros, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha="right")
    axes[1].set_ylabel("F1 Score (Macro)")
    axes[1].set_title("F1 Macro Comparison")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, f1_weighteds, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha="right")
    axes[2].set_ylabel("F1 Score (Weighted)")
    axes[2].set_title("F1 Weighted Comparison")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved comparison plot to {RESULTS_DIR / 'model_comparison.png'}")

print("\n" + "=" * 80)
print("✓ TASK 12 COMPLETE - All models evaluated")
print("=" * 80)
