"""Evaluation metrics and utilities."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
    return_predictions: bool = False,
) -> Dict:
    """Evaluate model on a dataset.

    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to use
        return_predictions: If True, return predictions and labels

    Returns:
        Dictionary with metrics and optionally predictions
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in tqdm(data_loader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
    }

    if return_predictions:
        metrics["predictions"] = all_preds
        metrics["labels"] = all_labels
        metrics["probabilities"] = all_probs

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Path = None,
    figsize: Tuple[int, int] = (12, 10),
) -> np.ndarray:
    """Compute and optionally plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure (if provided)
        figsize: Figure size

    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)

    if save_path or class_names:
        plt.figure(figsize=figsize)

        # Normalize for better visualization if many classes
        if cm.shape[0] > 20:
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(
                cm_norm,
                annot=False,
                fmt=".2f",
                cmap="Blues",
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto",
            )
        else:
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto",
            )

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved confusion matrix to {save_path}")

        plt.close()

    return cm


def plot_training_curves(
    history: Dict, save_path: Path = None, figsize: Tuple[int, int] = (12, 5)
):
    """Plot training and validation curves.

    Args:
        history: Dictionary with train/val loss and accuracy
        save_path: Path to save figure (if provided)
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train")
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    plt.close()


def save_metrics_report(
    metrics: Dict, save_path: Path, experiment_name: str = "experiment"
):
    """Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON file
        experiment_name: Name of experiment
    """
    import json

    report = {"experiment": experiment_name, "metrics": metrics}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved metrics report to {save_path}")
