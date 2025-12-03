"""Train audio models - Task 13"""

import sys
from pathlib import Path
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.audio_cnn import AudioCNN
from src.models.audio_vit import AudioViT
from src.datasets.audio import AudioMFCCDataset
from src.training.trainer import Trainer

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
MODELS_DIR.mkdir(exist_ok=True)

device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = str(device_obj)
print(f"Using device: {device}")

print("=" * 80)
print("TASK 13: TRAIN AUDIO MODELS")
print("=" * 80)

# Load filtered data and splits
xc_df = pd.read_parquet(ARTIFACTS / "xeno_canto_filtered.parquet")

# Filter to species with >=2 samples
xc_counts = xc_df["species_normalized"].value_counts()
species_to_keep = xc_counts[xc_counts >= 2].index
xc_df = xc_df[xc_df["species_normalized"].isin(species_to_keep)].copy()

with open(ARTIFACTS / "splits" / "xeno_canto_audio_splits.json", "r") as f:
    splits = json.load(f)

# Create species to label mapping
species_list = sorted(xc_df["species_normalized"].unique())
species_to_idx = {sp: i for i, sp in enumerate(species_list)}
num_classes = len(species_list)

print(f"\nDataset: {len(xc_df)} recordings, {num_classes} species")
print(
    f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
)

# Create datasets
cache_dir = ARTIFACTS / "audio_mfcc_cache" / "xeno_canto"

train_dataset = AudioMFCCDataset(
    df=xc_df,
    cache_dir=cache_dir,
    indices=splits["train"],
    species_to_idx=species_to_idx,
    transform=None,
)

val_dataset = AudioMFCCDataset(
    df=xc_df,
    cache_dir=cache_dir,
    indices=splits["val"],
    species_to_idx=species_to_idx,
    transform=None,
)

test_dataset = AudioMFCCDataset(
    df=xc_df,
    cache_dir=cache_dir,
    indices=splits["test"],
    species_to_idx=species_to_idx,
    transform=None,
)

print(
    f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# -----------------------------------------------------------
# Train AudioCNN
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("Training AudioCNN")
print("=" * 80)

model = AudioCNN(num_classes=num_classes).to(device_obj)
print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    checkpoint_dir=MODELS_DIR / "audio_cnn",
    experiment_name="AudioCNN",
    use_amp=True,
    gradient_clip=1.0,
    early_stopping_patience=7,
)

history = trainer.train(num_epochs=50)

# Save training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(history["train_loss"], label="Train")
axes[0, 0].plot(history["val_loss"], label="Val")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("AudioCNN - Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history["train_acc"], label="Train")
axes[0, 1].plot(history["val_acc"], label="Val")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("AudioCNN - Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Learning rate (if tracked)
if "lr" in history:
    axes[1, 0].plot(history["lr"])
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True)
else:
    axes[1, 0].text(
        0.5,
        0.5,
        "LR not tracked",
        ha="center",
        va="center",
        transform=axes[1, 0].transAxes,
    )
    axes[1, 0].set_title("Learning Rate")

# Plot training progress over time
epochs = list(range(1, len(history["train_loss"]) + 1))
axes[1, 1].plot(epochs, history["train_loss"], label="Train Loss", alpha=0.7)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Learning Rate")
axes[1, 1].set_title("Learning Rate Schedule")
axes[1, 1].set_yscale("log")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(MODELS_DIR / "audio_cnn" / "training_curves.png", dpi=150)
plt.close()

# Save history
with open(MODELS_DIR / "audio_cnn" / "history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

print(f"\n✓ AudioCNN training complete")
print(f"✓ Best val accuracy: {max(history['val_acc']):.4f}")
print(f"✓ Checkpoint saved to {MODELS_DIR / 'audio_cnn'}")

# -----------------------------------------------------------
# Train AudioViT
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("Training AudioViT")
print("=" * 80)

model = AudioViT(num_classes=num_classes, pretrained="google/vit-base-patch16-224").to(
    device_obj
)
print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    checkpoint_dir=MODELS_DIR / "audio_vit",
    experiment_name="AudioViT",
    use_amp=True,
    gradient_clip=1.0,
    early_stopping_patience=10,
)

history = trainer.train(num_epochs=50)

# Save training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(history["train_loss"], label="Train")
axes[0, 0].plot(history["val_loss"], label="Val")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("AudioViT - Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history["train_acc"], label="Train")
axes[0, 1].plot(history["val_acc"], label="Val")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("AudioViT - Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Learning rate (if tracked)
if "lr" in history:
    axes[1, 0].plot(history["lr"])
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True)
else:
    axes[1, 0].text(
        0.5,
        0.5,
        "LR not tracked",
        ha="center",
        va="center",
        transform=axes[1, 0].transAxes,
    )
    axes[1, 0].set_title("Learning Rate")

# Plot training progress over time
epochs = list(range(1, len(history["train_loss"]) + 1))
axes[1, 1].plot(epochs, history["train_loss"], label="Train Loss", alpha=0.7)
axes[1, 1].plot(epochs, history["val_loss"], label="Val Loss", alpha=0.7)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].set_title("Loss Comparison")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(MODELS_DIR / "audio_vit" / "training_curves.png", dpi=150)
plt.close()

# Save history
with open(MODELS_DIR / "audio_vit" / "history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

print(f"\n✓ AudioViT training complete")
print(f"✓ Best val accuracy: {max(history['val_acc']):.4f}")
print(f"✓ Checkpoint saved to {MODELS_DIR / 'audio_vit'}")

print("\n" + "=" * 80)
print("✓ TASK 13 COMPLETE - Audio models trained")
print("=" * 80)
