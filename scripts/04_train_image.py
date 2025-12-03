"""Train image models - Task 11"""

import sys
from pathlib import Path
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.image_resnet import ImageResNet
from src.models.image_vit import ImageViT
from src.datasets.image import ImageDataset, get_image_transforms
from src.training.trainer import Trainer

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
MODELS_DIR.mkdir(exist_ok=True)

device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = str(device_obj)
print(f"Using device: {device}")

print("=" * 80)
print("TASK 11: TRAIN IMAGE MODELS")
print("=" * 80)

# Load filtered data and splits
cub_df = pd.read_parquet(ARTIFACTS / "cub_filtered.parquet")

# Filter to species with >=2 samples
cub_counts = cub_df["species_normalized"].value_counts()
species_to_keep = cub_counts[cub_counts >= 2].index
cub_df = cub_df[cub_df["species_normalized"].isin(species_to_keep)].copy()

with open(ARTIFACTS / "splits" / "cub_image_splits.json", "r") as f:
    splits = json.load(f)

# Create species to label mapping
species_list = sorted(cub_df["species_normalized"].unique())
species_to_idx = {sp: i for i, sp in enumerate(species_list)}
num_classes = len(species_list)

print(f"\nDataset: {len(cub_df)} images, {num_classes} species")
print(
    f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
)

# Create datasets with transforms
train_dataset = ImageDataset(
    df=cub_df,
    indices=splits["train"],
    species_to_idx=species_to_idx,
    transform=get_image_transforms(train=True, image_size=224),
)

val_dataset = ImageDataset(
    df=cub_df,
    indices=splits["val"],
    species_to_idx=species_to_idx,
    transform=get_image_transforms(train=False, image_size=224),
)

test_dataset = ImageDataset(
    df=cub_df,
    indices=splits["test"],
    species_to_idx=species_to_idx,
    transform=get_image_transforms(train=False, image_size=224),
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
# Train ResNet-18
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("Training ResNet-18")
print("=" * 80)

model = ImageResNet(num_classes=num_classes, pretrained=True).to(device_obj)
print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    checkpoint_dir=MODELS_DIR / "image_resnet18",
    experiment_name="ImageResNet18",
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
axes[0, 0].set_title("ResNet-18 - Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history["train_acc"], label="Train")
axes[0, 1].plot(history["val_acc"], label="Val")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("ResNet-18 - Accuracy")
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
plt.savefig(MODELS_DIR / "image_resnet18" / "training_curves.png", dpi=150)
plt.close()

# Save history
with open(MODELS_DIR / "image_resnet18" / "history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

print("\n✓ ResNet-18 training complete")
print(f"✓ Best val accuracy: {max(history['val_acc']):.4f}")
print(f"✓ Checkpoint saved to {MODELS_DIR / 'image_resnet18'}")

# -----------------------------------------------------------
# Train ViT-B/16
# -----------------------------------------------------------
print("\n" + "=" * 80)
print("Training ViT-B/16")
print("=" * 80)

model = ImageViT(num_classes=num_classes, pretrained="google/vit-base-patch16-224").to(
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
    checkpoint_dir=MODELS_DIR / "image_vit",
    experiment_name="ImageViT",
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
axes[0, 0].set_title("ViT-B/16 - Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history["train_acc"], label="Train")
axes[0, 1].plot(history["val_acc"], label="Val")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("ViT-B/16 - Accuracy")
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
plt.savefig(MODELS_DIR / "image_vit" / "training_curves.png", dpi=150)
plt.close()

# Save history
with open(MODELS_DIR / "image_vit" / "history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

print("\n✓ ViT-B/16 training complete")
print(f"✓ Best val accuracy: {max(history['val_acc']):.4f}")
print(f"✓ Checkpoint saved to {MODELS_DIR / 'image_vit'}")

print("\n" + "=" * 80)
print("✓ TASK 11 COMPLETE - Image models trained")
print("=" * 80)
