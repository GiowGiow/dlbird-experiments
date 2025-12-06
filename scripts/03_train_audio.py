"""Train audio models - Phase 1: Class Weights + Normalization"""

import sys
from pathlib import Path
import json
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.audio_cnn import AudioCNN
from src.models.audio_cnn_v2 import AudioCNNv2
from src.models.audio_vit import AudioViT
from src.models.audio_ast import AudioAST
from src.datasets.audio import AudioMFCCDataset
from src.datasets.audio_spectrogram import AudioSpectrogramDataset, collate_spectrograms
from src.training.trainer import Trainer, WarmupScheduler
from src.training.losses import FocalLoss


def load_class_weights(species_list, method="sqrt_reweighting", weights_path=None):
    """Load class weights from validation artifacts.

    Args:
        species_list: Ordered list of species names matching dataset order
        method: Weighting method - 'balanced', 'inverse_frequency', or 'sqrt_reweighting' (recommended)
        weights_path: Path to weights JSON file (default: artifacts/validation/recommended_class_weights.json)

    Returns:
        torch.FloatTensor of shape (num_classes,) with class weights
    """
    if weights_path is None:
        weights_path = (
            Path(__file__).parent.parent
            / "artifacts"
            / "validation"
            / "recommended_class_weights.json"
        )

    with open(weights_path, "r") as f:
        weights_dict = json.load(f)[method]

    # Map weights to dataset species order
    class_weights = [weights_dict[species] for species in species_list]
    return torch.FloatTensor(class_weights)


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train audio classification models")
parser.add_argument(
    "--model",
    type=str,
    default="AudioCNN",
    choices=["AudioCNN", "AudioCNNv2", "AudioViT", "AST"],
    help="Model architecture to train",
)
parser.add_argument(
    "--use-class-weights",
    action="store_true",
    help="Use class-balanced weighting in loss function",
)
parser.add_argument(
    "--weight-method",
    type=str,
    default="sqrt_reweighting",
    choices=["balanced", "inverse_frequency", "sqrt_reweighting"],
    help="Class weighting method (default: sqrt_reweighting, most stable)",
)
parser.add_argument(
    "--loss-type",
    type=str,
    default="ce",
    choices=["ce", "focal"],
    help="Loss function: ce (CrossEntropy) or focal (FocalLoss)",
)
parser.add_argument(
    "--focal-gamma",
    type=float,
    default=2.0,
    help="Focal loss gamma parameter (default: 2.0)",
)
parser.add_argument(
    "--focal-alpha",
    type=float,
    default=None,
    help="Focal loss alpha parameter (default: None, no alpha weighting)",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=0,
    help="Number of warmup epochs for learning rate (default: 0, no warmup)",
)
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument(
    "--batch-size", type=int, default=32, help="Batch size for training"
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument(
    "--save-name",
    type=str,
    default=None,
    help="Name for saved checkpoint (default: model name)",
)
parser.add_argument(
    "--ast-lr-backbone",
    type=float,
    default=5e-5,
    help="Learning rate for AST backbone (default: 5e-5)",
)
parser.add_argument(
    "--ast-lr-head",
    type=float,
    default=1e-3,
    help="Learning rate for AST classification head (default: 1e-3)",
)
parser.add_argument(
    "--use-lms",
    action="store_true",
    help="Use Log-Mel Spectrograms instead of MFCCs (required for AST)",
)
parser.add_argument(
    "--specaugment",
    action="store_true",
    help="Apply SpecAugment (frequency + time masking) during training",
)
parser.add_argument(
    "--mixup", action="store_true", help="Apply MixUp augmentation during training"
)
parser.add_argument(
    "--mixup-alpha",
    type=float,
    default=0.4,
    help="MixUp alpha parameter for Beta distribution (default: 0.4)",
)
parser.add_argument(
    "--mixup-prob",
    type=float,
    default=0.5,
    help="Probability of applying MixUp to a batch (default: 0.5)",
)
parser.add_argument(
    "--specaugment-freq",
    type=int,
    default=15,
    help="Frequency mask parameter for SpecAugment (default: 15 bins)",
)
parser.add_argument(
    "--specaugment-time",
    type=int,
    default=35,
    help="Time mask parameter for SpecAugment (default: 35 frames)",
)
parser.add_argument(
    "--specaugment-prob",
    type=float,
    default=0.8,
    help="Probability of applying SpecAugment (default: 0.8)",
)
args = parser.parse_args()

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
MODELS_DIR.mkdir(exist_ok=True)

device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = str(device_obj)
print(f"Using device: {device}")

print("=" * 80)
print(f"TRAIN AUDIO MODEL: {args.model}")
print(f"Loss: {args.loss_type.upper()}", end="")
if args.loss_type == "focal":
    print(f" (γ={args.focal_gamma}, α={args.focal_alpha})")
else:
    print(f" (Class Weights={'ON' if args.use_class_weights else 'OFF'})")
print(f"Normalization: ON, Warmup Epochs: {args.warmup_epochs}")
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

# Auto-enable LMS for AST model
use_lms = args.use_lms or (args.model == "AST")

# Set up augmentation for training
train_augment = None
if args.specaugment and use_lms:
    from src.augmentation.spec_augment import SpecAugment

    train_augment = SpecAugment(
        freq_mask_param=args.specaugment_freq,
        time_mask_param=args.specaugment_time,
        num_freq_masks=1,
        num_time_masks=1,
        prob=args.specaugment_prob,
    )
    print(
        f"✓ SpecAugment enabled: freq_mask={args.specaugment_freq}, time_mask={args.specaugment_time}, prob={args.specaugment_prob}"
    )

# Create datasets
if use_lms:
    print("\n📊 Using Log-Mel Spectrograms (LMS)")
    cache_dir = ARTIFACTS / "audio_lms_cache" / "xeno_canto"

    train_dataset = AudioSpectrogramDataset(
        df=xc_df,
        cache_dir=cache_dir,
        split=splits["train"],
        species_to_idx=species_to_idx,
        target_duration=4.0,
        normalize=True,
        augment=train_augment,  # Apply SpecAugment only to training set
    )

    val_dataset = AudioSpectrogramDataset(
        df=xc_df,
        cache_dir=cache_dir,
        split=splits["val"],
        species_to_idx=species_to_idx,
        target_duration=4.0,
        normalize=True,
        augment=None,  # No augmentation for validation
    )

    test_dataset = AudioSpectrogramDataset(
        df=xc_df,
        cache_dir=cache_dir,
        split=splits["test"],
        species_to_idx=species_to_idx,
        target_duration=4.0,
        normalize=True,
        augment=None,  # No augmentation for test
    )

    # Check cache coverage
    cache_stats = train_dataset.get_cache_stats()
    print(
        f"Cache coverage: {cache_stats['cache_coverage'] * 100:.1f}% ({cache_stats['cached_samples']}/{cache_stats['total_samples']})"
    )

else:
    print("\n📊 Using MFCCs")
    cache_dir = ARTIFACTS / "audio_mfcc_cache" / "xeno_canto"

    train_dataset = AudioMFCCDataset(
        df=xc_df,
        cache_dir=cache_dir,
        indices=splits["train"],
        species_to_idx=species_to_idx,
        transform=None,
        normalize=True,
    )

    val_dataset = AudioMFCCDataset(
        df=xc_df,
        cache_dir=cache_dir,
        indices=splits["val"],
        species_to_idx=species_to_idx,
        transform=None,
        normalize=True,
    )

    test_dataset = AudioMFCCDataset(
        df=xc_df,
        cache_dir=cache_dir,
        indices=splits["test"],
        species_to_idx=species_to_idx,
        transform=None,
        normalize=True,
    )

print(
    f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
)

# Load class weights if requested
class_weights = None
if args.use_class_weights:
    print(f"\nLoading class weights (method: {args.weight_method})...")
    class_weights = load_class_weights(species_list, method=args.weight_method)
    print(f"Class weights shape: {class_weights.shape}")
    print(f"Weight range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
    print(f"Weight mean: {class_weights.mean():.4f}, std: {class_weights.std():.4f}")

# Create dataloaders with custom collate for LMS
collate_fn = collate_spectrograms if use_lms else None

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)

# -----------------------------------------------------------
# Train Model
# -----------------------------------------------------------
print("\n" + "=" * 80)
print(f"Training {args.model}")
print("=" * 80)

# Select model architecture
if args.model == "AudioCNN":
    model = AudioCNN(num_classes=num_classes).to(device_obj)
    default_lr = 1e-3
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr if args.lr != 1e-3 else default_lr,
        weight_decay=1e-4,
    )
elif args.model == "AudioCNNv2":
    model = AudioCNNv2(num_classes=num_classes).to(device_obj)
    default_lr = 1e-3
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr if args.lr != 1e-3 else default_lr,
        weight_decay=1e-4,
    )
elif args.model == "AudioViT":
    model = AudioViT(num_classes=num_classes).to(device_obj)
    default_lr = 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr if args.lr != 1e-3 else default_lr,
        weight_decay=1e-4,
    )
else:  # AST
    model = AudioAST(num_classes=num_classes, freeze_backbone=False).to(device_obj)
    params = model.count_parameters()
    print(
        f"AST Model: {params['total']:,} total params ({params['trainable']:,} trainable, {params['frozen']:,} frozen)"
    )

    # Use discriminative learning rates for AST
    param_groups = model.get_param_groups(
        backbone_lr=args.ast_lr_backbone, head_lr=args.ast_lr_head
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
    print(
        f"Using AdamW with backbone_lr={args.ast_lr_backbone}, head_lr={args.ast_lr_head}"
    )
    default_lr = args.ast_lr_head

if args.model != "AST":
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    default_lr = (
        args.lr if args.lr != 1e-3 else (1e-4 if args.model == "AudioViT" else 1e-3)
    )
    lr = default_lr
else:
    lr = args.ast_lr_head

# Create base scheduler (use CosineAnnealing for AST, StepLR for others)
if args.model == "AST":
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    print("Using CosineAnnealingLR scheduler for AST")
else:
    base_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Wrap with warmup scheduler if requested
if args.warmup_epochs > 0:
    scheduler = WarmupScheduler(
        optimizer,
        base_scheduler=base_scheduler,
        warmup_epochs=args.warmup_epochs,
        base_lr=lr,
    )
    print(f"\nUsing warmup scheduler: {args.warmup_epochs} epochs, base_lr={lr}")
else:
    scheduler = base_scheduler

# Create loss function
loss_fn = None
if args.loss_type == "focal":
    loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    print(f"\nUsing Focal Loss: gamma={args.focal_gamma}, alpha={args.focal_alpha}")
elif args.use_class_weights:
    # class_weights will be passed to Trainer, which creates CrossEntropyLoss with weights
    print(f"\nUsing CrossEntropyLoss with class weights ({args.weight_method})")
else:
    print("\nUsing standard CrossEntropyLoss")

# Determine checkpoint directory
save_name = args.save_name if args.save_name else args.model.lower()
checkpoint_dir = MODELS_DIR / save_name

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    checkpoint_dir=checkpoint_dir,
    experiment_name=args.model,
    use_amp=True,
    gradient_clip=1.0,
    early_stopping_patience=7,
    class_weights=class_weights
    if not loss_fn
    else None,  # Only use if not using focal loss
    loss_fn=loss_fn,  # Pass custom loss function
)

history = trainer.train(num_epochs=args.epochs)

# Save training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(history["train_loss"], label="Train")
axes[0, 0].plot(history["val_loss"], label="Val")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title(f"{args.model} - Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history["train_acc"], label="Train")
axes[0, 1].plot(history["val_acc"], label="Val")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title(f"{args.model} - Accuracy")
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
plt.savefig(checkpoint_dir / "training_curves.png", dpi=150)
plt.close()

# Save history
with open(checkpoint_dir / "history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

print(f"\n✓ {args.model} training complete")
print(f"✓ Best val accuracy: {max(history['val_acc']):.4f}")
print(f"✓ Checkpoint saved to {checkpoint_dir}")
