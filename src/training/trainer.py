"""Training loop with AMP, checkpointing, and early stopping."""

from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json


class WarmupScheduler:
    """Learning rate warmup wrapper for any scheduler.

    Linearly increases learning rate from lr/10 to lr over warmup_epochs,
    then delegates to the base scheduler.

    Args:
        optimizer: PyTorch optimizer
        base_scheduler: Base scheduler to use after warmup (can be None)
        warmup_epochs: Number of epochs for warmup
        base_lr: Base learning rate (taken from optimizer if None)
    """

    def __init__(self, optimizer, base_scheduler=None, warmup_epochs=5, base_lr=None):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.current_epoch = 0

    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup: lr_min + (lr_max - lr_min) * current / warmup
            lr_min = self.base_lr / 10.0
            lr = (
                lr_min
                + (self.base_lr - lr_min)
                * (self.current_epoch + 1)
                / self.warmup_epochs
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        elif self.base_scheduler is not None:
            # After warmup, use base scheduler
            self.base_scheduler.step()

        self.current_epoch += 1

    def get_last_lr(self):
        """Get current learning rate."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def state_dict(self):
        """Get scheduler state for checkpointing."""
        state = {
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "base_lr": self.base_lr,
        }
        if self.base_scheduler is not None and hasattr(
            self.base_scheduler, "state_dict"
        ):
            state["base_scheduler"] = self.base_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state_dict.get("current_epoch", 0)
        self.warmup_epochs = state_dict.get("warmup_epochs", self.warmup_epochs)
        self.base_lr = state_dict.get("base_lr", self.base_lr)
        if "base_scheduler" in state_dict and self.base_scheduler is not None:
            if hasattr(self.base_scheduler, "load_state_dict"):
                self.base_scheduler.load_state_dict(state_dict["base_scheduler"])


class Trainer:
    """Unified trainer for all models with AMP and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
        experiment_name: str = "experiment",
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name for this experiment
            use_amp: Use automatic mixed precision
            gradient_clip: Max gradient norm (0 to disable)
            early_stopping_patience: Epochs without improvement before stopping
            class_weights: Optional class weights for CrossEntropyLoss (tensor of shape [num_classes])
                          Only used if loss_fn is None
            loss_fn: Custom loss function (e.g., FocalLoss). If None, uses CrossEntropyLoss
            mixup_alpha: Alpha parameter for MixUp Beta distribution (0 to disable)
            mixup_prob: Probability of applying MixUp to a batch
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

        # Create criterion: use provided loss_fn or default to CrossEntropyLoss
        if loss_fn is not None:
            self.criterion = loss_fn
        elif class_weights is not None:
            self.class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Tracking
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": running_loss / total, "acc": 100.0 * correct / total}
            )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return {"loss": epoch_loss, "acc": epoch_acc}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.val_loader, desc="Validation"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return {"loss": epoch_loss, "acc": epoch_acc}

    def train(self, num_epochs: int) -> Dict:
        """Train for multiple epochs with checkpointing and early stopping.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Training history dictionary
        """
        print(f"Training {self.experiment_name} for {num_epochs} epochs...")
        print(f"Device: {self.device}, AMP: {self.use_amp}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])

            # Validate
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Print metrics
            print(
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}%"
            )
            print(
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}%"
            )

            # Checkpointing
            if val_metrics["acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["acc"]
                self.epochs_without_improvement = 0

                if self.checkpoint_dir:
                    self.save_checkpoint(epoch, val_metrics["acc"])
                    print(f"✓ Saved checkpoint (Val Acc: {val_metrics['acc']:.2f}%)")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"\nEarly stopping after {epoch + 1} epochs (no improvement for {self.early_stopping_patience} epochs)"
                )
                break

        print(f"\nTraining complete! Best Val Acc: {self.best_val_acc:.2f}%")

        # Save history
        if self.checkpoint_dir:
            history_path = self.checkpoint_dir / f"{self.experiment_name}_history.json"
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)

        return self.history

    def save_checkpoint(self, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "history": self.history,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.history = checkpoint.get("history", self.history)
        self.best_val_acc = checkpoint.get("val_acc", 0.0)

        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_acc']:.2f}%)"
        )
