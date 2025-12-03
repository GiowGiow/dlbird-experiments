"""PyTorch dataset for image data."""

from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class ImageDataset(Dataset):
    """Dataset for loading bird images with preprocessing transforms.

    Loads images from file paths and applies transformations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: List[int],
        species_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        species_col: str = "species_normalized",
    ):
        """
        Args:
            df: DataFrame with image metadata
            indices: List of dataframe indices to use for this split
            species_to_idx: Mapping from species name to class index
            transform: torchvision transforms to apply to images
            species_col: Column containing species labels
        """
        self.df = df.iloc[indices].reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.transform = transform
        self.species_col = species_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load image and return tensor and label.

        Returns:
            image: (3, H, W) tensor
            label: Integer class label
        """
        row = self.df.iloc[idx]

        # Load image
        image_path = row["file_path"]
        image = Image.open(image_path).convert("RGB")

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Get label
        species = (
            row[self.species_col]
            if self.species_col in row
            else row.get("species", "unknown")
        )
        label = self.species_to_idx[species]

        return image, label


def get_image_transforms(train: bool = True, image_size: int = 224):
    """Get standard image transforms for training or evaluation.

    Args:
        train: If True, return training transforms with augmentation
        image_size: Target image size

    Returns:
        torchvision.transforms.Compose object
    """
    from torchvision import transforms

    # ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),  # Resize to 256 for 224
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
