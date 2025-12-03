"""PyTorch dataset for cached MFCC features."""

from pathlib import Path
from typing import Union, List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class AudioMFCCDataset(Dataset):
    """Dataset for loading cached MFCC features.

    Features are expected to be cached as numpy arrays in format:
    cache_dir/species/record_id.npy

    Each array should have shape (H, W, 3) where:
    - H = n_mfcc (number of coefficients)
    - W = number of time frames
    - 3 = [MFCC, Delta, Delta-Delta]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Union[str, Path],
        indices: List[int],
        species_to_idx: Dict[str, int],
        transform=None,
    ):
        """
        Args:
            df: DataFrame with audio metadata
            cache_dir: Directory containing cached features
            indices: List of dataframe indices to use for this split
            species_to_idx: Mapping from species name to class index
            transform: Optional transform to apply to features
        """
        self.df = df.iloc[indices].reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.species_to_idx = species_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load cached features and return tensor and label.

        Returns:
            features: (3, H, W) tensor (transposed to channels-first)
            label: Integer class label
        """
        row = self.df.iloc[idx]

        # Get species and record ID
        species = row.get("species_normalized", row.get("species", "unknown"))
        record_id = row.get("record_id", row.get("sample_id", idx))

        # Build cache path
        species_safe = str(species).replace("/", "_").replace(" ", "_")
        cache_file = self.cache_dir / species_safe / f"{record_id}.npy"

        # Load cached features
        if not cache_file.exists():
            raise FileNotFoundError(f"Cached features not found: {cache_file}")

        features = np.load(cache_file)  # (H, W, 3)

        # Transpose to (3, H, W) for PyTorch
        features = np.transpose(features, (2, 0, 1))

        # Convert to tensor
        features = torch.from_numpy(features).float()

        # Apply transform if provided
        if self.transform:
            features = self.transform(features)

        # Get label
        label = self.species_to_idx[species]

        return features, label


def create_species_mapping(
    df: pd.DataFrame, species_col: str = "species_normalized"
) -> Dict[str, int]:
    """Create mapping from species names to integer indices.

    Args:
        df: DataFrame with species information
        species_col: Column containing species names

    Returns:
        Dictionary mapping species name to integer index
    """
    unique_species = sorted(df[species_col].unique())
    return {species: idx for idx, species in enumerate(unique_species)}


def get_num_classes(species_to_idx: Dict[str, int]) -> int:
    """Get number of classes from species mapping.

    Args:
        species_to_idx: Species to index mapping

    Returns:
        Number of unique classes
    """
    return len(species_to_idx)
