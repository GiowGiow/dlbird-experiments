"""Stratified dataset splitting utilities."""

from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_stratified_splits(
    df: pd.DataFrame,
    species_col: str = "species",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Dict[str, List[int]]:
    """Create stratified train/val/test splits ensuring balanced species distribution.

    Args:
        df: DataFrame to split
        species_col: Column name containing species labels
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing row indices
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    # Get species labels
    y = df[species_col].values
    indices = np.arange(len(df))

    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, stratify=y, random_state=random_state
    )

    # Second split: separate train and val from remaining data
    train_val_labels = y[train_val_indices]
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size_adjusted,
        stratify=train_val_labels,
        random_state=random_state,
    )

    splits = {
        "train": train_indices.tolist(),
        "val": val_indices.tolist(),
        "test": test_indices.tolist(),
    }

    # Verify splits
    print(
        f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    # Check species distribution
    for split_name, split_indices in splits.items():
        split_species = df.iloc[split_indices][species_col].nunique()
        print(f"  {split_name}: {split_species} unique species")

    return splits


def verify_no_leakage(
    splits: Dict[str, List[int]], df: pd.DataFrame, id_col: str = "record_id"
) -> bool:
    """Verify no sample leakage between splits.

    Args:
        splits: Dictionary of split indices
        df: DataFrame
        id_col: Column to check for uniqueness

    Returns:
        True if no leakage detected
    """
    train_ids = set(df.iloc[splits["train"]][id_col])
    val_ids = set(df.iloc[splits["val"]][id_col])
    test_ids = set(df.iloc[splits["test"]][id_col])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap:
        print(
            f"WARNING: {len(train_val_overlap)} samples overlap between train and val"
        )
        return False

    if train_test_overlap:
        print(
            f"WARNING: {len(train_test_overlap)} samples overlap between train and test"
        )
        return False

    if val_test_overlap:
        print(f"WARNING: {len(val_test_overlap)} samples overlap between val and test")
        return False

    print("No leakage detected between splits")
    return True
