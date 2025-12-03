"""CUB-200-2011 dataset indexing.

This module provides functions to parse and index the CUB-200-2011 bird images dataset.
"""

from pathlib import Path
from typing import Union
import pandas as pd


def index_cub(root_path: Union[str, Path]) -> pd.DataFrame:
    """Index CUB-200-2011 image dataset.

    The function reads the CUB-200 metadata files:
    - images.txt: image_id and relative path
    - classes.txt: class_id and class name
    - image_class_labels.txt: image_id to class_id mapping

    Args:
        root_path: Root directory of CUB-200-2011 dataset

    Returns:
        DataFrame with columns:
            - image_id: Unique image identifier
            - class_id: Numeric class identifier
            - species: Species name (from class name)
            - file_path: Absolute path to image file
    """
    root_path = Path(root_path)

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root_path}")

    # Find metadata directory (usually in root or in a subdirectory)
    metadata_candidates = [
        root_path,
        root_path / "CUB_200_2011",
        root_path / "CUB-200-2011",
    ]

    metadata_dir = None
    for candidate in metadata_candidates:
        if (candidate / "images.txt").exists():
            metadata_dir = candidate
            break

    if metadata_dir is None:
        # Search recursively
        images_txt_files = list(root_path.rglob("images.txt"))
        if images_txt_files:
            metadata_dir = images_txt_files[0].parent
        else:
            raise FileNotFoundError(f"Could not find CUB metadata files in {root_path}")

    print(f"Found metadata in: {metadata_dir}")

    # Read images.txt
    images_path = metadata_dir / "images.txt"
    images_df = pd.read_csv(
        images_path, sep=" ", names=["image_id", "image_path"], header=None
    )

    # Read classes.txt
    classes_path = metadata_dir / "classes.txt"
    classes_df = pd.read_csv(
        classes_path, sep=" ", names=["class_id", "class_name"], header=None
    )

    # Read image_class_labels.txt
    labels_path = metadata_dir / "image_class_labels.txt"
    labels_df = pd.read_csv(
        labels_path, sep=" ", names=["image_id", "class_id"], header=None
    )

    # Merge dataframes
    df = images_df.merge(labels_df, on="image_id")
    df = df.merge(classes_df, on="class_id")

    # Extract species name from class name (format: "001.Black_footed_Albatross")
    df["species"] = (
        df["class_name"].str.replace(r"^\d+\.", "", regex=True).str.replace("_", " ")
    )

    # Find images directory
    images_dir_candidates = [
        metadata_dir / "images",
        metadata_dir.parent / "images",
        root_path / "images",
    ]

    images_dir = None
    for candidate in images_dir_candidates:
        if candidate.exists() and candidate.is_dir():
            images_dir = candidate
            break

    if images_dir is None:
        # Search for any directory containing many image files
        for subdir in root_path.rglob("*"):
            if subdir.is_dir():
                image_count = len(list(subdir.glob("*.jpg"))) + len(
                    list(subdir.glob("*.png"))
                )
                if image_count > 100:  # CUB has ~12k images
                    images_dir = subdir
                    break

    if images_dir is None:
        raise FileNotFoundError(f"Could not find images directory in {root_path}")

    print(f"Found images in: {images_dir}")

    # Build full file paths
    df["file_path"] = df["image_path"].apply(lambda x: str((images_dir / x).absolute()))

    # Verify files exist (sample check)
    sample_files = df["file_path"].head(10)
    existing_count = sum(1 for f in sample_files if Path(f).exists())
    if existing_count == 0:
        raise FileNotFoundError(
            "Image files not found at expected paths. Check dataset structure."
        )

    # Select output columns
    df = df[["image_id", "class_id", "species", "file_path"]]

    print(f"Indexed {len(df)} images")
    print(f"Unique classes: {df['class_id'].nunique()}")
    print(f"Unique species: {df['species'].nunique()}")

    return df
