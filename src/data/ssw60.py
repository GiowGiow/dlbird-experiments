"""SSW60 dataset extraction and indexing.

This module provides functions to extract and index the SSW60 bird dataset
containing both images and audio files.
"""

import tarfile
from pathlib import Path
from typing import Union
import pandas as pd


def extract_and_index_ssw60(
    tarball_path: Union[str, Path], extract_to: Union[str, Path]
) -> pd.DataFrame:
    """Extract SSW60 tarball and index images and audio files.

    Args:
        tarball_path: Path to ssw60.tar.gz file
        extract_to: Directory to extract files to

    Returns:
        DataFrame with columns:
            - sample_id: Unique sample identifier
            - species: Species name
            - modality: 'image' or 'audio'
            - file_path: Absolute path to file
    """
    tarball_path = Path(tarball_path)
    extract_to = Path(extract_to)

    if not tarball_path.exists():
        raise FileNotFoundError(f"Tarball not found: {tarball_path}")

    # Extract if not already extracted
    if not extract_to.exists() or len(list(extract_to.iterdir())) == 0:
        print(f"Extracting {tarball_path} to {extract_to}...")
        extract_to.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(extract_to)

        print("Extraction complete")
    else:
        print(f"Dataset already extracted at {extract_to}")

    # Index files
    records = []

    # Look for images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in image_extensions:
        for image_file in extract_to.rglob(f"*{ext}"):
            # Try to infer species from directory or file name
            species = infer_species_from_path(image_file)
            records.append(
                {
                    "sample_id": image_file.stem,
                    "species": species,
                    "modality": "image",
                    "file_path": str(image_file.absolute()),
                }
            )

    # Look for audio
    audio_extensions = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]
    for ext in audio_extensions:
        for audio_file in extract_to.rglob(f"*{ext}"):
            species = infer_species_from_path(audio_file)
            records.append(
                {
                    "sample_id": audio_file.stem,
                    "species": species,
                    "modality": "audio",
                    "file_path": str(audio_file.absolute()),
                }
            )

    df = pd.DataFrame(records)

    print(f"Indexed {len(df)} samples")
    print(f"  Images: {len(df[df['modality'] == 'image'])}")
    print(f"  Audio: {len(df[df['modality'] == 'audio'])}")
    print(f"  Unique species: {df['species'].nunique()}")

    return df


def infer_species_from_path(file_path: Path) -> str:
    """Infer species name from file or directory path.

    SSW60 typically organizes files as: species_name/file.ext
    or includes species in filename.

    Args:
        file_path: Path to file

    Returns:
        Species name (best guess)
    """
    # Try parent directory name first
    parent = file_path.parent.name
    if parent and parent not in ["image", "audio", "images", "audios", "data", "ssw60"]:
        return parent.replace("_", " ").replace("-", " ")

    # Try grandparent
    grandparent = file_path.parent.parent.name
    if grandparent and grandparent not in [
        "image",
        "audio",
        "images",
        "audios",
        "data",
        "ssw60",
    ]:
        return grandparent.replace("_", " ").replace("-", " ")

    # Try filename
    name = file_path.stem
    # Remove common suffixes like numbers
    import re

    name = re.sub(r"[_-]\d+$", "", name)
    name = name.replace("_", " ").replace("-", " ")

    return name
