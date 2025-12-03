"""Xeno-Canto dataset indexing.

This module provides functions to parse and index the Xeno-Canto bird recordings
dataset (extended A-M and N-Z versions).
"""

import json
from pathlib import Path
from typing import Union, List
import pandas as pd


def find_metadata_file(root_path: Path) -> Path:
    """Find the metadata CSV or JSON file in the Xeno-Canto directory.

    Args:
        root_path: Root directory of the Xeno-Canto dataset

    Returns:
        Path to metadata file

    Raises:
        FileNotFoundError: If no metadata file is found
    """
    # Look for common metadata file patterns
    patterns = ["*.csv", "*.json", "metadata.csv", "metadata.json", "birds.csv"]

    for pattern in patterns:
        files = list(root_path.glob(pattern))
        if files:
            return files[0]

    # Search recursively
    for pattern in ["**/*.csv", "**/*.json"]:
        files = list(root_path.glob(pattern))
        if files:
            # Prefer files with 'metadata' or 'birds' in name
            for f in files:
                if "metadata" in f.name.lower() or "birds" in f.name.lower():
                    return f
            # Otherwise return first found
            return files[0]

    raise FileNotFoundError(f"No metadata file found in {root_path}")


def index_xeno_canto(root_path: Union[str, Path]) -> pd.DataFrame:
    """Index Xeno-Canto audio recordings and metadata.

    The function looks for metadata files (CSV or JSON) and creates an index
    mapping species names to audio file paths.

    Args:
        root_path: Root directory of the Xeno-Canto dataset

    Returns:
        DataFrame with columns:
            - record_id: Unique recording identifier
            - species: Species name (scientific or common)
            - file_path: Absolute path to audio file
            - duration: Recording duration in seconds (if available)
            - sampling_rate: Sampling rate in Hz (if available)
            - quality: Quality rating (if available)
    """
    root_path = Path(root_path)

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root_path}")

    # Find metadata file
    metadata_path = find_metadata_file(root_path)
    print(f"Found metadata: {metadata_path}")

    # Load metadata
    if metadata_path.suffix == ".csv":
        df = pd.read_csv(metadata_path)
    elif metadata_path.suffix == ".json":
        df = pd.read_json(metadata_path)
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")

    # Standardize column names (case-insensitive matching)
    # Use priority order to avoid duplicate mappings
    col_mapping = {}

    # Find record_id (prefer xc_id or columns with just 'id')
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["xc_id", "id", "record_id"] or (col_lower == "id"):
            if "record_id" not in col_mapping.values():
                col_mapping[col] = "record_id"
                break

    # Find species (prefer 'species' over 'sci_name')
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "species":
            col_mapping[col] = "species"
            break
        elif "sci" in col_lower and "name" in col_lower:
            if "species" not in col_mapping.values():
                col_mapping[col] = "species"

    # Find filename (prefer 'filename' over 'file_type')
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["filename", "file_name", "file"]:
            col_mapping[col] = "file_name"
            break

    # Find duration
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["duration", "length"]:
            col_mapping[col] = "duration"
            break

    # Find sampling_rate (prefer exact match)
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["sampling_rate", "sample_rate", "sr"]:
            col_mapping[col] = "sampling_rate"
            break

    # Find quality/rating
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["quality", "rating"]:
            col_mapping[col] = "quality"
            break

    df = df.rename(columns=col_mapping)

    # Ensure required columns exist
    if "record_id" not in df.columns:
        df["record_id"] = range(len(df))

    if "species" not in df.columns:
        # Try to find any column that might contain species info
        for col in df.columns:
            if "name" in col.lower() or "bird" in col.lower():
                df["species"] = df[col]
                break
        else:
            raise ValueError("No species column found in metadata")

    # Find audio files
    audio_extensions = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(root_path.rglob(f"*{ext}"))

    print(f"Found {len(audio_files)} audio files")

    # Build file path mapping
    if "file_name" in df.columns or "filename" in df.columns:
        # Try both common column names
        file_col = "file_name" if "file_name" in df.columns else "filename"

        print(f"Using column: {file_col}, DataFrame shape: {df.shape}")

        # Create a lookup dict from filename to full path
        file_lookup = {}
        for audio_file in audio_files:
            file_lookup[audio_file.name] = str(audio_file.absolute())
            # Also add without extension
            file_lookup[audio_file.stem] = str(audio_file.absolute())

        # Map file names to full paths using list comprehension
        file_paths = []
        for filename in df[file_col].values:  # Use .values to get numpy array
            if pd.isna(filename):
                file_paths.append(None)
            else:
                path_obj = Path(str(filename))
                # Try full name first, then stem
                file_paths.append(
                    file_lookup.get(path_obj.name, file_lookup.get(path_obj.stem, None))
                )

        print(f"Built {len(file_paths)} file paths")
        df["file_path"] = file_paths
    else:
        # If no file column, try to match by record_id
        file_lookup = {f.stem: str(f.absolute()) for f in audio_files}
        df["file_path"] = (
            df["record_id"].astype(str).apply(lambda x: file_lookup.get(x, None))
        )

    # Filter out rows without valid file paths
    df = df[df["file_path"].notna()].copy()

    # Select and order columns
    output_cols = ["record_id", "species", "file_path"]
    for optional_col in ["duration", "sampling_rate", "quality"]:
        if optional_col in df.columns:
            output_cols.append(optional_col)

    df = df[output_cols]

    print(f"Indexed {len(df)} recordings with valid file paths")
    print(f"Unique species: {df['species'].nunique()}")

    return df
