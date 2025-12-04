"""Species name normalization and matching utilities.

This module provides functions to normalize bird species names across different
datasets, handling variations in nomenclature, authorship, and formatting.
"""

import re
from typing import Dict, Set
import pandas as pd


def normalize_species_name(name: str) -> str:
    """Normalize a species name to a canonical form.

    Steps:
    1. Convert to lowercase
    2. Remove authorship strings (content in parentheses)
    3. Strip punctuation except hyphens
    4. Collapse multiple spaces to single space
    5. Unify hyphens and underscores to spaces
    6. Strip leading/trailing whitespace

    Args:
        name: Raw species name

    Returns:
        Normalized species name

    Examples:
        >>> normalize_species_name("Parus major (Linnaeus, 1758)")
        'parus major'
        >>> normalize_species_name("Red-winged_Blackbird")
        'red winged blackbird'
    """
    if not isinstance(name, str):
        return ""

    # Lowercase
    name = name.lower()

    # Remove authorship (content in parentheses)
    name = re.sub(r"\([^)]*\)", "", name)

    # Remove special punctuation but keep hyphens temporarily
    name = re.sub(r"[^\w\s-]", " ", name)

    # Convert hyphens and underscores to spaces
    name = name.replace("-", " ").replace("_", " ")

    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)

    # Strip
    name = name.strip()

    return name


def build_species_mapping(
    df: pd.DataFrame, species_col: str = "species"
) -> Dict[str, str]:
    """Build a mapping from normalized species names to original names.

    Args:
        df: DataFrame containing species names
        species_col: Column name containing species names

    Returns:
        Dictionary mapping normalized_name -> original_name
    """
    mapping = {}
    for species in df[species_col].unique():
        if pd.notna(species):
            normalized = normalize_species_name(species)
            if normalized:
                # Prefer longer/more complete original names
                if normalized not in mapping or len(species) > len(mapping[normalized]):
                    mapping[normalized] = species
    return mapping


def normalize_species_names(
    xeno_canto_df: pd.DataFrame, cub_df: pd.DataFrame
) -> Dict[str, Dict[str, str]]:
    """Normalize species names across all datasets and create mappings.

    Args:
        xeno_canto_df: Xeno-Canto index DataFrame
        cub_df: CUB-200 index DataFrame

    Returns:
        Dictionary with keys 'xeno_canto', 'cub' containing
        normalized -> original mappings for each dataset
    """
    # Build mappings for each dataset
    mappings = {
        "xeno_canto": build_species_mapping(xeno_canto_df, "species"),
        "cub": build_species_mapping(cub_df, "species"),
    }

    # Add normalized names to DataFrames (in-place)
    for df_name, df in [
        ("xeno_canto", xeno_canto_df),
        ("cub", cub_df),
    ]:
        if "species" in df.columns:
            df["species_normalized"] = df["species"].apply(normalize_species_name)

    return mappings


def compute_intersection(
    xeno_canto_df: pd.DataFrame,
    cub_df: pd.DataFrame,
    species_mapping: Dict[str, Dict[str, str]],
) -> Set[str]:
    """Compute the intersection of species between Xeno-Canto and CUB datasets.

    Args:
        xeno_canto_df: Xeno-Canto index with 'species_normalized' column
        cub_df: CUB index with 'species_normalized' column
        species_mapping: Species mappings from normalize_species_names()

    Returns:
        Set of normalized species names present in both datasets
    """
    if "species_normalized" not in xeno_canto_df.columns:
        xeno_canto_df["species_normalized"] = xeno_canto_df["species"].apply(
            normalize_species_name
        )

    if "species_normalized" not in cub_df.columns:
        cub_df["species_normalized"] = cub_df["species"].apply(normalize_species_name)

    xc_species = set(xeno_canto_df["species_normalized"].dropna().unique())
    cub_species = set(cub_df["species_normalized"].dropna().unique())

    # Remove empty strings
    xc_species.discard("")
    cub_species.discard("")

    intersection = xc_species & cub_species

    return intersection


def get_species_stats(df: pd.DataFrame, species_col: str = "species") -> pd.DataFrame:
    """Get statistics about species distribution in a dataset.

    Args:
        df: DataFrame with species information
        species_col: Column name for species

    Returns:
        DataFrame with species counts and percentages
    """
    stats = df[species_col].value_counts().reset_index()
    stats.columns = ["species", "count"]
    stats["percentage"] = (stats["count"] / len(df) * 100).round(2)
    return stats
