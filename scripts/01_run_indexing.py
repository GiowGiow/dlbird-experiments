"""Direct script to index datasets - bypassing notebook caching issues"""

import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xeno_canto import index_xeno_canto
from src.data.cub import index_cub
from src.utils.species import normalize_species_name

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Dataset paths
XENO_CANTO_AM = Path(
    "/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m/versions/11"
)
XENO_CANTO_NZ = Path(
    "/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/rohanrao/xeno-canto-bird-recordings-extended-n-z/versions/11"
)
CUB_ROOT = Path(
    "/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/wenewone/cub2002011/versions/7"
)

print("=" * 80)
print("TASK 3: INDEX XENO-CANTO")
print("=" * 80)

print("\nIndexing Xeno-Canto A-M...")
xc_am_df = index_xeno_canto(XENO_CANTO_AM)
print(f"Found {len(xc_am_df)} recordings in A-M")

print("\nIndexing Xeno-Canto N-Z...")
xc_nz_df = index_xeno_canto(XENO_CANTO_NZ)
print(f"Found {len(xc_nz_df)} recordings in N-Z")

print("\nCombining datasets...")
xc_full = pd.concat([xc_am_df, xc_nz_df], ignore_index=True)
print(f"Total Xeno-Canto recordings: {len(xc_full)}")
print(f"Unique species: {xc_full['species'].nunique()}")

# Save
xc_full.to_parquet(ARTIFACTS / "xeno_canto_index.parquet", index=False)
print(f"\n✓ Saved to {ARTIFACTS / 'xeno_canto_index.parquet'}")

print("\n" + "=" * 80)
print("TASK 4: INDEX CUB-200-2011")
print("=" * 80)

print("\nIndexing CUB-200-2011...")
cub_df = index_cub(CUB_ROOT)
print(f"Found {len(cub_df)} images")
print(f"Unique species: {cub_df['species'].nunique()}")

# Save
cub_df.to_parquet(ARTIFACTS / "cub_index.parquet", index=False)
print(f"\n✓ Saved to {ARTIFACTS / 'cub_index.parquet'}")

print("\n" + "=" * 80)
print("TASK 5: NORMALIZE SPECIES NAMES")
print("=" * 80)

# Normalize Xeno-Canto species
print("\nNormalizing Xeno-Canto species...")
xc_full["species_normalized"] = xc_full["species"].apply(normalize_species_name)
print(f"Unique normalized species: {xc_full['species_normalized'].nunique()}")
xc_full.to_parquet(ARTIFACTS / "xeno_canto_index.parquet", index=False)

# Normalize CUB species
print("\nNormalizing CUB species...")
cub_df["species_normalized"] = cub_df["species"].apply(normalize_species_name)
print(f"Unique normalized species: {cub_df['species_normalized'].nunique()}")
cub_df.to_parquet(ARTIFACTS / "cub_index.parquet", index=False)

print("\n" + "=" * 80)
print("TASK 6: COMPUTE SPECIES INTERSECTION")
print("=" * 80)

# Find intersection
xc_species = set(xc_full["species_normalized"].unique())
cub_species = set(cub_df["species_normalized"].unique())

intersection = xc_species & cub_species
print(f"\nXeno-Canto species: {len(xc_species)}")
print(f"CUB species: {len(cub_species)}")
print(f"Intersection: {len(intersection)}")

# Filter datasets to intersection
print("\nFiltering to intersection...")
xc_filtered = xc_full[xc_full["species_normalized"].isin(intersection)].copy()
cub_filtered = cub_df[cub_df["species_normalized"].isin(intersection)].copy()

print(
    f"Xeno-Canto filtered: {len(xc_filtered)} recordings, {xc_filtered['species_normalized'].nunique()} species"
)
print(
    f"CUB filtered: {len(cub_filtered)} images, {cub_filtered['species_normalized'].nunique()} species"
)

# Save filtered versions
xc_filtered.to_parquet(ARTIFACTS / "xeno_canto_filtered.parquet", index=False)
cub_filtered.to_parquet(ARTIFACTS / "cub_filtered.parquet", index=False)

# Save intersection metadata
intersection_metadata = {
    "xeno_canto_total_species": len(xc_species),
    "cub_total_species": len(cub_species),
    "intersection_species": sorted(list(intersection)),
    "intersection_count": len(intersection),
    "xeno_canto_filtered_count": len(xc_filtered),
    "cub_filtered_count": len(cub_filtered),
}

with open(ARTIFACTS / "intersection_metadata.json", "w") as f:
    json.dump(intersection_metadata, f, indent=2)

print("\n✓ Saved filtered datasets and intersection metadata")
print("\nSample intersecting species:")
for sp in sorted(list(intersection))[:10]:
    xc_count = (xc_filtered["species_normalized"] == sp).sum()
    cub_count = (cub_filtered["species_normalized"] == sp).sum()
    print(f"  {sp}: {xc_count} audio, {cub_count} images")

print("\n" + "=" * 80)
print("✓ TASKS 3-6 COMPLETE")
print("=" * 80)
