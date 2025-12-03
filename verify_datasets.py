#!/usr/bin/env python3
"""Verify dataset paths and accessibility."""

import sys
from pathlib import Path


def check_path(name: str, path: Path) -> bool:
    """Check if a path exists and is accessible."""
    print(f"\nChecking {name}:")
    print(f"  Path: {path}")

    if not path.exists():
        print(f"  ✗ Does not exist")
        return False

    if not path.is_dir():
        if path.suffix in [".tar", ".gz", ".zip"]:
            print(f"  ✓ Archive file exists")
            print(f"  Size: {path.stat().st_size / (1024**3):.2f} GB")
            return True
        else:
            print(f"  ✗ Not a directory")
            return False

    # Count files
    files = list(path.rglob("*"))
    file_count = len([f for f in files if f.is_file()])
    dir_count = len([f for f in files if f.is_dir()])

    print(f"  ✓ Directory exists")
    print(f"  Files: {file_count:,}")
    print(f"  Subdirectories: {dir_count:,}")

    return True


def main():
    """Check all dataset paths."""
    print("=" * 60)
    print("Dataset Path Verification")
    print("=" * 60)

    datasets = {
        "Xeno-Canto A-M v11": Path(
            "/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m/versions/11"
        ),
        "Xeno-Canto N-Z v11": Path(
            "/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/rohanrao/xeno-canto-bird-recordings-extended-n-z/versions/11"
        ),
        "CUB-200-2011 v7": Path(
            "/media/giovanni/TOSHIBA EXT/dlbird/datasets/datasets/wenewone/cub2002011/versions/7"
        ),
        "SSW60 tarball": Path(
            "/media/giovanni/TOSHIBA EXT/dlbird/datasets/mixed/ssw60.tar.gz"
        ),
    }

    results = {}
    for name, path in datasets.items():
        results[name] = check_path(name, path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_good = True
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")
        if not status:
            all_good = False

    if all_good:
        print("\n✓ All datasets accessible!")
        print("\nYou can now run the notebooks:")
        print("  1. jupyter notebook notebooks/")
        print("  2. Select kernel: Python (speckitdlbird)")
        print("  3. Run notebooks 00 through 05 in order")
        return 0
    else:
        print("\n✗ Some datasets not accessible")
        print("\nPlease:")
        print("  1. Mount external drive if needed")
        print("  2. Update paths in notebooks to match your setup")
        print("  3. Run this script again to verify")
        return 1


if __name__ == "__main__":
    sys.exit(main())
