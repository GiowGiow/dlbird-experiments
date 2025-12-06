#!/usr/bin/env python3
"""
Generate and cache Log-Mel Spectrograms for all audio files.

This script pre-computes spectrograms for all 11,075 recordings,
enabling faster training with no on-the-fly feature extraction overhead.

Usage:
    python scripts/generate_lms_cache.py [--workers 8] [--force-regenerate]
"""

import argparse
from pathlib import Path
import sys
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from functools import partial

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.log_mel_spectrogram import (
    extract_segment_with_activity_detection,
    validate_spectrogram,
)
import numpy as np


def extract_and_validate(
    args: tuple,
    target_duration: float = 4.0,
    sr: int = 22050,
    n_mels: int = 128,
) -> dict:
    """
    Extract LMS for a single audio file and validate.
    
    Parameters
    ----------
    args : tuple
        (idx, row, cache_dir, audio_dir, force_regenerate)
        
    Returns
    -------
    result : dict
        Status and metadata for this extraction
    """
    idx, row, cache_dir, audio_dir, force_regenerate = args
    
    xc_id = row['record_id']
    species = row['species_normalized']
    
    # Paths
    species_dir = cache_dir / species
    species_dir.mkdir(parents=True, exist_ok=True)
    cache_path = species_dir / f"{xc_id}.npy"
    
    # Skip if cached and not forcing regeneration
    if cache_path.exists() and not force_regenerate:
        # Validate cached file
        try:
            lms = np.load(cache_path)
            is_valid, msg = validate_spectrogram(lms, expected_mels=n_mels)
            if is_valid:
                return {
                    'status': 'cached',
                    'xc_id': xc_id,
                    'species': species,
                    'shape': lms.shape,
                }
            else:
                # Invalid cache, regenerate
                pass
        except Exception:
            # Corrupt cache, regenerate
            pass
    
    # Find audio file - use file_path from DataFrame
    audio_path = Path(row.get('file_path', ''))
    
    if not audio_path.exists():
        # Try fallback locations
        for possible_path in [
            audio_dir / f"{xc_id}.mp3",
            Path('data/xeno_canto') / f"{xc_id}.mp3",
            Path('data/xeno_canto_raw') / f"{xc_id}.mp3",
        ]:
            if possible_path.exists():
                audio_path = possible_path
                break
    
    if not audio_path.exists():
        return {
            'status': 'missing_audio',
            'xc_id': xc_id,
            'species': species,
        }
    
    # Extract LMS
    try:
        lms = extract_segment_with_activity_detection(
            audio_path,
            target_duration=target_duration,
            sr=sr,
            n_mels=n_mels,
        )
        
        # Validate
        is_valid, msg = validate_spectrogram(lms, expected_mels=n_mels)
        if not is_valid:
            return {
                'status': 'invalid',
                'xc_id': xc_id,
                'species': species,
                'error': msg,
            }
        
        # Save to cache
        np.save(cache_path, lms)
        
        return {
            'status': 'success',
            'xc_id': xc_id,
            'species': species,
            'shape': lms.shape,
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'xc_id': xc_id,
            'species': species,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Generate Log-Mel Spectrogram cache')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes')
    parser.add_argument('--force-regenerate', action='store_true', help='Regenerate all caches')
    parser.add_argument('--target-duration', type=float, default=4.0, help='Target audio duration (seconds)')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate (Hz)')
    parser.add_argument('--n-mels', type=int, default=128, help='Number of mel bins')
    args = parser.parse_args()
    
    print("="*80)
    print("Log-Mel Spectrogram Cache Generation")
    print("="*80)
    
    # Paths
    artifacts = Path('artifacts')
    cache_dir = artifacts / 'audio_lms_cache' / 'xeno_canto'
    audio_dir = Path('data/xeno_canto_raw')
    
    # Load dataset
    print("\n📂 Loading dataset...")
    xc_df = pd.read_parquet(artifacts / 'xeno_canto_filtered.parquet')
    
    # Filter species with at least 2 samples
    xc_counts = xc_df['species_normalized'].value_counts()
    species_to_keep = xc_counts[xc_counts >= 2].index
    xc_df = xc_df[xc_df['species_normalized'].isin(species_to_keep)].copy()
    
    print(f"   Total recordings: {len(xc_df)}")
    print(f"   Species count: {xc_df['species_normalized'].nunique()}")
    
    # Check existing cache
    print(f"\n💾 Cache directory: {cache_dir}")
    if cache_dir.exists():
        cached_files = list(cache_dir.glob('**/*.npy'))
        print(f"   Existing cached files: {len(cached_files)}")
    else:
        print(f"   Creating cache directory...")
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for multiprocessing
    print(f"\n⚙️  Configuration:")
    print(f"   Workers: {args.workers}")
    print(f"   Force regenerate: {args.force_regenerate}")
    print(f"   Target duration: {args.target_duration}s")
    print(f"   Sample rate: {args.sr} Hz")
    print(f"   Mel bins: {args.n_mels}")
    
    extract_args = [
        (idx, row, cache_dir, audio_dir, args.force_regenerate)
        for idx, row in xc_df.iterrows()
    ]
    
    # Extract spectrograms with multiprocessing
    print(f"\n🚀 Extracting spectrograms...")
    
    extract_fn = partial(
        extract_and_validate,
        target_duration=args.target_duration,
        sr=args.sr,
        n_mels=args.n_mels,
    )
    
    results = []
    with mp.Pool(args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(extract_fn, extract_args),
            total=len(extract_args),
            desc="Processing audio",
            unit="files",
        ):
            results.append(result)
    
    # Summarize results
    print(f"\n📊 Results Summary:")
    status_counts = {}
    for result in results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in sorted(status_counts.items()):
        pct = 100 * count / len(results)
        print(f"   {status:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Report errors
    errors = [r for r in results if r['status'] in ('error', 'invalid', 'missing_audio')]
    if errors:
        print(f"\n⚠️  Errors ({len(errors)} total):")
        for err in errors[:10]:  # Show first 10
            print(f"   {err['xc_id']:15s} ({err['species']:30s}): {err.get('error', 'missing audio')}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
    
    # Save results summary
    summary = {
        'total_recordings': len(xc_df),
        'status_counts': status_counts,
        'config': {
            'target_duration': args.target_duration,
            'sr': args.sr,
            'n_mels': args.n_mels,
            'workers': args.workers,
            'force_regenerate': args.force_regenerate,
        },
        'errors': errors,
    }
    
    summary_path = cache_dir / 'cache_generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_path}")
    
    # Final status
    success_rate = 100 * status_counts.get('success', 0) / len(results)
    cached_rate = 100 * status_counts.get('cached', 0) / len(results)
    total_ok = success_rate + cached_rate
    
    print(f"\n{'='*80}")
    print(f"✅ Cache generation complete!")
    print(f"   Success rate: {total_ok:.1f}% ({status_counts.get('success', 0) + status_counts.get('cached', 0)}/{len(results)})")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
