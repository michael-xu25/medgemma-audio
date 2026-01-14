#!/usr/bin/env python
"""
Preprocess AudioSet and AudioCaps data into train/val/test splits.

Usage:
    python preprocess_data.py
    python preprocess_data.py --train-ratio 0.85 --val-ratio 0.05 --test-ratio 0.10
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pathlib import Path
from src.data.preprocessing import (
    load_audiocaps_annotations,
    load_audioset_features,
    create_data_splits,
    save_processed_data,
)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess AudioSet and AudioCaps data")
    parser.add_argument(
        "--audioset-dir",
        type=str,
        default="data/audioset",
        help="Directory containing AudioSet features"
    )
    parser.add_argument(
        "--audiocaps-dir",
        type=str,
        default="data/audiocaps",
        help="Directory containing AudioCaps annotations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Test set ratio"
    )
    
    args = parser.parse_args()
    
    # Load AudioCaps annotations
    print("Loading AudioCaps annotations...")
    audiocaps = load_audiocaps_annotations(args.audiocaps_dir)
    
    # Get video IDs from AudioCaps
    video_ids = list(audiocaps.keys())
    
    # Load AudioSet features (only for videos in AudioCaps)
    print("\nLoading AudioSet features...")
    features_dir = Path(args.audioset_dir) / "features"
    audioset = load_audioset_features(str(features_dir), video_ids=video_ids)
    
    # Create splits
    print("\nCreating data splits...")
    train, val, test = create_data_splits(
        audiocaps, audioset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Save processed data
    print("\nSaving processed data...")
    save_processed_data(train, val, test, args.output_dir)
    
    print("\nPreprocessing complete!")
