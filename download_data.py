#!/usr/bin/env python
"""
Download AudioSet features and AudioCaps annotations.

Usage:
    python download_data.py
    python download_data.py --skip-features
    python download_data.py --region eu
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data.download import download_all_data, download_audioset_metadata, download_audiocaps

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AudioSet and AudioCaps data")
    parser.add_argument(
        "--audioset-dir",
        type=str,
        default="data/audioset",
        help="Directory for AudioSet features"
    )
    parser.add_argument(
        "--audiocaps-dir",
        type=str,
        default="data/audiocaps",
        help="Directory for AudioCaps annotations"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us",
        choices=["us", "eu", "asia"],
        help="Google Cloud Storage region for AudioSet download"
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip downloading AudioSet features (only download metadata and AudioCaps)"
    )
    
    args = parser.parse_args()
    
    if args.skip_features:
        download_audioset_metadata(args.audioset_dir)
        download_audiocaps(args.audiocaps_dir)
    else:
        download_all_data(args.audioset_dir, args.audiocaps_dir, args.region)
    
    print("\nData download complete!")
