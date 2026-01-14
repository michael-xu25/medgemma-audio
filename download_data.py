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
import argparse
import requests
import tarfile
import csv
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_audioset_features(data_dir: str = "data/audioset", region: str = "us") -> Path:
    """Download AudioSet pre-extracted 128-dim VGGish features."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    features_dir = data_path / "features"
    tar_path = data_path / "features.tar.gz"
    
    if features_dir.exists() and any(features_dir.iterdir()):
        print(f"AudioSet features already exist at {features_dir}")
        return features_dir
    
    url = f"https://storage.googleapis.com/{region}_audioset/youtube_corpus/v1/features/features.tar.gz"
    print(f"Downloading AudioSet features from {url}...")
    print("This is approximately 2.4GB and may take a while...")
    
    download_file(url, tar_path)
    
    print("Extracting features...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=data_path)
    
    tar_path.unlink()
    print(f"AudioSet features extracted to {features_dir}")
    
    return features_dir


def download_audioset_metadata(data_dir: str = "data/audioset") -> dict:
    """Download AudioSet metadata CSV files."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    base_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv"
    
    files = {
        "eval": "eval_segments.csv",
        "balanced_train": "balanced_train_segments.csv",
        "unbalanced_train": "unbalanced_train_segments.csv",
        "class_labels": "class_labels_indices.csv",
    }
    
    paths = {}
    for key, filename in files.items():
        filepath = data_path / filename
        if not filepath.exists():
            url = f"{base_url}/{filename}"
            print(f"Downloading {filename}...")
            download_file(url, filepath)
        paths[key] = filepath
    
    return paths


def download_audiocaps(data_dir: str = "data/audiocaps") -> dict:
    """Download AudioCaps dataset annotations."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset"
    
    files = {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",
    }
    
    paths = {}
    for split, filename in files.items():
        filepath = data_path / filename
        if not filepath.exists():
            url = f"{base_url}/{filename}"
            print(f"Downloading AudioCaps {split} split...")
            download_file(url, filepath)
        paths[split] = filepath
    
    print("\nAudioCaps dataset statistics:")
    for split, filepath in paths.items():
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
        print(f"  {split}: {count} samples")
    
    return paths


def main():
    parser = argparse.ArgumentParser(description="Download AudioSet and AudioCaps data")
    parser.add_argument("--audioset-dir", type=str, default="data/audioset")
    parser.add_argument("--audiocaps-dir", type=str, default="data/audiocaps")
    parser.add_argument("--region", type=str, default="us", choices=["us", "eu", "asia"])
    parser.add_argument("--skip-features", action="store_true", 
                        help="Skip downloading AudioSet features (only download metadata and AudioCaps)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Downloading AudioSet metadata...")
    print("=" * 60)
    download_audioset_metadata(args.audioset_dir)
    
    if not args.skip_features:
        print("\n" + "=" * 60)
        print("Downloading AudioSet features...")
        print("=" * 60)
        download_audioset_features(args.audioset_dir, args.region)
    
    print("\n" + "=" * 60)
    print("Downloading AudioCaps annotations...")
    print("=" * 60)
    download_audiocaps(args.audiocaps_dir)
    
    print("\n" + "=" * 60)
    print("Data download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
