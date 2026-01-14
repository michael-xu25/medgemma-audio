#!/usr/bin/env python
"""
Train MAE (Masked Autoencoder) for audio encoder pretraining.

Usage:
    python train_mae.py
    python train_mae.py --config config/mae_config.yaml
    python train_mae.py --batch-size 32 --num-epochs 50
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run
from src.training.mae_pretrain import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE Pretraining")
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data-path", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints/mae", help="Output directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--run-name", type=str, help="Run name for logging")
    
    # Model args
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    
    # Training args
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Misc
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--save-every", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
