#!/usr/bin/env python
"""
Train SFT (Supervised Fine-Tuning) for MedGemma Audio.

Usage:
    python train_sft.py
    python train_sft.py --config config/sft_config.yaml
    python train_sft.py --batch-size 4 --num-epochs 3
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run
from src.training.sft_train import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training for MedGemma Audio")
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model-name", type=str, help="HuggingFace model name")
    parser.add_argument("--audio-encoder-path", type=str, help="Path to pretrained audio encoder")
    parser.add_argument("--data-path", type=str, help="Path to processed data")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--run-name", type=str, help="Run name for logging")
    
    # Training args
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    
    # LoRA args
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    
    # Misc
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
