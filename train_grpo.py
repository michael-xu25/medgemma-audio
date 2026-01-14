#!/usr/bin/env python
"""
Train GRPO (Group Relative Policy Optimization) for MedGemma Audio.

Usage:
    python train_grpo.py
    python train_grpo.py --config config/grpo_config.yaml
    python train_grpo.py --num-steps 500
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run
from src.training.grpo_train import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training for MedGemma Audio")
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model-path", type=str, help="Path to SFT checkpoint")
    parser.add_argument("--model-name", type=str, help="Base model name")
    parser.add_argument("--data-path", type=str, help="Path to processed data")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--run-name", type=str, help="Run name for logging")
    
    # GRPO args
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--reward-metric", type=str, choices=['cider', 'bleu', 'rouge'])
    parser.add_argument("--kl-coef", type=float)
    parser.add_argument("--clip-range", type=float)
    
    # Training args
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    
    # Misc
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
