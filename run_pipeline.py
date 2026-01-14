#!/usr/bin/env python
"""
Run the full MedGemma Audio training pipeline.

Usage:
    python run_pipeline.py                    # Run everything
    python run_pipeline.py --skip-download    # Skip data download
    python run_pipeline.py --skip-mae         # Skip MAE pretraining
    python run_pipeline.py --only-download    # Only download data
"""

import os
import sys
import argparse
import subprocess

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def run_step(name: str, script: str, args: list = None):
    """Run a pipeline step."""
    print("\n" + "=" * 60)
    print(f"STEP: {name}")
    print("=" * 60 + "\n")
    
    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode != 0:
        print(f"\nError: {name} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {name} complete!")


def main():
    parser = argparse.ArgumentParser(description="Run full MedGemma Audio pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip-mae", action="store_true", help="Skip MAE pretraining")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT training")
    parser.add_argument("--skip-grpo", action="store_true", help="Skip GRPO training")
    parser.add_argument("--only-download", action="store_true", help="Only run download step")
    parser.add_argument("--only-preprocess", action="store_true", help="Only run preprocess step")
    parser.add_argument("--only-mae", action="store_true", help="Only run MAE step")
    parser.add_argument("--only-sft", action="store_true", help="Only run SFT step")
    parser.add_argument("--only-grpo", action="store_true", help="Only run GRPO step")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Check for HuggingFace token
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.environ.get("HF_TOKEN") and not args.skip_sft and not args.skip_grpo:
        if not args.only_download and not args.only_preprocess and not args.only_mae:
            print("Warning: HF_TOKEN not set. SFT and GRPO training will fail.")
            print("Set it in .env file: HF_TOKEN=your_token_here")
    
    print("\n" + "=" * 60)
    print("MedGemma Audio - Full Training Pipeline")
    print("=" * 60)
    
    extra_args = ["--no-wandb"] if args.no_wandb else []
    
    # Determine which steps to run
    if args.only_download:
        run_step("1/5: Download Data", "download_data.py")
        return
    
    if args.only_preprocess:
        run_step("2/5: Preprocess Data", "preprocess_data.py")
        return
    
    if args.only_mae:
        run_step("3/5: MAE Pretraining", "train_mae.py", extra_args)
        return
    
    if args.only_sft:
        run_step("4/5: SFT Training", "train_sft.py", extra_args)
        return
    
    if args.only_grpo:
        run_step("5/5: GRPO Training", "train_grpo.py", extra_args)
        return
    
    # Run full pipeline
    if not args.skip_download:
        run_step("1/5: Download Data", "download_data.py")
    else:
        print("\nSkipping download...")
    
    if not args.skip_preprocess:
        run_step("2/5: Preprocess Data", "preprocess_data.py")
    else:
        print("\nSkipping preprocessing...")
    
    if not args.skip_mae:
        run_step("3/5: MAE Pretraining", "train_mae.py", extra_args)
    else:
        print("\nSkipping MAE pretraining...")
    
    if not args.skip_sft:
        run_step("4/5: SFT Training", "train_sft.py", extra_args)
    else:
        print("\nSkipping SFT training...")
    
    if not args.skip_grpo:
        run_step("5/5: GRPO Training", "train_grpo.py", extra_args)
    else:
        print("\nSkipping GRPO training...")
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print("\nCheckpoints saved to:")
    print("  - MAE encoder:  checkpoints/mae/audio_encoder.pt")
    print("  - SFT model:    checkpoints/sft/best_model/")
    print("  - GRPO model:   checkpoints/grpo/best_model/")


if __name__ == "__main__":
    main()
