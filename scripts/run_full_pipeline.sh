#!/bin/bash
# Run full MedGemma Audio training pipeline
# ==========================================
#
# This script runs all stages of the training pipeline:
# 1. Download data (AudioSet + AudioCaps)
# 2. MAE pretraining for audio encoder
# 3. SFT (Supervised Fine-Tuning)
# 4. GRPO (Reinforcement Learning)

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add project root to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "MedGemma Audio - Full Training Pipeline"
echo "=============================================="
echo ""

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set."
    echo "Please create a .env file with your HuggingFace token:"
    echo "  cp env.template .env"
    echo "  # Edit .env and add your token"
    exit 1
fi

# Configuration
SKIP_DOWNLOAD=false
SKIP_MAE=false
SKIP_SFT=false
SKIP_GRPO=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-mae)
            SKIP_MAE=true
            shift
            ;;
        --skip-sft)
            SKIP_SFT=true
            shift
            ;;
        --skip-grpo)
            SKIP_GRPO=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-download  Skip data download step"
            echo "  --skip-mae       Skip MAE pretraining step"
            echo "  --skip-sft       Skip SFT training step"
            echo "  --skip-grpo      Skip GRPO training step"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: Download data
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "=============================================="
    echo "Step 1/4: Downloading data..."
    echo "=============================================="
    ./scripts/download_data.sh
else
    echo "Skipping data download..."
fi

# Step 2: MAE pretraining
if [ "$SKIP_MAE" = false ]; then
    echo ""
    echo "=============================================="
    echo "Step 2/4: MAE pretraining..."
    echo "=============================================="
    ./scripts/run_mae.sh
else
    echo "Skipping MAE pretraining..."
fi

# Step 3: SFT training
if [ "$SKIP_SFT" = false ]; then
    echo ""
    echo "=============================================="
    echo "Step 3/4: SFT training..."
    echo "=============================================="
    ./scripts/run_sft.sh
else
    echo "Skipping SFT training..."
fi

# Step 4: GRPO training
if [ "$SKIP_GRPO" = false ]; then
    echo ""
    echo "=============================================="
    echo "Step 4/4: GRPO training..."
    echo "=============================================="
    ./scripts/run_grpo.sh
else
    echo "Skipping GRPO training..."
fi

echo ""
echo "=============================================="
echo "Full pipeline complete!"
echo "=============================================="
echo ""
echo "Checkpoints:"
echo "  - MAE encoder: checkpoints/mae/audio_encoder.pt"
echo "  - SFT model: checkpoints/sft/best_model/"
echo "  - GRPO model: checkpoints/grpo/best_model/"
