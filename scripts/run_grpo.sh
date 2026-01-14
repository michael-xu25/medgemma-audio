#!/bin/bash
# Run GRPO (Group Relative Policy Optimization) training
# ======================================================

set -e

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set. Please set it in .env file or environment."
    exit 1
fi

# Configuration
CONFIG="config/grpo_config.yaml"
DATA_PATH="data/processed"
OUTPUT_DIR="checkpoints/grpo"
MODEL_PATH="checkpoints/sft/best_model"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --reward-metric)
            REWARD_METRIC="$2"
            shift 2
            ;;
        --no-wandb)
            NO_WANDB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "MedGemma Audio - GRPO Training"
echo "=============================================="
echo ""
echo "Config: $CONFIG"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "SFT Model: $MODEL_PATH"
echo ""

# Check if SFT model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: SFT model not found at $MODEL_PATH"
    echo "Please run SFT training first: ./scripts/run_sft.sh"
    exit 1
fi

# Build command
CMD="python -m src.training.grpo_train"
CMD="$CMD --config $CONFIG"
CMD="$CMD --data-path $DATA_PATH"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --model-path $MODEL_PATH"

if [ -n "$NUM_STEPS" ]; then
    CMD="$CMD --num-steps $NUM_STEPS"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

if [ -n "$LEARNING_RATE" ]; then
    CMD="$CMD --learning-rate $LEARNING_RATE"
fi

if [ -n "$REWARD_METRIC" ]; then
    CMD="$CMD --reward-metric $REWARD_METRIC"
fi

if [ "$NO_WANDB" = true ]; then
    CMD="$CMD --no-wandb"
fi

# Run training
echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "=============================================="
echo "GRPO training complete!"
echo "=============================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
