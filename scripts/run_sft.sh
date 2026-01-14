#!/bin/bash
# Run SFT (Supervised Fine-Tuning) for MedGemma Audio
# ===================================================

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add project root to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

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
CONFIG="config/sft_config.yaml"
DATA_PATH="data/processed"
OUTPUT_DIR="checkpoints/sft"
AUDIO_ENCODER="checkpoints/mae/audio_encoder.pt"

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
        --audio-encoder)
            AUDIO_ENCODER="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
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
echo "MedGemma Audio - SFT Training"
echo "=============================================="
echo ""
echo "Config: $CONFIG"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Audio Encoder: $AUDIO_ENCODER"
echo ""

# Check if audio encoder exists
if [ ! -f "$AUDIO_ENCODER" ]; then
    echo "Warning: Audio encoder not found at $AUDIO_ENCODER"
    echo "Training will use randomly initialized audio encoder."
    echo "Consider running MAE pretraining first: ./scripts/run_mae.sh"
    echo ""
fi

# Build command
CMD="python -m src.training.sft_train"
CMD="$CMD --config $CONFIG"
CMD="$CMD --data-path $DATA_PATH"
CMD="$CMD --output-dir $OUTPUT_DIR"

if [ -f "$AUDIO_ENCODER" ]; then
    CMD="$CMD --audio-encoder-path $AUDIO_ENCODER"
fi

if [ -n "$MODEL_NAME" ]; then
    CMD="$CMD --model-name $MODEL_NAME"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

if [ -n "$NUM_EPOCHS" ]; then
    CMD="$CMD --num-epochs $NUM_EPOCHS"
fi

if [ -n "$LEARNING_RATE" ]; then
    CMD="$CMD --learning-rate $LEARNING_RATE"
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
echo "SFT training complete!"
echo "=============================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
