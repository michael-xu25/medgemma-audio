#!/bin/bash
# Run MAE (Masked Autoencoder) pretraining for audio encoder
# ==========================================================

set -e

# Configuration
CONFIG="config/mae_config.yaml"
DATA_PATH="data/processed"
OUTPUT_DIR="checkpoints/mae"
LOG_DIR="logs"

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
echo "MedGemma Audio - MAE Pretraining"
echo "=============================================="
echo ""
echo "Config: $CONFIG"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Build command
CMD="python -m src.training.mae_pretrain"
CMD="$CMD --config $CONFIG"
CMD="$CMD --data-path $DATA_PATH"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --log-dir $LOG_DIR"

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
echo "MAE pretraining complete!"
echo "=============================================="
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Audio encoder: $OUTPUT_DIR/audio_encoder.pt"
