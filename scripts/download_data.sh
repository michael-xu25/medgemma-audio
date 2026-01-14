#!/bin/bash
# Download AudioSet features and AudioCaps annotations
# =====================================================

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add project root to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# Configuration
AUDIOSET_DIR="data/audioset"
AUDIOCAPS_DIR="data/audiocaps"
PROCESSED_DIR="data/processed"
REGION="us"  # Options: us, eu, asia

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --audioset-dir)
            AUDIOSET_DIR="$2"
            shift 2
            ;;
        --audiocaps-dir)
            AUDIOCAPS_DIR="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --skip-features)
            SKIP_FEATURES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "MedGemma Audio - Data Download"
echo "=============================================="
echo ""
echo "AudioSet directory: $AUDIOSET_DIR"
echo "AudioCaps directory: $AUDIOCAPS_DIR"
echo "Region: $REGION"
echo ""

# Create directories
mkdir -p "$AUDIOSET_DIR"
mkdir -p "$AUDIOCAPS_DIR"
mkdir -p "$PROCESSED_DIR"

# Download data using Python script
if [ "$SKIP_FEATURES" = true ]; then
    echo "Skipping AudioSet features download..."
    python -m src.data.download \
        --audioset-dir "$AUDIOSET_DIR" \
        --audiocaps-dir "$AUDIOCAPS_DIR" \
        --skip-features
else
    python -m src.data.download \
        --audioset-dir "$AUDIOSET_DIR" \
        --audiocaps-dir "$AUDIOCAPS_DIR" \
        --region "$REGION"
fi

# Preprocess and create splits
echo ""
echo "=============================================="
echo "Preprocessing data and creating splits..."
echo "=============================================="

python -m src.data.preprocessing \
    --audioset-dir "$AUDIOSET_DIR" \
    --audiocaps-dir "$AUDIOCAPS_DIR" \
    --output-dir "$PROCESSED_DIR" \
    --train-ratio 0.85 \
    --val-ratio 0.05 \
    --test-ratio 0.10

echo ""
echo "=============================================="
echo "Data download and preprocessing complete!"
echo "=============================================="
echo ""
echo "Data saved to:"
echo "  - AudioSet features: $AUDIOSET_DIR"
echo "  - AudioCaps captions: $AUDIOCAPS_DIR"
echo "  - Processed splits: $PROCESSED_DIR"
