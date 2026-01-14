# MedGemma Audio

Fine-tune Google's MedGemma for audio understanding using Masked Autoencoders (MAE) and the AudioCaps dataset.

## Overview

This pipeline enables training MedGemma 4B for audio captioning by:

1. **MAE Pretraining**: Train an audio encoder on AudioSet using masked autoencoders
2. **SFT (Supervised Fine-Tuning)**: Fine-tune MedGemma with LoRA on audio-caption pairs
3. **GRPO (Group Relative Policy Optimization)**: Improve caption quality with reinforcement learning

## Architecture

```
Audio Features (128-dim @ 1Hz) → Audio Encoder (Transformer) → Projector (MLP) → MedGemma 4B → Caption
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp env.template .env
# Edit .env and add your HuggingFace token
```

### 2. Download Data

```bash
./scripts/download_data.sh
```

This downloads:
- AudioSet pre-extracted VGGish features (~2.4GB)
- AudioCaps annotations (~50k human-written captions)

### 3. Run Training

**Full Pipeline:**
```bash
./scripts/run_full_pipeline.sh
```

**Or run individual stages:**

```bash
# Stage 1: MAE Pretraining
./scripts/run_mae.sh

# Stage 2: SFT Training
./scripts/run_sft.sh

# Stage 3: GRPO Training
./scripts/run_grpo.sh
```

## Project Structure

```
medgemma-audio/
├── config/
│   ├── mae_config.yaml      # MAE pretraining config
│   ├── sft_config.yaml      # SFT config
│   └── grpo_config.yaml     # GRPO config
├── src/
│   ├── data/
│   │   ├── download.py      # Data download utilities
│   │   ├── dataset.py       # PyTorch datasets
│   │   └── preprocessing.py # Data preprocessing
│   ├── models/
│   │   ├── audio_encoder.py # Audio encoder with MAE
│   │   ├── projector.py     # Audio-to-LLM projector
│   │   └── medgemma_audio.py # Combined model
│   ├── training/
│   │   ├── mae_pretrain.py  # MAE training script
│   │   ├── sft_train.py     # SFT training script
│   │   └── grpo_train.py    # GRPO training script
│   └── utils/
│       ├── logging.py       # Training logging
│       └── metrics.py       # Evaluation metrics
├── scripts/
│   ├── download_data.sh     # Data download
│   ├── run_mae.sh           # MAE training
│   ├── run_sft.sh           # SFT training
│   ├── run_grpo.sh          # GRPO training
│   └── run_full_pipeline.sh # Full pipeline
├── .gitignore
├── env.template
├── requirements.txt
└── README.md
```

## Configuration

### MAE Pretraining

Edit `config/mae_config.yaml`:

```yaml
mask_ratio: 0.75      # Ratio of patches to mask
encoder_dim: 512      # Encoder hidden dimension
encoder_layers: 6     # Number of encoder layers
batch_size: 64
num_epochs: 100
learning_rate: 1.0e-4
```

### SFT Training

Edit `config/sft_config.yaml`:

```yaml
model_name: "google/medgemma-4b-it"
lora_r: 16
lora_alpha: 16
batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 2.0e-4
```

### GRPO Training

Edit `config/grpo_config.yaml`:

```yaml
num_generations: 4    # Responses per sample
reward_metric: "cider"  # cider, bleu, or rouge
kl_coef: 0.1
num_steps: 500
learning_rate: 5.0e-6
```

## Data

The pipeline uses:

- **AudioSet**: 2M+ audio clips with 527 sound classes
  - Uses pre-extracted 128-dim VGGish features at 1Hz
  - 10-second clips = 10 timesteps × 128 features

- **AudioCaps**: ~50k human-written captions for AudioSet clips
  - Split: 85% train / 5% val / 10% test

## Hardware Requirements

- **MAE Pretraining**: ~8GB GPU memory
- **SFT Training**: ~16GB GPU memory (with 4-bit quantization)
- **GRPO Training**: ~24GB GPU memory (generates multiple responses)

For lower memory, adjust batch sizes and gradient accumulation.

## Experiment Tracking

Training logs to Weights & Biases by default. Disable with `--no-wandb`:

```bash
./scripts/run_sft.sh --no-wandb
```

Or set in `.env`:
```
WANDB_MODE=offline
```

## Model Checkpoints

After training, find checkpoints at:

- `checkpoints/mae/audio_encoder.pt` - Pretrained audio encoder
- `checkpoints/sft/best_model/` - Best SFT model
- `checkpoints/grpo/best_model/` - Best GRPO model

## Usage Example

```python
from src.models.medgemma_audio import MedGemmaAudio, MedGemmaAudioConfig

# Load trained model
model = MedGemmaAudio.from_pretrained("checkpoints/grpo/best_model")

# Generate caption
audio_features = ...  # Shape: (1, 10, 128)
captions = model.generate(audio_features, prompt="Describe this audio:")
print(captions[0])
```

## License

This project uses:
- AudioSet: Creative Commons Attribution 4.0 International (CC BY 4.0)
- AudioCaps: MIT License
- MedGemma: Google's model license

## References

- [AudioSet](https://research.google.com/audioset/)
- [AudioCaps](https://github.com/cdjkim/audiocaps)
- [MedGemma](https://huggingface.co/google/medgemma-4b-it)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Masked Autoencoders](https://arxiv.org/abs/2111.06377)
