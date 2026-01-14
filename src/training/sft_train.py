"""
Supervised Fine-Tuning (SFT) for MedGemma Audio using Unsloth.

Based on: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb
"""

import os
import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

from src.models.medgemma_audio import MedGemmaAudio, MedGemmaAudioConfig
from src.models.audio_encoder import AudioEncoder
from src.data.dataset import AudioCapsDataset, AudioCapsCollator
from src.utils.logging import TrainingLogger


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    
    # Model
    model_name: str = "google/medgemma-4b-it"
    audio_encoder_path: Optional[str] = None
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 512
    
    # Data
    data_path: str = "data/processed"
    
    # Output
    output_dir: str = "checkpoints/sft"
    
    # Misc
    use_4bit: bool = True
    seed: int = 42


def format_prompt(caption: str, include_response: bool = True) -> str:
    """
    Format prompt for training/inference.
    
    Args:
        caption: The caption text
        include_response: Whether to include the response (for training)
    
    Returns:
        Formatted prompt string
    """
    # Using Gemma chat format
    prompt = "<start_of_turn>user\n[AUDIO] Describe this audio in detail.<end_of_turn>\n<start_of_turn>model\n"
    
    if include_response:
        prompt += f"{caption}<end_of_turn>"
    
    return prompt


class AudioSFTDataset(torch.utils.data.Dataset):
    """Dataset for SFT training with audio and captions."""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        tokenizer=None,
        max_length: int = 512,
        target_length: int = 10,
    ):
        self.audiocaps = AudioCapsDataset(
            data_path=data_path,
            split=split,
            target_length=target_length,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.audiocaps)
    
    def __getitem__(self, idx):
        sample = self.audiocaps[idx]
        
        # Format prompt
        text = format_prompt(sample['caption'], include_response=True)
        
        return {
            'audio_features': sample['audio_features'],
            'text': text,
            'caption': sample['caption'],
            'video_id': sample['video_id'],
        }


def create_sft_trainer(
    model: MedGemmaAudio,
    tokenizer,
    train_dataset,
    val_dataset,
    config: SFTConfig,
):
    """
    Create SFT trainer using TRL's SFTTrainer.
    
    This follows the Unsloth pattern for efficient training.
    """
    from trl import SFTTrainer, SFTConfig as TRLSFTConfig
    from transformers import TrainingArguments
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit" if config.use_4bit else "adamw_torch",
        seed=config.seed,
        report_to="wandb",
    )
    
    # Data collator that handles audio
    def collate_fn(batch):
        audio_features = torch.stack([item['audio_features'] for item in batch])
        texts = [item['text'] for item in batch]
        
        # Tokenize texts
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
        
        return {
            'audio_features': audio_features,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone(),
        }
    
    # Create trainer
    trainer = SFTTrainer(
        model=model.llm,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )
    
    return trainer


def load_model_with_unsloth(config: SFTConfig, logger):
    """
    Load MedGemma with Unsloth for efficient training.
    
    Following the Gemma3 notebook pattern.
    """
    from unsloth import FastLanguageModel
    
    logger.info(f"Loading {config.model_name} with Unsloth...")
    
    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=config.use_4bit,
        token=os.environ.get("HF_TOKEN"),
    )
    
    logger.info("Applying LoRA...")
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    
    return model, tokenizer


def train_with_custom_loop(
    model: MedGemmaAudio,
    train_dataset,
    val_dataset,
    tokenizer,
    config: SFTConfig,
    logger: TrainingLogger,
):
    """
    Custom training loop for MedGemma Audio.
    
    This is needed because we need to handle audio embeddings specially.
    """
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dataloaders
    def collate_fn(batch):
        audio_features = torch.stack([item['audio_features'] for item in batch])
        texts = [item['text'] for item in batch]
        captions = [item['caption'] for item in batch]
        
        # Tokenize texts
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
        
        # Create labels (shift for causal LM)
        labels = tokenized['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            'audio_features': audio_features,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
            'captions': captions,
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Optimizer - only optimize audio encoder, projector, and LoRA params
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config.num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config.num_epochs}")
        logger.info(f"{'='*60}")
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            # Move to device
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                audio_features=audio_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += outputs.loss.item()
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log
                if global_step % 10 == 0:
                    logger.log_metrics({
                        'train/loss': outputs.loss.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                    }, step=global_step)
            
            pbar.set_postfix({'loss': f"{outputs.loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / num_batches
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                audio_features = batch['audio_features'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    audio_features=audio_features,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                val_loss += outputs.loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        logger.log_metrics({
            'epoch': epoch,
            'train/epoch_loss': avg_train_loss,
            'val/loss': avg_val_loss,
        }, step=global_step)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(str(output_dir / "best_model"))
            logger.info(f"Saved best model with val_loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        model.save_pretrained(str(output_dir / f"checkpoint_epoch_{epoch}"))
    
    # Save final model
    model.save_pretrained(str(output_dir / "final_model"))
    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")


def main(args):
    """Main SFT training function."""
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = SFTConfig(**config_dict)
    else:
        config = SFTConfig()
    
    # Override with command line args
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Setup logging
    logger = TrainingLogger(
        log_dir="logs",
        project="medgemma-audio-sft",
        run_name=args.run_name,
        config=vars(config),
        use_wandb=args.use_wandb,
    )
    
    logger.info("Starting SFT training...")
    logger.info(f"Config: {vars(config)}")
    
    # Load LLM with Unsloth
    llm, tokenizer = load_model_with_unsloth(config, logger)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create MedGemma Audio model
    logger.info("Creating MedGemma Audio model...")
    
    model_config = MedGemmaAudioConfig(
        llm_model_name=config.model_name,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    
    model = MedGemmaAudio(model_config)
    model.set_llm(llm, tokenizer)
    
    # Load pretrained audio encoder if provided
    if config.audio_encoder_path:
        logger.info(f"Loading audio encoder from {config.audio_encoder_path}")
        state_dict = torch.load(config.audio_encoder_path, map_location="cpu")
        model.audio_encoder.load_state_dict(state_dict)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = AudioSFTDataset(
        data_path=config.data_path,
        split="train",
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )
    
    val_dataset = AudioSFTDataset(
        data_path=config.data_path,
        split="val",
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Train
    train_with_custom_loop(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        config=config,
        logger=logger,
    )
    
    logger.finish()


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
