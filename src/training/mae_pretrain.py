"""
Masked Autoencoder (MAE) pretraining for audio encoder.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.audio_encoder import AudioMAE
from src.data.dataset import MAEAudioDataset
from src.utils.logging import TrainingLogger, get_logger
from src.utils.metrics import compute_mae_metrics


def train_epoch(
    model: AudioMAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_masked_mse = 0.0
    total_cos_sim = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        features = batch['features'].to(device)
        
        # Forward pass
        reconstruction, target, mask = model(features)
        
        # Compute loss
        loss = model.compute_loss(reconstruction, target, mask)
        loss = loss / grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        with torch.no_grad():
            metrics = compute_mae_metrics(reconstruction, target, mask)
        
        total_loss += loss.item() * grad_accum_steps
        total_masked_mse += metrics['masked_mse']
        total_cos_sim += metrics['cosine_similarity']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item() * grad_accum_steps:.4f}",
            'mse': f"{metrics['masked_mse']:.4f}",
            'cos_sim': f"{metrics['cosine_similarity']:.4f}",
        })
        
        # Log periodically
        if step % 100 == 0:
            logger.log_metrics({
                'train/loss': loss.item() * grad_accum_steps,
                'train/masked_mse': metrics['masked_mse'],
                'train/cosine_similarity': metrics['cosine_similarity'],
            })
    
    return {
        'loss': total_loss / num_batches,
        'masked_mse': total_masked_mse / num_batches,
        'cosine_similarity': total_cos_sim / num_batches,
    }


@torch.no_grad()
def validate(
    model: AudioMAE,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_masked_mse = 0.0
    total_cos_sim = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        features = batch['features'].to(device)
        
        # Forward pass
        reconstruction, target, mask = model(features)
        
        # Compute loss
        loss = model.compute_loss(reconstruction, target, mask)
        
        # Metrics
        metrics = compute_mae_metrics(reconstruction, target, mask)
        
        total_loss += loss.item()
        total_masked_mse += metrics['masked_mse']
        total_cos_sim += metrics['cosine_similarity']
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'masked_mse': total_masked_mse / num_batches,
        'cosine_similarity': total_cos_sim / num_batches,
    }


def main(args):
    """Main training function."""
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with command line args
    config.update({k: v for k, v in vars(args).items() if v is not None})
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup logging
    logger = TrainingLogger(
        log_dir=config.get('log_dir', 'logs'),
        project="medgemma-audio-mae",
        run_name=config.get('run_name', None),
        config=config,
        use_wandb=config.get('use_wandb', True),
    )
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = MAEAudioDataset(
        data_path=config.get('data_path', 'data/processed'),
        split="train",
        target_length=config.get('target_length', 10),
        mask_ratio=config.get('mask_ratio', 0.75),
    )
    
    val_dataset = MAEAudioDataset(
        data_path=config.get('data_path', 'data/processed'),
        split="val",
        target_length=config.get('target_length', 10),
        mask_ratio=config.get('mask_ratio', 0.75),
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    # Create model
    logger.info("Creating model...")
    model = AudioMAE(
        input_dim=config.get('input_dim', 128),
        encoder_dim=config.get('encoder_dim', 512),
        encoder_layers=config.get('encoder_layers', 6),
        encoder_heads=config.get('encoder_heads', 8),
        decoder_dim=config.get('decoder_dim', 256),
        decoder_layers=config.get('decoder_layers', 2),
        decoder_heads=config.get('decoder_heads', 4),
        mask_ratio=config.get('mask_ratio', 0.75),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('target_length', 10) + 10,
    ).to(device)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.05),
        betas=(0.9, 0.95),
    )
    
    # Create scheduler
    num_epochs = config.get('num_epochs', 100)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=config.get('min_lr', 1e-6),
    )
    
    # Training loop
    best_val_loss = float('inf')
    output_dir = Path(config.get('output_dir', 'checkpoints/mae'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, logger,
            grad_accum_steps=config.get('grad_accum_steps', 1),
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch metrics
        logger.log_metrics({
            'epoch': epoch,
            'train/epoch_loss': train_metrics['loss'],
            'train/epoch_masked_mse': train_metrics['masked_mse'],
            'train/epoch_cosine_similarity': train_metrics['cosine_similarity'],
            'val/loss': val_metrics['loss'],
            'val/masked_mse': val_metrics['masked_mse'],
            'val/cosine_similarity': val_metrics['cosine_similarity'],
            'lr': scheduler.get_last_lr()[0],
        })
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
            }, output_dir / 'best_model.pt')
            logger.info(f"Saved best model with val_loss: {val_metrics['loss']:.4f}")
        
        # Save checkpoint periodically
        if epoch % config.get('save_every', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, output_dir / 'final_model.pt')
    
    # Extract and save encoder
    encoder = model.get_encoder()
    torch.save(encoder.state_dict(), output_dir / 'audio_encoder.pt')
    
    logger.info(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to {output_dir}")
    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE Pretraining")
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data-path", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints/mae", help="Output directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--run-name", type=str, help="Run name for logging")
    
    # Model args
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    
    # Training args
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Misc
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--save-every", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
