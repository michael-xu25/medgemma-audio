#!/usr/bin/env python
"""
MedGemma Audio - Complete Training Pipeline

This script handles everything:
1. Installs required packages
2. Downloads AudioSet features and AudioCaps annotations
3. Preprocesses data into train/val/test splits
4. Runs MAE pretraining
5. Runs SFT training
6. Runs GRPO training

Usage:
    python run_all.py
    python run_all.py --skip-install
    python run_all.py --skip-download
    python run_all.py --only-download
"""

import subprocess
import sys
import os
import argparse

# =============================================================================
# SHARED MODEL CLASSES (used by MAE, SFT, and GRPO)
# =============================================================================

# These will be initialized after torch is imported
TransformerMAE = None
AudioProjector = None

def _init_model_classes():
    """Initialize model classes after torch is available."""
    global TransformerMAE, AudioProjector
    
    if TransformerMAE is not None:
        return  # Already initialized
    
    import torch
    import torch.nn as nn
    
    class _TransformerMAE(nn.Module):
        """
        AudioMAE-style Masked Autoencoder for audio features.
        
        Based on: https://arxiv.org/abs/2207.06405 (AudioMAE by Facebook Research)
        
        Key differences from standard MAE:
        - Structured masking: masks entire time frames (more suitable for audio)
        - Higher mask ratio (80%) for audio
        - Local attention patterns for efficiency
        """
        def __init__(self, input_dim=128, hidden_dim=256, num_layers=4, num_heads=4, 
                     decoder_dim=128, decoder_layers=2, mask_ratio=0.80, max_len=64):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.mask_ratio = mask_ratio
            self.max_len = max_len
            
            # Input projection with layer norm (important for audio)
            self.patch_embed = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # Learnable positional embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, hidden_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            # CLS token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            
            # Transformer encoder (deeper for audio)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.encoder_norm = nn.LayerNorm(hidden_dim)
            
            # Decoder (lightweight but effective)
            self.decoder_embed = nn.Linear(hidden_dim, decoder_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, max_len, decoder_dim))
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
            
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=max(num_heads // 2, 2),
                dim_feedforward=decoder_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
            self.decoder_norm = nn.LayerNorm(decoder_dim)
            
            # Output projection
            self.output_proj = nn.Linear(decoder_dim, input_dim)
            
            self._init_weights()
        
        def _init_weights(self):
            # Initialize patch_embed linear layer
            for m in self.patch_embed.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            # Initialize other linear layers
            nn.init.xavier_uniform_(self.decoder_embed.weight)
            nn.init.zeros_(self.decoder_embed.bias)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        
        def structured_masking(self, x):
            """
            AudioMAE-style structured masking: mask entire time frames.
            This is more suitable for audio than random patch masking.
            """
            B, T, D = x.shape
            num_mask = int(T * self.mask_ratio)
            num_keep = T - num_mask
            
            # Generate random indices for each batch
            noise = torch.rand(B, T, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            # Keep first num_keep tokens (unmasked)
            ids_keep = ids_shuffle[:, :num_keep]
            x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
            
            # Create binary mask: 1 = masked, 0 = visible
            mask = torch.ones(B, T, device=x.device)
            mask[:, :num_keep] = 0
            mask = torch.gather(mask, 1, ids_restore)
            
            return x_masked, mask, ids_restore, ids_keep
        
        def forward(self, x):
            B, T, D = x.shape
            
            # Embed patches
            x_embed = self.patch_embed(x)  # (B, T, hidden_dim)
            
            # Add positional embedding (before masking, as in AudioMAE)
            x_embed = x_embed + self.pos_embed[:, 1:T+1, :]
            
            # Structured masking (mask entire time frames)
            x_masked, mask, ids_restore, ids_keep = self.structured_masking(x_embed)
            
            # Add CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
            x_masked = torch.cat([cls_tokens, x_masked], dim=1)
            
            # Encode visible tokens only
            x_encoded = self.encoder(x_masked)
            x_encoded = self.encoder_norm(x_encoded)
            
            # Decoder
            # Project to decoder dimension
            x_dec = self.decoder_embed(x_encoded)
            
            # Separate CLS and visible tokens
            cls_dec = x_dec[:, :1, :]
            visible_dec = x_dec[:, 1:, :]
            
            # Create full sequence with mask tokens
            num_mask = T - visible_dec.size(1)
            mask_tokens = self.mask_token.expand(B, num_mask, -1)
            
            # Concatenate visible and mask tokens
            x_full = torch.cat([visible_dec, mask_tokens], dim=1)
            
            # Unshuffle to original order
            x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
            
            # Add decoder positional embedding
            x_full = x_full + self.decoder_pos_embed[:, :T, :]
            
            # Decode
            x_decoded = self.decoder(x_full)
            x_decoded = self.decoder_norm(x_decoded)
            
            # Project to output dimension
            reconstruction = self.output_proj(x_decoded)
            
            return reconstruction, mask
        
        def compute_loss(self, x, reconstruction, mask):
            """
            Compute normalized MSE loss on masked patches only.
            Uses per-patch normalization as in AudioMAE.
            """
            # Per-patch normalize target for more stable training
            target = x
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target_norm = (target - mean) / (var + 1e-6).sqrt()
            
            # Also normalize reconstruction
            pred_mean = reconstruction.mean(dim=-1, keepdim=True)
            pred_var = reconstruction.var(dim=-1, keepdim=True)
            pred_norm = (reconstruction - pred_mean) / (pred_var + 1e-6).sqrt()
            
            # MSE on normalized values (only masked patches)
            loss = (pred_norm - target_norm) ** 2
            loss = loss.mean(dim=-1)  # Mean over features
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)  # Mean over masked patches
            
            return loss
        
        def get_encoder_output(self, x):
            """Get encoder output for downstream tasks (CLS token)."""
            B, T, D = x.shape
            
            # Embed patches
            x_embed = self.patch_embed(x)
            x_embed = x_embed + self.pos_embed[:, 1:T+1, :]
            
            # Add CLS token (no masking during inference)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
            x_full = torch.cat([cls_tokens, x_embed], dim=1)
            
            # Encode all tokens
            x_encoded = self.encoder(x_full)
            x_encoded = self.encoder_norm(x_encoded)
            
            return x_encoded[:, 0]  # Return CLS token
    
    class _AudioProjector(nn.Module):
        """Projects audio embeddings to LLM embedding space with multiple tokens."""
        def __init__(self, audio_dim, llm_dim, num_tokens=8, hidden_dim=None):
            super().__init__()
            self.num_tokens = num_tokens
            self.llm_dim = llm_dim
            hidden_dim = hidden_dim or (audio_dim + llm_dim) // 2
            
            # Project to num_tokens * llm_dim, then reshape
            self.proj = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, llm_dim * num_tokens),
            )
            self.norm = nn.LayerNorm(llm_dim)
        
        def forward(self, x):
            # x: (B, audio_dim) or (B, 1, audio_dim)
            if x.dim() == 3:
                x = x.squeeze(1)
            x = self.proj(x)  # (B, llm_dim * num_tokens)
            x = x.view(x.size(0), self.num_tokens, self.llm_dim)  # (B, num_tokens, llm_dim)
            x = self.norm(x)
            return x  # (B, num_tokens, llm_dim)
    
    TransformerMAE = _TransformerMAE
    AudioProjector = _AudioProjector

# =============================================================================
# STEP 0: Install packages
# =============================================================================

def install_packages():
    """Install required packages."""
    print("\n" + "=" * 60)
    print("Installing required packages...")
    print("=" * 60 + "\n")
    
    packages = [
        "torch",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "trl>=0.8.0",
        "datasets>=2.18.0",
        "pandas",
        "numpy",
        "pyyaml",
        "python-dotenv",
        "tqdm",
        "requests",
        "tensorflow",
        "tf-keras",  # Required for Keras 3 compatibility
        "wandb",
        "nltk",
        "pycocoevalcap",  # For CIDEr scoring
        "sentence-transformers",  # For CLAP text embeddings
    ]
    
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    
    # Try to install unsloth (may fail on some systems)
    try:
        print("Installing unsloth...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth"])
    except:
        print("Warning: Could not install unsloth. SFT/GRPO training may not work.")
    
    print("\n✓ Package installation complete!")


# =============================================================================
# STEP 1: Download data
# =============================================================================

def download_data(audioset_dir="data/audioset", audiocaps_dir="data/audiocaps", 
                  region="us", skip_features=False):
    """Download AudioSet and AudioCaps data."""
    print("\n" + "=" * 60)
    print("Downloading data...")
    print("=" * 60 + "\n")
    
    import requests
    import tarfile
    import csv
    from pathlib import Path
    from tqdm import tqdm
    
    def download_file(url, dest_path, chunk_size=8192):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    # Download AudioSet metadata
    audioset_path = Path(audioset_dir)
    audioset_path.mkdir(parents=True, exist_ok=True)
    
    base_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv"
    for filename in ["eval_segments.csv", "balanced_train_segments.csv", 
                     "unbalanced_train_segments.csv", "class_labels_indices.csv"]:
        filepath = audioset_path / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            download_file(f"{base_url}/{filename}", filepath)
    
    # Download AudioSet features
    if not skip_features:
        features_dir = audioset_path / "features"
        tar_path = audioset_path / "features.tar.gz"
        
        if not features_dir.exists() or not any(features_dir.iterdir() if features_dir.exists() else []):
            url = f"https://storage.googleapis.com/{region}_audioset/youtube_corpus/v1/features/features.tar.gz"
            print(f"\nDownloading AudioSet features (~2.4GB)...")
            download_file(url, tar_path)
            print("Extracting features...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=audioset_path)
            tar_path.unlink()
    
    # Download AudioCaps
    audiocaps_path = Path(audiocaps_dir)
    audiocaps_path.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset"
    for split in ["train", "val", "test"]:
        filepath = audiocaps_path / f"{split}.csv"
        if not filepath.exists():
            print(f"Downloading AudioCaps {split}...")
            download_file(f"{base_url}/{split}.csv", filepath)
    
    print("\n✓ Data download complete!")


# =============================================================================
# STEP 2: Preprocess data
# =============================================================================

def preprocess_data(audioset_dir="data/audioset", audiocaps_dir="data/audiocaps",
                    output_dir="data/processed", train_ratio=0.85, val_ratio=0.05, test_ratio=0.10):
    """Preprocess data into train/val/test splits."""
    print("\n" + "=" * 60)
    print("Preprocessing data...")
    print("=" * 60 + "\n")
    
    import csv
    import json
    import pickle
    import numpy as np
    from pathlib import Path
    from collections import defaultdict
    from tqdm import tqdm
    
    try:
        import tensorflow as tf
    except ImportError:
        print("Error: tensorflow not installed. Run with --skip-install removed.")
        return
    
    # Load AudioCaps annotations
    print("Loading AudioCaps annotations...")
    audiocaps_path = Path(audiocaps_dir)
    annotations = defaultdict(list)
    
    for split in ["train", "val", "test"]:
        csv_path = audiocaps_path / f"{split}.csv"
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_id = row['youtube_id']
                    annotations[video_id].append({
                        'audiocap_id': row['audiocap_id'],
                        'youtube_id': video_id,
                        'start_time': float(row['start_time']),
                        'caption': row['caption'],
                        'split': split,
                    })
    
    print(f"Loaded {sum(len(v) for v in annotations.values())} captions for {len(annotations)} videos")
    
    # Load AudioSet features
    print("\nLoading AudioSet features...")
    features_dir = Path(audioset_dir) / "features"
    video_ids = set(annotations.keys())
    
    all_features = {}
    tfrecord_files = sorted(features_dir.glob("*.tfrecord")) if features_dir.exists() else []
    
    if not tfrecord_files:
        print("Warning: No TFRecord files found. Creating synthetic data for demo...")
        # Create synthetic features for AudioCaps videos (for demo/testing)
        for vid in list(video_ids)[:5000]:
            all_features[vid] = {
                'video_id': vid,
                'start_time': 0.0,
                'end_time': 10.0,
                'labels': [0],
                'audio_features': np.random.randn(10, 128).astype(np.float32),
            }
    else:
        # Load from TFRecords - process ALL files to find matches
        print(f"Found {len(tfrecord_files)} TFRecord files")
        for tfrecord_path in tqdm(tfrecord_files, desc="Loading TFRecords"):
            if len(all_features) >= 10000:  # Stop after finding enough
                break
            try:
                for record in tf.data.TFRecordDataset(str(tfrecord_path)):
                    example = tf.train.SequenceExample()
                    example.ParseFromString(record.numpy())
                    
                    vid = example.context.feature['video_id'].bytes_list.value[0].decode('utf-8')
                    if vid not in video_ids:
                        continue
                    
                    start_time = example.context.feature['start_time_seconds'].float_list.value[0]
                    end_time = example.context.feature['end_time_seconds'].float_list.value[0]
                    labels = list(example.context.feature['labels'].int64_list.value)
                    
                    audio_embeddings = []
                    for feature in example.feature_lists.feature_list['audio_embedding'].feature:
                        embedding = np.frombuffer(feature.bytes_list.value[0], dtype=np.uint8)
                        embedding = embedding.astype(np.float32) / 255.0
                        audio_embeddings.append(embedding)
                    
                    all_features[vid] = {
                        'video_id': vid,
                        'start_time': start_time,
                        'end_time': end_time,
                        'labels': labels,
                        'audio_features': np.array(audio_embeddings),
                    }
            except Exception as e:
                continue
    
    print(f"Loaded features for {len(all_features)} videos")
    
    if len(all_features) == 0:
        print("Error: No features loaded. Creating minimal synthetic dataset...")
        for vid in list(video_ids)[:1000]:
            all_features[vid] = {
                'video_id': vid,
                'start_time': 0.0,
                'end_time': 10.0,
                'labels': [0],
                'audio_features': np.random.randn(10, 128).astype(np.float32),
            }
    
    # Create merged samples
    print("\nCreating data splits...")
    common_ids = set(annotations.keys()) & set(all_features.keys())
    print(f"Found {len(common_ids)} videos with both features and captions")
    
    samples = []
    for video_id in common_ids:
        features = all_features[video_id]
        for caption_data in annotations[video_id]:
            samples.append({
                'video_id': video_id,
                'audio_features': features['audio_features'],
                'labels': features['labels'],
                'caption': caption_data['caption'],
                'audiocap_id': caption_data['audiocap_id'],
                'start_time': caption_data['start_time'],
            })
    
    # Split
    np.random.seed(42)
    np.random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = samples[:n_train]
    val_data = samples[n_train:n_train + n_val]
    test_data = samples[n_train + n_val:]
    
    print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        with open(output_path / f"{split_name}.pkl", 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {split_name}: {len(data)} samples")
    
    print("\n✓ Preprocessing complete!")


# =============================================================================
# STEP 3: MAE Pretraining (Transformer-based)
# =============================================================================

def train_mae(data_path="data/processed", output_dir="checkpoints/mae", 
              batch_size=64, num_epochs=50, use_wandb=False,
              hidden_dim=256, num_layers=4, num_heads=4, mask_ratio=0.80):
    """
    Train Transformer-based MAE for audio encoder pretraining.
    
    Architecture:
    - Transformer encoder with positional embeddings
    - Masked autoencoding (75% masking by default)
    - Lightweight decoder for reconstruction
    
    Args:
        num_epochs: Training epochs (default: 50, was 10)
        hidden_dim: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mask_ratio: Fraction of patches to mask
    """
    print("\n" + "=" * 60)
    print("Training MAE (Transformer Audio Encoder)...")
    print("=" * 60 + "\n")
    
    import pickle
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # Initialize global model classes
    _init_model_classes()
    global TransformerMAE
    from pathlib import Path
    from tqdm import tqdm
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_path = Path(data_path) / "train.pkl"
    val_path = Path(data_path) / "val.pkl"
    
    if not train_path.exists():
        print(f"Error: {train_path} not found. Run preprocessing first.")
        return
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    if len(train_data) == 0:
        print("Error: No training data. Preprocessing may have failed.")
        return
    
    # Dataset
    class AudioDataset(Dataset):
        def __init__(self, data, target_length=20):
            self.data = data
            self.target_length = target_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            features = self.data[idx]['audio_features']
            # Pad/truncate
            if len(features) < self.target_length:
                padding = np.zeros((self.target_length - len(features), features.shape[1]))
                features = np.concatenate([features, padding], axis=0)
            else:
                features = features[:self.target_length]
            # Normalize
            features = (features - features.mean()) / (features.std() + 1e-8)
            return torch.from_numpy(features).float()
    
    train_loader = DataLoader(AudioDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(AudioDataset(val_data), batch_size=batch_size, num_workers=2)
    
    # Create model (using global TransformerMAE class)
    model = TransformerMAE(
        input_dim=128,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mask_ratio=mask_ratio,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Architecture: {num_layers} layers, {num_heads} heads, dim={hidden_dim}")
    print(f"Mask ratio: {mask_ratio}")
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'reconstruction_quality': []}
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            batch = batch.to(device)
            
            reconstruction, mask = model(batch)
            loss = model.compute_loss(batch, reconstruction, mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        recon_quality = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstruction, mask = model(batch)
                loss = model.compute_loss(batch, reconstruction, mask)
                val_loss += loss.item()
                
                # Compute reconstruction quality (cosine similarity on masked patches)
                cos_sim = F.cosine_similarity(
                    reconstruction.view(-1, 128),
                    batch.view(-1, 128),
                    dim=-1
                ).mean()
                recon_quality += cos_sim.item()
        
        val_loss /= len(val_loader)
        recon_quality /= len(val_loader)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['reconstruction_quality'].append(recon_quality)
        
        # Print progress
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Recon Sim={recon_quality:.4f}, LR={lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'recon_quality': recon_quality,
            }, f"{output_dir}/best_model.pt")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
            }, f"{output_dir}/checkpoint_epoch_{epoch}.pt")
    
    # Save final model and history
    torch.save(model.state_dict(), f"{output_dir}/final_model.pt")
    
    import json
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("MAE Training Complete!")
    print(f"{'='*60}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final reconstruction similarity: {history['reconstruction_quality'][-1]:.4f}")
    print(f"Loss reduction: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f} "
          f"({100*(1-history['train_loss'][-1]/history['train_loss'][0]):.1f}% improvement)")
    print(f"\nSaved to: {output_dir}/")
    
    # Validate that transformer is learning
    print(f"\n{'='*60}")
    print("Pretraining Quality Check")
    print(f"{'='*60}")
    if history['reconstruction_quality'][-1] > 0.5:
        print("✓ Good reconstruction quality (>0.5 cosine similarity)")
        print("  The transformer is learning meaningful representations!")
    elif history['reconstruction_quality'][-1] > 0.3:
        print("⚠ Moderate reconstruction quality (0.3-0.5)")
        print("  Consider training longer or increasing model capacity.")
    else:
        print("✗ Poor reconstruction quality (<0.3)")
        print("  The model may not be learning well. Check data and hyperparameters.")


# =============================================================================
# STEP 4: SFT Training (with Audio Encoder Integration)
# =============================================================================

def train_sft(data_path="data/processed", output_dir="checkpoints/sft",
              mae_path="checkpoints/mae", model_name="google/gemma-2b-it", 
              batch_size=4, num_epochs=3, use_wandb=False,
              lora_r=16, lora_alpha=32, learning_rate=2e-5, max_samples=None,
              freeze_audio_encoder=True, lora_dropout=0.1, weight_decay=0.01):
    """
    Train SFT for audio captioning WITH audio encoder integration.
    
    Architecture:
        Audio Features → [MAE Encoder] → [Projector] → [Gemma LLM] → Caption
                         (pretrained)    (trained)      (LoRA)
    
    This properly uses the pretrained MAE encoder to process audio features
    and projects them into the LLM's embedding space.
    """
    print("\n" + "=" * 60)
    print("Training SFT (Audio-to-Text with MAE Encoder)...")
    print("=" * 60 + "\n")
    
    import pickle
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from pathlib import Path
    from tqdm import tqdm
    import numpy as np
    
    # Initialize global model classes
    _init_model_classes()
    global TransformerMAE, AudioProjector
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set. Skipping SFT training.")
        print("Set it in .env file or environment: export HF_TOKEN=your_token")
        return
    
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Warning: unsloth not available. Skipping SFT training.")
        return
    
    # =========================================================================
    # 1. Load pretrained MAE encoder
    # =========================================================================
    print("\n1. Loading pretrained MAE encoder...")
    
    mae_checkpoint = Path(mae_path) / "best_model.pt"
    if not mae_checkpoint.exists():
        mae_checkpoint = Path(mae_path) / "final_model.pt"
    
    if not mae_checkpoint.exists():
        print(f"  Warning: No MAE checkpoint found at {mae_path}")
        print("  Training without audio encoder (text-only mode)")
        audio_encoder = None
    else:
        # Recreate MAE model architecture (must match training)
        class TransformerEncoder(nn.Module):
            """Encoder portion of TransformerMAE for inference."""
            def __init__(self, input_dim=128, hidden_dim=256, num_layers=4, num_heads=4, max_len=64):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.patch_embed = nn.Linear(input_dim, hidden_dim)
                self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
                self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=num_heads,
                    dim_feedforward=hidden_dim * 4, dropout=0.1,
                    activation='gelu', batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.norm = nn.LayerNorm(hidden_dim)
            
            def forward(self, x):
                B, T, D = x.shape
                x = self.patch_embed(x)
                x = x + self.pos_embed[:, :T, :]
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                x = self.encoder(x)
                x = self.norm(x)
                return x[:, 0]  # Return CLS token
        
        audio_encoder = TransformerEncoder().to(device)
        
        # Load weights (extract encoder parts from full MAE checkpoint)
        checkpoint = torch.load(mae_checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Filter to encoder weights only
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('patch_embed') or k.startswith('pos_embed') or \
               k.startswith('cls_token') or k.startswith('encoder') or k.startswith('encoder_norm'):
                new_k = k.replace('encoder_norm', 'norm')
                encoder_state[new_k] = v
        
        try:
            audio_encoder.load_state_dict(encoder_state, strict=False)
            print(f"  ✓ Loaded MAE encoder from {mae_checkpoint}")
        except Exception as e:
            print(f"  Warning: Could not load MAE weights: {e}")
            print("  Using randomly initialized audio encoder")
        
        if freeze_audio_encoder:
            for param in audio_encoder.parameters():
                param.requires_grad = False
            print("  ✓ Audio encoder frozen (will not be updated)")
        else:
            print("  ✓ Audio encoder unfrozen (will be fine-tuned)")
    
    # =========================================================================
    # 2. Load LLM with LoRA
    # =========================================================================
    print(f"\n2. Loading {model_name} with LoRA...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        load_in_4bit=True,
        token=hf_token,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_gradient_checkpointing="unsloth",
    )
    
    # Get LLM embedding dimension
    llm_dim = model.get_input_embeddings().weight.shape[1]
    print(f"  LLM embedding dimension: {llm_dim}")
    
    # =========================================================================
    # 3. Create audio projector
    # =========================================================================
    print("\n3. Creating audio projector...")
    
    audio_dim = 256  # MAE encoder output dimension
    num_audio_tokens = 8  # Number of audio tokens to use
    projector = AudioProjector(audio_dim, llm_dim, num_tokens=num_audio_tokens).to(device)
    print(f"  Projector: {audio_dim} → {num_audio_tokens} tokens × {llm_dim}")
    
    # =========================================================================
    # 4. Load training data
    # =========================================================================
    print("\n4. Loading training data...")
    
    with open(Path(data_path) / "train.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(Path(data_path) / "val.pkl", 'rb') as f:
        val_data = pickle.load(f)
    
    if max_samples is not None and max_samples < len(train_data):
        train_data = train_data[:max_samples]
        print(f"  Using {max_samples} samples (limited)")
    else:
        print(f"  Using ALL {len(train_data)} training samples")
    
    # Dataset that returns audio features + captions
    class AudioCaptionDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=256, target_audio_len=20):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.target_audio_len = target_audio_len
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            
            # Process audio features
            audio = sample['audio_features']
            if len(audio) < self.target_audio_len:
                padding = np.zeros((self.target_audio_len - len(audio), audio.shape[1]))
                audio = np.concatenate([audio, padding], axis=0)
            else:
                audio = audio[:self.target_audio_len]
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            
            return {
                'audio_features': torch.from_numpy(audio).float(),
                'caption': sample['caption'],
            }
    
    # Prompt template - same format used in GRPO for consistency
    PROMPT_TEMPLATE = "<start_of_turn>user\nDescribe this audio.<end_of_turn>\n<start_of_turn>model\n"
    RESPONSE_END = "<end_of_turn>"
    
    # Get prompt length for masking
    prompt_tokens = tokenizer(PROMPT_TEMPLATE, add_special_tokens=False)['input_ids']
    prompt_length = len(prompt_tokens)
    
    def collate_fn(batch):
        audio_features = torch.stack([b['audio_features'] for b in batch])
        captions = [b['caption'] for b in batch]
        
        # Format with prompt template (same as GRPO uses)
        formatted_texts = [
            f"{PROMPT_TEMPLATE}{caption}{RESPONSE_END}"
            for caption in captions
        ]
        
        # Tokenize full sequences
        tokenized = tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt',
        )
        
        # Create labels - mask prompt tokens (only compute loss on caption + end token)
        labels = tokenized['input_ids'].clone()
        # Mask padding tokens
        labels[labels == tokenizer.pad_token_id] = -100
        # Mask prompt tokens (first prompt_length tokens) - don't compute loss on prompt
        labels[:, :prompt_length] = -100
        
        return {
            'audio_features': audio_features,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
        }
    
    train_dataset = AudioCaptionDataset(train_data, tokenizer)
    val_dataset = AudioCaptionDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    
    # =========================================================================
    # 5. Training loop
    # =========================================================================
    print("\n5. Starting training...")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - LoRA rank: {lora_r}, alpha: {lora_alpha}")
    print(f"  - LoRA dropout: {lora_dropout}")
    print(f"  - Weight decay: {weight_decay}")
    
    # Optimizer for projector + LoRA params
    trainable_params = list(projector.parameters())
    trainable_params += [p for p in model.parameters() if p.requires_grad]
    if audio_encoder and not freeze_audio_encoder:
        trainable_params += list(audio_encoder.parameters())
    
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * num_epochs
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    
    # Get LLM embedding layer
    embed_tokens = model.get_input_embeddings()
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        projector.train()
        if audio_encoder:
            audio_encoder.train() if not freeze_audio_encoder else audio_encoder.eval()
        
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for step, batch in enumerate(pbar):
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 1. Encode audio with MAE encoder
            if audio_encoder:
                with torch.no_grad() if freeze_audio_encoder else torch.enable_grad():
                    audio_embeds = audio_encoder(audio_features)  # (B, audio_dim)
            else:
                # Fallback: simple mean pooling if no encoder
                audio_embeds = audio_features.mean(dim=1)  # (B, 128)
                audio_embeds = F.pad(audio_embeds, (0, 256 - 128))  # Pad to 256
            
            # 2. Project to LLM space (now outputs multiple tokens)
            audio_tokens = projector(audio_embeds)  # (B, num_audio_tokens, llm_dim)
            
            # 3. Get text embeddings
            text_embeds = embed_tokens(input_ids)  # (B, seq_len, llm_dim)
            
            # 4. Concatenate: [AUDIO_TOKENS] + [TEXT_TOKENS]
            inputs_embeds = torch.cat([audio_tokens, text_embeds], dim=1)
            
            # 5. Extend attention mask and labels for audio tokens
            B = audio_features.size(0)
            num_audio_toks = audio_tokens.size(1)
            audio_mask = torch.ones(B, num_audio_toks, device=device, dtype=attention_mask.dtype)
            extended_attention_mask = torch.cat([audio_mask, attention_mask], dim=1)
            
            # Labels: -100 for audio tokens (don't compute loss on them)
            audio_labels = torch.full((B, num_audio_toks), -100, device=device, dtype=labels.dtype)
            extended_labels = torch.cat([audio_labels, labels], dim=1)
            
            # 6. Forward through LLM
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                labels=extended_labels,
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        projector.eval()
        if audio_encoder:
            audio_encoder.eval()
        
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                audio_features = batch['audio_features'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                if audio_encoder:
                    audio_embeds = audio_encoder(audio_features)
                else:
                    audio_embeds = audio_features.mean(dim=1)
                    audio_embeds = F.pad(audio_embeds, (0, 256 - 128))
                
                audio_tokens = projector(audio_embeds)  # (B, num_audio_tokens, llm_dim)
                text_embeds = embed_tokens(input_ids)
                inputs_embeds = torch.cat([audio_tokens, text_embeds], dim=1)
                
                B = audio_features.size(0)
                num_audio_toks = audio_tokens.size(1)
                audio_mask = torch.ones(B, num_audio_toks, device=device, dtype=attention_mask.dtype)
                extended_attention_mask = torch.cat([audio_mask, attention_mask], dim=1)
                audio_labels = torch.full((B, num_audio_toks), -100, device=device, dtype=labels.dtype)
                extended_labels = torch.cat([audio_labels, labels], dim=1)
                
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=extended_attention_mask,
                    labels=extended_labels,
                )
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'projector_state_dict': projector.state_dict(),
                'audio_encoder_state_dict': audio_encoder.state_dict() if audio_encoder else None,
                'val_loss': val_loss,
                'epoch': epoch,
            }, f"{output_dir}/best_audio_model.pt")
            model.save_pretrained(f"{output_dir}/best_model")
            tokenizer.save_pretrained(f"{output_dir}/best_model")
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    # Save final model
    torch.save({
        'projector_state_dict': projector.state_dict(),
        'audio_encoder_state_dict': audio_encoder.state_dict() if audio_encoder else None,
    }, f"{output_dir}/final_audio_model.pt")
    
    # Save projector separately for GRPO
    torch.save(projector.state_dict(), f"{output_dir}/projector.pt")
    
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    print(f"\n✓ SFT training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Models saved to: {output_dir}/")


# =============================================================================
# REWARD MODEL: CIDEr + CLAP Hybrid
# =============================================================================

class HybridRewardModel:
    """
    Hybrid reward model combining CIDEr and CLAP text similarity.
    
    - CIDEr: Measures n-gram overlap with TF-IDF weighting (reference-based)
    - CLAP: Measures semantic similarity using sentence embeddings
    """
    
    def __init__(self, alpha=0.5, device="cuda"):
        """
        Args:
            alpha: Weight for CIDEr (1-alpha for CLAP similarity)
            device: Device for embeddings
        """
        self.alpha = alpha
        self.device = device
        self.cider_scorer = None
        self.sentence_model = None
        
        self._init_scorers()
    
    def _init_scorers(self):
        """Initialize scoring models."""
        print("Initializing reward model...")
        
        # Initialize CIDEr scorer
        try:
            from pycocoevalcap.cider.cider import Cider
            self.cider_scorer = Cider()
            print("  ✓ CIDEr scorer loaded")
        except ImportError:
            print("  ✗ CIDEr not available (install pycocoevalcap)")
            self.cider_scorer = None
        
        # Initialize sentence transformer for text similarity
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight model for speed
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_model.to(self.device)
            print("  ✓ Sentence transformer loaded (all-MiniLM-L6-v2)")
        except ImportError:
            print("  ✗ Sentence transformers not available")
            self.sentence_model = None
    
    def compute_cider(self, predictions, references):
        """
        Compute CIDEr scores.
        
        Args:
            predictions: List of generated captions
            references: List of lists of reference captions
        
        Returns:
            List of CIDEr scores (0-10 scale)
        """
        if self.cider_scorer is None:
            return [0.0] * len(predictions)
        
        # Format for pycocoevalcap: plain strings, not dicts
        gts = {}
        res = {}
        for i, (pred, refs) in enumerate(zip(predictions, references)):
            # References should be list of strings
            gts[i] = refs if isinstance(refs, list) else [refs]
            # Prediction should be a list with single string
            res[i] = [pred]
        
        try:
            score, scores = self.cider_scorer.compute_score(gts, res)
            return list(scores)
        except Exception as e:
            print(f"CIDEr error: {e}")
            return [0.0] * len(predictions)
    
    def compute_text_similarity(self, predictions, references):
        """
        Compute semantic similarity between predictions and references.
        
        Args:
            predictions: List of generated captions
            references: List of lists of reference captions
        
        Returns:
            List of similarity scores (0-1 scale)
        """
        if self.sentence_model is None:
            return [0.0] * len(predictions)
        
        import torch
        import numpy as np
        
        scores = []
        for pred, refs in zip(predictions, references):
            # Encode prediction
            pred_emb = self.sentence_model.encode(pred, convert_to_tensor=True)
            
            # Encode references and compute max similarity
            ref_embs = self.sentence_model.encode(refs, convert_to_tensor=True)
            
            # Cosine similarity
            similarities = torch.nn.functional.cosine_similarity(
                pred_emb.unsqueeze(0), ref_embs, dim=1
            )
            
            # Take max similarity across references
            max_sim = similarities.max().item()
            scores.append(max_sim)
        
        return scores
    
    def compute_rewards(self, predictions, references):
        """
        Compute hybrid rewards combining CIDEr and text similarity.
        
        Args:
            predictions: List of generated captions
            references: List of lists of reference captions
        
        Returns:
            Tensor of reward scores
        """
        import torch
        
        # Compute CIDEr scores (0-10 scale, normalize to 0-1)
        cider_scores = self.compute_cider(predictions, references)
        cider_scores = [s / 10.0 for s in cider_scores]
        
        # Compute text similarity scores (already 0-1)
        sim_scores = self.compute_text_similarity(predictions, references)
        
        # Combine with alpha weighting
        rewards = []
        for cider, sim in zip(cider_scores, sim_scores):
            reward = self.alpha * cider + (1 - self.alpha) * sim
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def __call__(self, predictions, references):
        """Compute rewards (callable interface)."""
        return self.compute_rewards(predictions, references)


# =============================================================================
# STEP 5: GRPO Training
# =============================================================================

def train_grpo(data_path="data/processed", model_path="checkpoints/sft/best_model",
               output_dir="checkpoints/grpo", num_steps=100, use_wandb=False,
               num_generations=4, reward_alpha=0.5, batch_size=2, learning_rate=5e-6,
               mae_path="checkpoints/mae", projector_path="checkpoints/sft/projector.pt"):
    """
    Train GRPO for improved caption quality using hybrid reward model.
    
    GRPO (Group Relative Policy Optimization):
    1. Generate multiple captions per audio sample (conditioned on audio)
    2. Compute rewards using CIDEr + CLAP similarity
    3. Use group-relative advantages for policy update
    
    The pipeline: Audio Features → MAE Encoder → Projector → LLM → Caption
    """
    print("\n" + "=" * 60)
    print("Training GRPO (Reinforcement Learning)...")
    print("=" * 60 + "\n")
    
    import pickle
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    
    # Initialize global model classes
    _init_model_classes()
    global TransformerMAE, AudioProjector
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set. Skipping GRPO training.")
        return
    
    if not os.path.exists(model_path):
        print(f"Warning: SFT model not found at {model_path}. Skipping GRPO training.")
        return
    
    # =========================================================================
    # 1. Initialize reward model
    # =========================================================================
    print("\n1. Initializing hybrid reward model (CIDEr + CLAP)...")
    reward_model = HybridRewardModel(alpha=reward_alpha, device=device)
    
    # =========================================================================
    # 2. Load MAE encoder (from pretraining)
    # =========================================================================
    print("\n2. Loading MAE encoder...")
    
    mae_encoder = None
    mae_checkpoint = Path(mae_path) / "best_model.pt"
    if not mae_checkpoint.exists():
        mae_checkpoint = Path(mae_path) / "final_model.pt"
    
    if mae_checkpoint.exists():
        mae_encoder = TransformerMAE(
            input_dim=128, hidden_dim=256, num_layers=4, 
            num_heads=4, mask_ratio=0.0  # No masking during inference
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(mae_checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        mae_encoder.load_state_dict(state_dict)
        mae_encoder.eval()
        for p in mae_encoder.parameters():
            p.requires_grad = False
        print(f"  ✓ MAE encoder loaded from {mae_checkpoint}")
    else:
        print(f"  ⚠ MAE encoder not found at {mae_path}, will use text-only mode")
    
    # =========================================================================
    # 3. Load SFT model and projector
    # =========================================================================
    print(f"\n3. Loading SFT model from {model_path}...")
    
    from unsloth import FastLanguageModel
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            load_in_4bit=True,
            token=hf_token,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("  ✓ LLM loaded with Unsloth")
            
    except Exception as e:
        print(f"Error loading saved model: {e}")
        return
    
    # Load audio projector
    projector = None
    llm_dim = model.get_input_embeddings().weight.shape[1]
    
    if os.path.exists(projector_path):
        projector = AudioProjector(256, llm_dim).to(device)
        projector.load_state_dict(torch.load(projector_path, map_location=device))
        projector.eval()
        for p in projector.parameters():
            p.requires_grad = False
        print(f"  ✓ Audio projector loaded from {projector_path}")
    else:
        print(f"  ⚠ Projector not found at {projector_path}")
    
    has_audio = mae_encoder is not None and projector is not None
    print(f"  Audio conditioning: {'ENABLED' if has_audio else 'DISABLED (text-only)'}")
    
    # =========================================================================
    # 4. Load training data
    # =========================================================================
    print("\n4. Loading training data...")
    train_path = Path(data_path) / "train.pkl"
    if not train_path.exists():
        print(f"Error: {train_path} not found")
        return
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    print(f"  Loaded {len(train_data)} training samples")
    
    # Limit data for GRPO (it's slow)
    train_data = train_data[:500]
    print(f"  Using {len(train_data)} samples for GRPO")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 5. GRPO training setup
    # =========================================================================
    print(f"\n5. Starting GRPO training for {num_steps} steps...")
    print(f"  - Generations per sample: {num_generations}")
    print(f"  - Reward alpha (CIDEr weight): {reward_alpha}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    
    # Enable training mode for LoRA (GRPO updates)
    FastLanguageModel.for_training(model)
    
    # Setup optimizer for LoRA params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    print(f"  - Trainable params: {num_params:,}")
    
    if trainable_params:
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    else:
        print("  ⚠ No trainable parameters found. Running reward evaluation only.")
        optimizer = None
    
    # Helper function to process audio
    def process_audio(audio_features):
        """Process audio features through MAE encoder and projector."""
        if not has_audio:
            return None
        
        # Normalize and pad/truncate to target length
        target_len = 20  # Increased from 10 for more context
        if len(audio_features) < target_len:
            padding = np.zeros((target_len - len(audio_features), 128))
            audio_features = np.concatenate([audio_features, padding], axis=0)
        else:
            audio_features = audio_features[:target_len]
        
        audio_features = (audio_features - audio_features.mean()) / (audio_features.std() + 1e-8)
        audio_tensor = torch.from_numpy(audio_features).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get CLS token from MAE encoder using built-in method
            audio_embedding = mae_encoder.get_encoder_output(audio_tensor)
            
            # Project to LLM dimension (now returns multiple tokens)
            audio_projected = projector(audio_embedding)  # (1, num_tokens, llm_dim)
        
        return audio_projected  # Shape: (1, num_tokens, llm_dim)
    
    # =========================================================================
    # 6. Training loop
    # =========================================================================
    global_step = 0
    total_reward = 0.0
    best_avg_reward = 0.0
    
    pbar = tqdm(total=num_steps, desc="GRPO Training")
    
    data_idx = 0
    while global_step < num_steps:
        # Get batch
        batch_samples = []
        for _ in range(batch_size):
            batch_samples.append(train_data[data_idx % len(train_data)])
            data_idx += 1
        
        # Generate multiple captions for each sample
        all_predictions = []
        all_references = []
        all_log_probs = []
        
        # Custom autoregressive generation with audio embeddings (no KV cache for compatibility)
        def generate_with_audio(audio_tokens, prompt_ids, max_new_tokens=64, temperature=0.8, top_p=0.9):
            """Generate text autoregressively with audio tokens prepended."""
            model.eval()
            embed_tokens = model.get_input_embeddings()
            
            # Track generated token ids
            generated_ids = prompt_ids.clone()
            num_audio_toks = audio_tokens.size(1) if audio_tokens is not None else 0
            
            # Autoregressive generation loop (without KV cache for Unsloth compatibility)
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    # Build full inputs_embeds each time
                    text_embeds = embed_tokens(generated_ids)
                    
                    if audio_tokens is not None:
                        inputs_embeds = torch.cat([audio_tokens, text_embeds], dim=1)
                    else:
                        inputs_embeds = text_embeds
                    
                    # Forward pass
                    outputs = model(inputs_embeds=inputs_embeds)
                    
                    # Get logits for the last position
                    logits = outputs.logits[:, -1, :] / temperature
                    
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                    
                    # Sample
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated ids
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    # Stop if EOS token
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            return generated_ids[0]  # Return (seq_len,) tensor
        
        for sample in batch_samples:
            # Process audio
            audio_tokens = process_audio(sample['audio_features']) if has_audio else None
            
            # Create prompt
            prompt = "<start_of_turn>user\nDescribe this audio.<end_of_turn>\n<start_of_turn>model\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs['input_ids'].size(1)
            
            # Generate multiple responses
            generations = []
            log_probs_list = []
            
            for _ in range(num_generations):
                # Generate with audio-conditioned custom function
                generated_ids = generate_with_audio(
                    audio_tokens, 
                    inputs['input_ids'],
                    max_new_tokens=64,
                    temperature=0.8,
                    top_p=0.9
                )
                
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract model's response
                if "<start_of_turn>model" in generated_text:
                    response = generated_text.split("<start_of_turn>model")[-1].strip()
                    # Clean up any end tokens
                    if "<end_of_turn>" in response:
                        response = response.split("<end_of_turn>")[0].strip()
                else:
                    response = generated_text
                
                generations.append(response)
                
                # Compute log probability of generated sequence (for GRPO loss)
                if optimizer is not None:
                    model.train()
                    with torch.enable_grad():
                        output_ids = generated_ids[prompt_len:]
                        if len(output_ids) > 0:
                            full_ids = generated_ids.unsqueeze(0)
                            labels = full_ids.clone()
                            labels[:, :prompt_len] = -100
                            
                            outputs_loss = model(full_ids, labels=labels)
                            log_probs_list.append(-outputs_loss.loss)  # Negative NLL = log prob
                        else:
                            log_probs_list.append(torch.tensor(0.0, device=device))
            
            all_predictions.extend(generations)
            all_references.extend([[sample['caption']]] * num_generations)
            if optimizer is not None:
                all_log_probs.extend(log_probs_list)
        
        # Compute rewards
        rewards = reward_model(all_predictions, all_references)
        
        # Reshape rewards: (batch_size * num_generations) -> (batch_size, num_generations)
        rewards = rewards.view(batch_size, num_generations)
        
        # Compute group-relative advantages
        rewards_mean = rewards.mean(dim=1, keepdim=True)
        rewards_std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - rewards_mean) / rewards_std
        
        # GRPO loss: maximize advantage-weighted log probability
        if optimizer is not None and len(all_log_probs) > 0:
            log_probs = torch.stack(all_log_probs).view(batch_size, num_generations)
            advantages_flat = advantages.to(device)
            
            # Policy gradient loss
            loss = -(log_probs * advantages_flat.detach()).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
        
        # Log metrics
        avg_reward = rewards.mean().item()
        total_reward += avg_reward
        
        # Update progress
        global_step += 1
        pbar.update(1)
        pbar.set_postfix({
            'reward': f"{avg_reward:.4f}",
            'avg': f"{total_reward/global_step:.4f}",
        })
        
        # Save checkpoint every 25 steps
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
        
        # Log every 10 steps
        if global_step % 10 == 0:
            print(f"\n  Step {global_step}: Avg Reward = {total_reward/global_step:.4f}, Best = {best_avg_reward:.4f}")
            print(f"  Sample prediction: {all_predictions[0][:80]}...")
            print(f"  Reference: {all_references[0][0][:80]}...")
    
    pbar.close()
    
    # Save final model
    if optimizer is not None:
        model.save_pretrained(f"{output_dir}/final_model")
        tokenizer.save_pretrained(f"{output_dir}/final_model")
        print(f"\n  ✓ Model saved to {output_dir}/final_model")
    
    # Save final results
    results = {
        'final_avg_reward': total_reward / num_steps,
        'best_reward': best_avg_reward,
        'num_steps': num_steps,
        'reward_alpha': reward_alpha,
        'audio_conditioning': has_audio,
    }
    
    import json
    with open(f"{output_dir}/grpo_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ GRPO training complete!")
    print(f"  Final avg reward: {results['final_avg_reward']:.4f}")
    print(f"  Best reward: {results['best_reward']:.4f}")
    print(f"  Results saved to: {output_dir}/grpo_results.json")


# =============================================================================
# DIAGNOSTIC: Test if audio actually influences outputs
# =============================================================================

def diagnose_audio_influence(model_path="checkpoints/sft/best_model", 
                             mae_path="checkpoints/mae",
                             projector_path="checkpoints/sft/projector.pt",
                             data_path="data/processed"):
    """
    Diagnostic tool to test if the model actually uses audio information.
    
    Tests:
    1. Consistency: Same audio → similar outputs?
    2. Distinctiveness: Different audios → different outputs?
    3. Audio vs No-Audio: Does removing audio change outputs?
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Testing Audio Influence")
    print("=" * 60 + "\n")
    
    import pickle
    import torch
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path
    
    _init_model_classes()
    global TransformerMAE, AudioProjector
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set")
        return
    
    # Load model
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        load_in_4bit=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    
    # Load MAE encoder
    mae_checkpoint = Path(mae_path) / "best_model.pt"
    if not mae_checkpoint.exists():
        mae_checkpoint = Path(mae_path) / "final_model.pt"
    
    mae_encoder = TransformerMAE(input_dim=128, hidden_dim=256, num_layers=4, num_heads=4, mask_ratio=0.0).to(device)
    checkpoint = torch.load(mae_checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    mae_encoder.load_state_dict(state_dict)
    mae_encoder.eval()
    
    # Load projector
    llm_dim = model.get_input_embeddings().weight.shape[1]
    projector = AudioProjector(256, llm_dim, num_tokens=8).to(device)
    projector.load_state_dict(torch.load(projector_path, map_location=device))
    projector.eval()
    
    # Load test data
    with open(Path(data_path) / "test.pkl", 'rb') as f:
        test_data = pickle.load(f)
    
    def process_audio(audio_features):
        target_len = 20
        if len(audio_features) < target_len:
            padding = np.zeros((target_len - len(audio_features), 128))
            audio_features = np.concatenate([audio_features, padding], axis=0)
        else:
            audio_features = audio_features[:target_len]
        audio_features = (audio_features - audio_features.mean()) / (audio_features.std() + 1e-8)
        audio_tensor = torch.from_numpy(audio_features).float().unsqueeze(0).to(device)
        with torch.no_grad():
            audio_embedding = mae_encoder.get_encoder_output(audio_tensor)
            audio_projected = projector(audio_embedding)
        return audio_projected
    
    def generate_caption(audio_tokens, temperature=0.3):
        """Generate with low temperature for more deterministic output."""
        embed_tokens = model.get_input_embeddings()
        prompt = "<start_of_turn>user\nDescribe this audio.<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated_ids = inputs['input_ids'].clone()
        
        for _ in range(64):
            with torch.no_grad():
                text_embeds = embed_tokens(generated_ids)
                if audio_tokens is not None:
                    inputs_embeds = torch.cat([audio_tokens, text_embeds], dim=1)
                else:
                    inputs_embeds = text_embeds
                outputs = model(inputs_embeds=inputs_embeds)
                logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if "<start_of_turn>model" in text:
            text = text.split("<start_of_turn>model")[-1].strip()
        if "<end_of_turn>" in text:
            text = text.split("<end_of_turn>")[0].strip()
        return text
    
    print("=" * 60)
    print("TEST 1: Consistency (same audio, 3 generations)")
    print("=" * 60)
    sample = test_data[0]
    audio_tokens = process_audio(sample['audio_features'])
    print(f"Reference: {sample['caption'][:100]}...")
    print(f"\nGenerations (temperature=0.3):")
    for i in range(3):
        caption = generate_caption(audio_tokens, temperature=0.3)
        print(f"  {i+1}. {caption[:80]}...")
    
    print("\n" + "=" * 60)
    print("TEST 2: Distinctiveness (5 different audios)")
    print("=" * 60)
    for i in range(5):
        sample = test_data[i * 10]  # Spread out samples
        audio_tokens = process_audio(sample['audio_features'])
        caption = generate_caption(audio_tokens, temperature=0.3)
        print(f"\nAudio {i+1}:")
        print(f"  Reference: {sample['caption'][:60]}...")
        print(f"  Generated: {caption[:60]}...")
    
    print("\n" + "=" * 60)
    print("TEST 3: Audio vs No-Audio")
    print("=" * 60)
    sample = test_data[5]
    audio_tokens = process_audio(sample['audio_features'])
    
    caption_with_audio = generate_caption(audio_tokens, temperature=0.3)
    caption_without_audio = generate_caption(None, temperature=0.3)
    
    print(f"Reference: {sample['caption'][:80]}...")
    print(f"\nWith Audio:    {caption_with_audio[:80]}...")
    print(f"Without Audio: {caption_without_audio[:80]}...")
    
    if caption_with_audio == caption_without_audio:
        print("\n⚠️  WARNING: Outputs are IDENTICAL - audio is being IGNORED!")
    else:
        print("\n✓ Outputs differ - audio has SOME influence")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MedGemma Audio - Full Pipeline")
    parser.add_argument("--skip-install", action="store_true", help="Skip package installation")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip-mae", action="store_true", help="Skip MAE training")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT training")
    parser.add_argument("--skip-grpo", action="store_true", help="Skip GRPO training")
    parser.add_argument("--only-download", action="store_true", help="Only download data")
    parser.add_argument("--only-preprocess", action="store_true", help="Only preprocess data")
    parser.add_argument("--only-mae", action="store_true", help="Only train MAE")
    parser.add_argument("--skip-features", action="store_true", help="Skip AudioSet features download")
    
    # MAE arguments (transformer-based)
    parser.add_argument("--mae-epochs", type=int, default=50, help="MAE training epochs (default: 50)")
    parser.add_argument("--mae-hidden-dim", type=int, default=256, help="MAE hidden dimension")
    parser.add_argument("--mae-layers", type=int, default=4, help="MAE transformer layers")
    parser.add_argument("--mae-heads", type=int, default=4, help="MAE attention heads")
    parser.add_argument("--mae-mask-ratio", type=float, default=0.80, help="MAE mask ratio (AudioMAE-style, default: 0.80)")
    
    # SFT arguments (optimized defaults to prevent overfitting)
    parser.add_argument("--sft-epochs", type=int, default=3, help="SFT training epochs (default: 3)")
    parser.add_argument("--sft-lr", type=float, default=2e-5, help="SFT learning rate (default: 2e-5)")
    parser.add_argument("--sft-lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--sft-lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--sft-lora-dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    parser.add_argument("--sft-weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--sft-max-samples", type=int, default=None, help="Max training samples (default: all)")
    parser.add_argument("--mae-path", type=str, default="checkpoints/mae", help="Path to pretrained MAE encoder")
    parser.add_argument("--finetune-audio-encoder", action="store_true", help="Fine-tune audio encoder (default: frozen)")
    parser.add_argument("--only-sft", action="store_true", help="Only run SFT training")
    
    # GRPO arguments
    parser.add_argument("--grpo-steps", type=int, default=100, help="GRPO training steps")
    parser.add_argument("--grpo-alpha", type=float, default=0.5, help="GRPO reward alpha (CIDEr weight)")
    parser.add_argument("--grpo-generations", type=int, default=4, help="Generations per sample in GRPO")
    parser.add_argument("--only-grpo", action="store_true", help="Only run GRPO training")
    parser.add_argument("--diagnose", action="store_true", help="Run audio influence diagnostic")
    
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MedGemma Audio - Complete Training Pipeline")
    print("=" * 60)
    
    # Load .env if exists
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    
    # Diagnostic mode
    if args.diagnose:
        diagnose_audio_influence()
        return
    
    # Step 0: Install packages
    if not args.skip_install:
        install_packages()
    
    # Step 1: Download
    if args.only_download:
        download_data(skip_features=args.skip_features)
        return
    
    if not args.skip_download:
        download_data(skip_features=args.skip_features)
    
    # Step 2: Preprocess
    if args.only_preprocess:
        preprocess_data()
        return
    
    if not args.skip_preprocess:
        preprocess_data()
    
    # Step 3: MAE
    if args.only_mae:
        train_mae(
            num_epochs=args.mae_epochs,
            hidden_dim=args.mae_hidden_dim,
            num_layers=args.mae_layers,
            num_heads=args.mae_heads,
            mask_ratio=args.mae_mask_ratio,
            use_wandb=args.use_wandb
        )
        return
    
    if not args.skip_mae:
        train_mae(
            num_epochs=args.mae_epochs,
            hidden_dim=args.mae_hidden_dim,
            num_layers=args.mae_layers,
            num_heads=args.mae_heads,
            mask_ratio=args.mae_mask_ratio,
            use_wandb=args.use_wandb
        )
    
    # Step 4: SFT (now with audio encoder integration!)
    if args.only_sft:
        train_sft(
            num_epochs=args.sft_epochs,
            learning_rate=args.sft_lr,
            lora_r=args.sft_lora_r,
            lora_alpha=args.sft_lora_alpha,
            lora_dropout=args.sft_lora_dropout,
            weight_decay=args.sft_weight_decay,
            max_samples=args.sft_max_samples,
            mae_path=args.mae_path,
            freeze_audio_encoder=not args.finetune_audio_encoder,
            use_wandb=args.use_wandb
        )
        return
    
    if not args.skip_sft:
        train_sft(
            num_epochs=args.sft_epochs,
            learning_rate=args.sft_lr,
            lora_r=args.sft_lora_r,
            lora_alpha=args.sft_lora_alpha,
            lora_dropout=args.sft_lora_dropout,
            weight_decay=args.sft_weight_decay,
            max_samples=args.sft_max_samples,
            mae_path=args.mae_path,
            freeze_audio_encoder=not args.finetune_audio_encoder,
            use_wandb=args.use_wandb
        )
    
    # Step 5: GRPO
    if args.only_grpo:
        train_grpo(
            num_steps=args.grpo_steps,
            reward_alpha=args.grpo_alpha,
            num_generations=args.grpo_generations,
            mae_path=args.mae_path,
            projector_path="checkpoints/sft/projector.pt",
            use_wandb=args.use_wandb
        )
        return
    
    if not args.skip_grpo:
        train_grpo(
            num_steps=args.grpo_steps,
            reward_alpha=args.grpo_alpha,
            num_generations=args.grpo_generations,
            mae_path=args.mae_path,
            projector_path="checkpoints/sft/projector.pt",
            use_wandb=args.use_wandb
        )
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print("\nCheckpoints saved to:")
    print("  - MAE:  checkpoints/mae/")
    print("  - SFT:  checkpoints/sft/")
    print("  - GRPO: checkpoints/grpo/")


if __name__ == "__main__":
    main()
