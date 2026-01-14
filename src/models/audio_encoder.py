"""
Audio Encoder with Masked Autoencoder (MAE) for AudioSet features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu if activation == "gelu" else F.relu
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feedforward with residual
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x


class AudioEncoder(nn.Module):
    """
    Transformer-based audio encoder for 128-dim AudioSet features.
    
    Processes sequences of (T, 128) audio features and produces
    contextualized embeddings.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        """
        Args:
            input_dim: Input feature dimension (128 for AudioSet)
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # CLS token for pooled representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            mask: Optional padding mask of shape (batch, seq_len)
        
        Returns:
            Tuple of (sequence_output, pooled_output)
            - sequence_output: (batch, seq_len + 1, d_model)
            - pooled_output: (batch, d_model)
        """
        batch_size = x.size(0)
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Update mask for CLS token if provided
        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        
        # Final layer norm
        x = self.norm(x)
        
        # Split CLS and sequence outputs
        sequence_output = x
        pooled_output = x[:, 0]  # CLS token
        
        return sequence_output, pooled_output
    
    def get_output_dim(self) -> int:
        return self.d_model


class MAEDecoder(nn.Module):
    """
    Lightweight decoder for MAE reconstruction.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        decoder_dim: int = 256,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        output_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.decoder_dim = decoder_dim
        
        # Project from encoder to decoder dimension
        self.decoder_embed = nn.Linear(d_model, decoder_dim)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim))
        
        # Positional encoding for decoder
        self.pos_encoder = PositionalEncoding(decoder_dim, max_seq_len, dropout)
        
        # Decoder transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(decoder_dim, decoder_heads, decoder_dim * 4, dropout)
            for _ in range(decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(decoder_dim, output_dim)
        
        self.norm = nn.LayerNorm(decoder_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Encoded visible patches of shape (batch, num_visible, d_model)
            ids_restore: Indices for restoring original order (batch, num_patches)
        
        Returns:
            Reconstructed features of shape (batch, num_patches, output_dim)
        """
        batch_size, num_visible, _ = x.shape
        num_patches = ids_restore.shape[1]
        num_masked = num_patches - num_visible
        
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Expand mask tokens
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)
        
        # Concatenate visible and mask tokens
        x = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to restore original order
        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        )
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Project to output dimension
        x = self.output_projection(x)
        
        return x


class AudioMAE(nn.Module):
    """
    Masked Autoencoder for Audio (using AudioSet features).
    
    This model learns audio representations by masking random patches
    of the input and reconstructing them.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        encoder_dim: int = 512,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        decoder_dim: int = 256,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        """
        Args:
            input_dim: Input feature dimension (128 for AudioSet)
            encoder_dim: Encoder transformer dimension
            encoder_layers: Number of encoder layers
            encoder_heads: Number of encoder attention heads
            decoder_dim: Decoder transformer dimension
            decoder_layers: Number of decoder layers
            decoder_heads: Number of decoder attention heads
            mask_ratio: Ratio of patches to mask
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.mask_ratio = mask_ratio
        
        # Encoder (without CLS token for MAE)
        self.encoder = nn.ModuleDict({
            'input_projection': nn.Linear(input_dim, encoder_dim),
            'pos_encoder': PositionalEncoding(encoder_dim, max_seq_len, dropout),
            'layers': nn.ModuleList([
                TransformerEncoderBlock(encoder_dim, encoder_heads, encoder_dim * 4, dropout)
                for _ in range(encoder_layers)
            ]),
            'norm': nn.LayerNorm(encoder_dim),
        })
        
        self.encoder_dim = encoder_dim
        
        # Decoder
        self.decoder = MAEDecoder(
            d_model=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            output_dim=input_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask_ratio: Override default mask ratio
        
        Returns:
            Tuple of (x_masked, mask, ids_restore)
        """
        batch_size, seq_len, dim = x.shape
        mask_ratio = mask_ratio or self.mask_ratio
        
        num_keep = int(seq_len * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(batch_size, seq_len, device=x.device)
        
        # Sort noise to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first num_keep indices
        ids_keep = ids_shuffle[:, :num_keep]
        
        # Gather kept patches
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, dim)
        )
        
        # Generate binary mask: 0 = kept, 1 = masked
        mask = torch.ones(batch_size, seq_len, device=x.device)
        mask[:, :num_keep] = 0
        # Unshuffle to get mask in original order
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode audio features (without masking).
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
        
        Returns:
            Encoded features of shape (batch, seq_len, encoder_dim)
        """
        x = self.encoder['input_projection'](x)
        x = self.encoder['pos_encoder'](x)
        
        for layer in self.encoder['layers']:
            x = layer(x)
        
        x = self.encoder['norm'](x)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with masking and reconstruction.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            mask_ratio: Override default mask ratio
        
        Returns:
            Tuple of (reconstruction, target, mask)
            - reconstruction: (batch, seq_len, input_dim)
            - target: Original input (batch, seq_len, input_dim)
            - mask: Binary mask (batch, seq_len)
        """
        target = x.clone()
        
        # Project input
        x = self.encoder['input_projection'](x)
        
        # Mask
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Add positional encoding to visible patches
        # Note: We need to handle positions correctly for visible patches
        batch_size = x.size(0)
        visible_positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.encoder['pos_encoder'].pe[:, :x.size(1), :]
        
        # Encode visible patches
        for layer in self.encoder['layers']:
            x = layer(x)
        
        x = self.encoder['norm'](x)
        
        # Decode
        reconstruction = self.decoder(x, ids_restore)
        
        return reconstruction, target, mask
    
    def compute_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss on masked patches.
        
        Args:
            reconstruction: Reconstructed features
            target: Original features
            mask: Binary mask (1 = masked)
        
        Returns:
            Mean squared error on masked patches
        """
        # Compute MSE
        loss = (reconstruction - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over feature dimension
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def get_encoder(self) -> nn.Module:
        """
        Get the encoder for downstream tasks.
        
        Returns:
            AudioEncoder instance initialized from MAE encoder weights
        """
        encoder = AudioEncoder(
            input_dim=self.input_dim,
            d_model=self.encoder_dim,
            nhead=self.encoder['layers'][0].self_attn.num_heads,
            num_layers=len(self.encoder['layers']),
        )
        
        # Copy weights
        encoder.input_projection.load_state_dict(
            self.encoder['input_projection'].state_dict()
        )
        encoder.norm.load_state_dict(self.encoder['norm'].state_dict())
        
        for i, layer in enumerate(encoder.layers):
            layer.load_state_dict(self.encoder['layers'][i].state_dict())
        
        return encoder
