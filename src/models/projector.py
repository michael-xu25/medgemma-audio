"""
Audio Projector for mapping audio embeddings to LLM embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AudioProjector(nn.Module):
    """
    MLP projector to map audio encoder outputs to MedGemma embedding space.
    
    This bridges the audio encoder with the language model.
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        llm_dim: int = 2048,  # MedGemma 4B hidden dimension
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Args:
            audio_dim: Audio encoder output dimension
            llm_dim: LLM embedding dimension
            hidden_dim: Hidden layer dimension (defaults to mean of audio_dim and llm_dim)
            num_layers: Number of MLP layers
            dropout: Dropout rate
            activation: Activation function ('gelu' or 'relu')
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim or (audio_dim + llm_dim) // 2
        
        # Build MLP layers
        layers = []
        in_dim = audio_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = self.hidden_dim
        
        # Final projection
        layers.append(nn.Linear(in_dim, llm_dim))
        
        self.projector = nn.Sequential(*layers)
        
        # Layer norm for output
        self.output_norm = nn.LayerNorm(llm_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project audio embeddings to LLM space.
        
        Args:
            x: Audio embeddings of shape (batch, seq_len, audio_dim) or (batch, audio_dim)
        
        Returns:
            Projected embeddings of shape (batch, seq_len, llm_dim) or (batch, llm_dim)
        """
        x = self.projector(x)
        x = self.output_norm(x)
        return x


class QFormerProjector(nn.Module):
    """
    Q-Former style projector using learnable queries.
    
    This approach uses cross-attention to compress variable-length
    audio sequences into a fixed number of tokens.
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        llm_dim: int = 2048,
        num_queries: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            audio_dim: Audio encoder output dimension
            llm_dim: LLM embedding dimension
            num_queries: Number of learnable query tokens
            num_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim))
        
        # Project audio features to LLM dimension
        self.audio_projection = nn.Linear(audio_dim, llm_dim)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    llm_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'norm1': nn.LayerNorm(llm_dim),
                'ffn': nn.Sequential(
                    nn.Linear(llm_dim, llm_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(llm_dim * 4, llm_dim),
                ),
                'norm2': nn.LayerNorm(llm_dim),
                'dropout': nn.Dropout(dropout),
            })
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(llm_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project audio features using Q-Former.
        
        Args:
            audio_features: Audio embeddings of shape (batch, seq_len, audio_dim)
            audio_mask: Optional attention mask (batch, seq_len)
        
        Returns:
            Projected tokens of shape (batch, num_queries, llm_dim)
        """
        batch_size = audio_features.size(0)
        
        # Project audio features
        audio_features = self.audio_projection(audio_features)
        
        # Expand queries for batch
        queries = self.queries.expand(batch_size, -1, -1)
        
        # Apply cross-attention layers
        for layer in self.cross_attn_layers:
            # Cross-attention
            attn_out, _ = layer['cross_attn'](
                queries, audio_features, audio_features,
                key_padding_mask=audio_mask
            )
            queries = queries + layer['dropout'](attn_out)
            queries = layer['norm1'](queries)
            
            # FFN
            ffn_out = layer['ffn'](queries)
            queries = queries + layer['dropout'](ffn_out)
            queries = layer['norm2'](queries)
        
        return self.output_norm(queries)


class PerceiverProjector(nn.Module):
    """
    Perceiver-style projector with cross-attention and self-attention.
    
    More powerful than simple MLP but more compute-intensive.
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        llm_dim: int = 2048,
        num_latents: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            audio_dim: Audio encoder output dimension
            llm_dim: LLM embedding dimension
            num_latents: Number of latent tokens
            num_layers: Number of perceiver layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.llm_dim = llm_dim
        self.num_latents = num_latents
        
        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(1, num_latents, llm_dim))
        
        # Project audio to latent dimension
        self.audio_projection = nn.Linear(audio_dim, llm_dim)
        
        # Perceiver layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # Cross-attention (latents attend to audio)
                'cross_attn': nn.MultiheadAttention(
                    llm_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'cross_norm': nn.LayerNorm(llm_dim),
                # Self-attention (latents attend to themselves)
                'self_attn': nn.MultiheadAttention(
                    llm_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'self_norm': nn.LayerNorm(llm_dim),
                # FFN
                'ffn': nn.Sequential(
                    nn.Linear(llm_dim, llm_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(llm_dim * 4, llm_dim),
                ),
                'ffn_norm': nn.LayerNorm(llm_dim),
                'dropout': nn.Dropout(dropout),
            }))
        
        self.output_norm = nn.LayerNorm(llm_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project audio features using Perceiver.
        
        Args:
            audio_features: Audio embeddings of shape (batch, seq_len, audio_dim)
            audio_mask: Optional attention mask (batch, seq_len)
        
        Returns:
            Projected tokens of shape (batch, num_latents, llm_dim)
        """
        batch_size = audio_features.size(0)
        
        # Project audio features
        audio_features = self.audio_projection(audio_features)
        
        # Expand latents for batch
        latents = self.latents.expand(batch_size, -1, -1)
        
        # Apply perceiver layers
        for layer in self.layers:
            # Cross-attention
            cross_out, _ = layer['cross_attn'](
                latents, audio_features, audio_features,
                key_padding_mask=audio_mask
            )
            latents = latents + layer['dropout'](cross_out)
            latents = layer['cross_norm'](latents)
            
            # Self-attention
            self_out, _ = layer['self_attn'](latents, latents, latents)
            latents = latents + layer['dropout'](self_out)
            latents = layer['self_norm'](latents)
            
            # FFN
            ffn_out = layer['ffn'](latents)
            latents = latents + layer['dropout'](ffn_out)
            latents = layer['ffn_norm'](latents)
        
        return self.output_norm(latents)


def create_projector(
    projector_type: str = "mlp",
    audio_dim: int = 512,
    llm_dim: int = 2048,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create audio projector.
    
    Args:
        projector_type: Type of projector ('mlp', 'qformer', 'perceiver')
        audio_dim: Audio encoder output dimension
        llm_dim: LLM embedding dimension
        **kwargs: Additional arguments for specific projector types
    
    Returns:
        Projector module
    """
    if projector_type == "mlp":
        return AudioProjector(audio_dim, llm_dim, **kwargs)
    elif projector_type == "qformer":
        return QFormerProjector(audio_dim, llm_dim, **kwargs)
    elif projector_type == "perceiver":
        return PerceiverProjector(audio_dim, llm_dim, **kwargs)
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")
