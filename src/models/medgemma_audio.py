"""
MedGemma Audio model combining audio encoder with MedGemma LLM.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from .audio_encoder import AudioEncoder, AudioMAE
from .projector import create_projector, AudioProjector


@dataclass
class MedGemmaAudioConfig:
    """Configuration for MedGemma Audio model."""
    
    # Audio encoder config
    audio_input_dim: int = 128
    audio_encoder_dim: int = 512
    audio_encoder_layers: int = 6
    audio_encoder_heads: int = 8
    audio_max_seq_len: int = 64
    
    # Projector config
    projector_type: str = "mlp"  # 'mlp', 'qformer', 'perceiver'
    projector_hidden_dim: Optional[int] = None
    projector_num_layers: int = 2
    num_audio_tokens: int = 8  # For qformer/perceiver
    
    # LLM config (MedGemma 4B)
    llm_model_name: str = "google/medgemma-4b-it"
    llm_dim: int = 2048
    
    # Training config
    freeze_llm: bool = True
    freeze_audio_encoder: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class MedGemmaAudio(nn.Module):
    """
    MedGemma Audio model for audio captioning.
    
    Architecture:
    1. Audio Encoder: Processes 128-dim AudioSet features
    2. Audio Projector: Maps audio embeddings to LLM space
    3. MedGemma LLM: Generates captions conditioned on audio
    """
    
    def __init__(
        self,
        config: MedGemmaAudioConfig,
        audio_encoder: Optional[AudioEncoder] = None,
        llm_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            config: Model configuration
            audio_encoder: Pre-trained audio encoder (optional)
            llm_model: Pre-loaded LLM model (optional)
            tokenizer: LLM tokenizer
        """
        super().__init__()
        
        self.config = config
        self.tokenizer = tokenizer
        
        # Audio encoder
        if audio_encoder is not None:
            self.audio_encoder = audio_encoder
        else:
            self.audio_encoder = AudioEncoder(
                input_dim=config.audio_input_dim,
                d_model=config.audio_encoder_dim,
                nhead=config.audio_encoder_heads,
                num_layers=config.audio_encoder_layers,
                max_seq_len=config.audio_max_seq_len,
            )
        
        # Audio projector
        self.audio_projector = create_projector(
            projector_type=config.projector_type,
            audio_dim=config.audio_encoder_dim,
            llm_dim=config.llm_dim,
            hidden_dim=config.projector_hidden_dim,
            num_layers=config.projector_num_layers,
            num_queries=config.num_audio_tokens if config.projector_type != "mlp" else None,
            num_latents=config.num_audio_tokens if config.projector_type == "perceiver" else None,
        )
        
        # LLM (will be loaded separately with Unsloth)
        self.llm = llm_model
        
        # Special token for audio
        self.audio_token_id = None
        
        # Freeze components as needed
        if config.freeze_audio_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
    
    def set_llm(self, llm_model: nn.Module, tokenizer: Any):
        """Set the LLM model and tokenizer."""
        self.llm = llm_model
        self.tokenizer = tokenizer
        
        # Add special audio token if not present
        if "<audio>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<audio>"]})
            # Resize embeddings
            if hasattr(llm_model, 'resize_token_embeddings'):
                llm_model.resize_token_embeddings(len(tokenizer))
        
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<audio>")
    
    def encode_audio(
        self,
        audio_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode audio features and project to LLM space.
        
        Args:
            audio_features: Audio features of shape (batch, seq_len, 128)
            audio_mask: Optional mask (batch, seq_len)
        
        Returns:
            Audio embeddings in LLM space (batch, num_tokens, llm_dim)
        """
        # Encode audio
        sequence_output, pooled_output = self.audio_encoder(audio_features, audio_mask)
        
        # Project to LLM space
        if self.config.projector_type == "mlp":
            # Use pooled output for simple MLP
            audio_embeds = self.audio_projector(pooled_output)
            audio_embeds = audio_embeds.unsqueeze(1)  # (batch, 1, llm_dim)
        else:
            # Use sequence output for attention-based projectors
            audio_embeds = self.audio_projector(sequence_output[:, 1:], audio_mask)
        
        return audio_embeds
    
    def prepare_inputs_for_generation(
        self,
        audio_features: torch.Tensor,
        prompt: str = "Describe this audio:",
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for LLM generation.
        
        Args:
            audio_features: Audio features (batch, seq_len, 128)
            prompt: Text prompt
            audio_mask: Optional audio mask
        
        Returns:
            Dictionary with inputs for LLM
        """
        batch_size = audio_features.size(0)
        device = audio_features.device
        
        # Encode audio
        audio_embeds = self.encode_audio(audio_features, audio_mask)
        num_audio_tokens = audio_embeds.size(1)
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(prompt_tokens['input_ids'])
        
        # Expand for batch
        if text_embeds.size(0) == 1 and batch_size > 1:
            text_embeds = text_embeds.expand(batch_size, -1, -1)
            prompt_tokens['attention_mask'] = prompt_tokens['attention_mask'].expand(batch_size, -1)
        
        # Concatenate audio and text embeddings
        # Format: [AUDIO_EMBEDS] [TEXT_PROMPT]
        input_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
        
        # Create attention mask
        audio_attn_mask = torch.ones(
            batch_size, num_audio_tokens,
            device=device, dtype=prompt_tokens['attention_mask'].dtype
        )
        attention_mask = torch.cat([audio_attn_mask, prompt_tokens['attention_mask']], dim=1)
        
        return {
            'inputs_embeds': input_embeds,
            'attention_mask': attention_mask,
            'audio_embeds': audio_embeds,
            'num_audio_tokens': num_audio_tokens,
        }
    
    def forward(
        self,
        audio_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            audio_features: Audio features (batch, seq_len, 128)
            input_ids: Text token IDs (batch, text_len)
            attention_mask: Attention mask (batch, text_len)
            labels: Target token IDs for loss computation
            audio_mask: Optional audio mask
        
        Returns:
            Dictionary with loss and logits
        """
        if self.llm is None:
            raise RuntimeError("LLM not set. Call set_llm() first.")
        
        batch_size = audio_features.size(0)
        device = audio_features.device
        
        # Encode audio
        audio_embeds = self.encode_audio(audio_features, audio_mask)
        num_audio_tokens = audio_embeds.size(1)
        
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Concatenate embeddings
        input_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
        
        # Extend attention mask for audio tokens
        audio_attn_mask = torch.ones(
            batch_size, num_audio_tokens,
            device=device, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([audio_attn_mask, attention_mask], dim=1)
        
        # Extend labels for audio tokens (use -100 to ignore in loss)
        if labels is not None:
            audio_labels = torch.full(
                (batch_size, num_audio_tokens),
                -100,
                device=device,
                dtype=labels.dtype
            )
            full_labels = torch.cat([audio_labels, labels], dim=1)
        else:
            full_labels = None
        
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            **kwargs,
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        audio_features: torch.Tensor,
        prompt: str = "Describe this audio:",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Generate captions for audio.
        
        Args:
            audio_features: Audio features (batch, seq_len, 128)
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            List of generated captions
        """
        if self.llm is None:
            raise RuntimeError("LLM not set. Call set_llm() first.")
        
        # Prepare inputs
        inputs = self.prepare_inputs_for_generation(audio_features, prompt)
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs['inputs_embeds'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        
        # Decode
        # Skip audio tokens in output
        num_audio_tokens = inputs['num_audio_tokens']
        captions = []
        for output in outputs:
            # The output includes the audio token positions
            text = self.tokenizer.decode(output[num_audio_tokens:], skip_special_tokens=True)
            captions.append(text)
        
        return captions
    
    def save_pretrained(self, save_path: str):
        """Save model components."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save audio encoder
        torch.save(
            self.audio_encoder.state_dict(),
            os.path.join(save_path, "audio_encoder.pt")
        )
        
        # Save projector
        torch.save(
            self.audio_projector.state_dict(),
            os.path.join(save_path, "audio_projector.pt")
        )
        
        # Save config
        import json
        config_dict = {
            k: v for k, v in vars(self.config).items()
            if not k.startswith('_')
        }
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        llm_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> "MedGemmaAudio":
        """Load model from pretrained checkpoint."""
        import os
        import json
        
        # Load config
        with open(os.path.join(load_path, "config.json"), 'r') as f:
            config_dict = json.load(f)
        config = MedGemmaAudioConfig(**config_dict)
        
        # Create model
        model = cls(config, llm_model=llm_model, tokenizer=tokenizer)
        
        # Load audio encoder
        model.audio_encoder.load_state_dict(
            torch.load(os.path.join(load_path, "audio_encoder.pt"))
        )
        
        # Load projector
        model.audio_projector.load_state_dict(
            torch.load(os.path.join(load_path, "audio_projector.pt"))
        )
        
        return model
