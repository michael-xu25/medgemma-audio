"""
Group Relative Policy Optimization (GRPO) training for MedGemma Audio.

Based on: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

from src.models.medgemma_audio import MedGemmaAudio, MedGemmaAudioConfig
from src.data.dataset import AudioCapsDataset
from src.utils.logging import TrainingLogger
from src.utils.metrics import compute_batch_rewards


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Model
    model_path: str = "checkpoints/sft/best_model"  # Start from SFT checkpoint
    model_name: str = "google/medgemma-4b-it"
    
    # GRPO
    num_generations: int = 4  # Number of generations per prompt
    max_new_tokens: int = 128
    temperature: float = 0.8
    reward_metric: str = "cider"  # 'cider', 'bleu', 'rouge'
    kl_coef: float = 0.1  # KL penalty coefficient
    clip_range: float = 0.2  # PPO clip range
    
    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 1
    num_steps: int = 500
    learning_rate: float = 5e-6
    warmup_steps: int = 50
    max_seq_length: int = 512
    
    # Data
    data_path: str = "data/processed"
    
    # Output
    output_dir: str = "checkpoints/grpo"
    
    # Misc
    seed: int = 42
    use_4bit: bool = True


class GRPOTrainer:
    """
    GRPO Trainer for audio captioning.
    
    GRPO (Group Relative Policy Optimization) is an RL method that:
    1. Generates multiple responses for each prompt
    2. Computes rewards using automatic metrics (BLEU, CIDEr, etc.)
    3. Uses relative rewards within the group to update policy
    """
    
    def __init__(
        self,
        model: MedGemmaAudio,
        ref_model: MedGemmaAudio,
        tokenizer,
        config: GRPOConfig,
        logger: TrainingLogger,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def generate_responses(
        self,
        audio_features: torch.Tensor,
        prompt: str = "Describe this audio in detail:",
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate multiple responses for each audio sample.
        
        Returns:
            Tuple of (responses, log_probs, response_ids)
        """
        batch_size = audio_features.size(0)
        all_responses = []
        all_log_probs = []
        all_response_ids = []
        
        for _ in range(self.config.num_generations):
            with torch.no_grad():
                # Generate response
                responses = self.model.generate(
                    audio_features,
                    prompt=prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                )
                all_responses.extend(responses)
            
            # Get log probabilities for the generated responses
            # Tokenize responses
            response_tokens = self.tokenizer(
                responses,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            
            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model(
                    audio_features=audio_features,
                    input_ids=response_tokens['input_ids'],
                    attention_mask=response_tokens['attention_mask'],
                )
            
            # Compute log probabilities
            logits = outputs.logits[:, :-1, :]  # Shift for next token prediction
            target_ids = response_tokens['input_ids'][:, 1:]  # Shift targets
            
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = torch.gather(
                log_probs, 2, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask padding
            mask = (target_ids != self.tokenizer.pad_token_id).float()
            seq_log_probs = (selected_log_probs * mask).sum(dim=1) / mask.sum(dim=1)
            
            all_log_probs.append(seq_log_probs)
            all_response_ids.append(response_tokens['input_ids'])
        
        # Reshape: (num_generations * batch_size) -> (batch_size, num_generations)
        log_probs = torch.stack(all_log_probs, dim=1)  # (batch, num_gen)
        
        return all_responses, log_probs, all_response_ids
    
    def compute_rewards(
        self,
        responses: List[str],
        references: List[List[str]],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses.
        
        Args:
            responses: Generated responses (batch_size * num_generations)
            references: Reference captions per sample
            batch_size: Original batch size
        
        Returns:
            Rewards tensor of shape (batch_size, num_generations)
        """
        num_gen = self.config.num_generations
        rewards = []
        
        for i in range(batch_size):
            sample_rewards = []
            refs = references[i]
            
            for j in range(num_gen):
                response = responses[i * num_gen + j]
                reward = compute_batch_rewards(
                    [response], [refs], metric=self.config.reward_metric
                )[0]
                sample_rewards.append(reward)
            
            rewards.append(sample_rewards)
        
        return torch.tensor(rewards, device=self.device)
    
    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GRPO loss.
        
        GRPO uses group-relative rewards:
        - Normalize rewards within each group (per sample)
        - Use advantage = reward - mean(rewards)
        
        Args:
            log_probs: Log probs from policy (batch, num_gen)
            ref_log_probs: Log probs from reference (batch, num_gen)
            rewards: Reward scores (batch, num_gen)
        
        Returns:
            GRPO loss
        """
        # Normalize rewards within each group
        rewards_mean = rewards.mean(dim=1, keepdim=True)
        rewards_std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - rewards_mean) / rewards_std
        
        # KL divergence penalty
        kl_div = log_probs - ref_log_probs
        
        # Policy ratio
        ratio = torch.exp(log_probs - ref_log_probs)
        
        # Clipped objective
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.config.clip_range,
            1 + self.config.clip_range
        )
        
        # GRPO objective
        obj1 = ratio * advantages
        obj2 = clipped_ratio * advantages
        policy_loss = -torch.min(obj1, obj2).mean()
        
        # KL penalty
        kl_loss = self.config.kl_coef * kl_div.mean()
        
        total_loss = policy_loss + kl_loss
        
        return total_loss, {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages.mean().item(),
        }
    
    def train_step(
        self,
        batch: Dict,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Single training step."""
        audio_features = batch['audio_features'].to(self.device)
        captions = batch['captions']  # List of caption lists (references)
        
        batch_size = audio_features.size(0)
        
        # Generate responses from policy
        responses, log_probs, _ = self.generate_responses(audio_features)
        
        # Get log probs from reference model
        with torch.no_grad():
            _, ref_log_probs, _ = self.ref_model_generate(audio_features, responses)
        
        # Compute rewards
        rewards = self.compute_rewards(responses, captions, batch_size)
        
        # Compute loss
        loss, metrics = self.compute_grpo_loss(log_probs, ref_log_probs, rewards)
        
        # Backward pass
        loss.backward()
        
        return metrics
    
    def ref_model_generate(
        self,
        audio_features: torch.Tensor,
        responses: List[str],
    ) -> Tuple[List[str], torch.Tensor, List[torch.Tensor]]:
        """Get log probs for given responses from reference model."""
        batch_size = audio_features.size(0)
        num_gen = self.config.num_generations
        
        all_log_probs = []
        
        for i in range(num_gen):
            batch_responses = responses[i * batch_size:(i + 1) * batch_size]
            
            response_tokens = self.tokenizer(
                batch_responses,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            
            outputs = self.ref_model(
                audio_features=audio_features,
                input_ids=response_tokens['input_ids'],
                attention_mask=response_tokens['attention_mask'],
            )
            
            logits = outputs.logits[:, :-1, :]
            target_ids = response_tokens['input_ids'][:, 1:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = torch.gather(
                log_probs, 2, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            mask = (target_ids != self.tokenizer.pad_token_id).float()
            seq_log_probs = (selected_log_probs * mask).sum(dim=1) / mask.sum(dim=1)
            
            all_log_probs.append(seq_log_probs)
        
        log_probs = torch.stack(all_log_probs, dim=1)
        return responses, log_probs, []


def load_models_for_grpo(config: GRPOConfig, logger):
    """Load policy and reference models."""
    from unsloth import FastLanguageModel
    
    logger.info(f"Loading models from {config.model_path}...")
    
    # Load base LLM
    llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.use_4bit,
        token=os.environ.get("HF_TOKEN"),
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create policy model
    policy_config = MedGemmaAudioConfig(llm_model_name=config.model_name)
    policy_model = MedGemmaAudio.from_pretrained(
        config.model_path,
        llm_model=llm,
        tokenizer=tokenizer,
    )
    
    # Create reference model (copy of policy, frozen)
    ref_llm, _ = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.use_4bit,
        token=os.environ.get("HF_TOKEN"),
    )
    
    ref_model = MedGemmaAudio.from_pretrained(
        config.model_path,
        llm_model=ref_llm,
        tokenizer=tokenizer,
    )
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    
    return policy_model, ref_model, tokenizer


def main(args):
    """Main GRPO training function."""
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = GRPOConfig(**config_dict)
    else:
        config = GRPOConfig()
    
    # Override with command line args
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Setup logging
    logger = TrainingLogger(
        log_dir="logs",
        project="medgemma-audio-grpo",
        run_name=args.run_name,
        config=vars(config),
        use_wandb=args.use_wandb,
    )
    
    logger.info("Starting GRPO training...")
    logger.info(f"Config: {vars(config)}")
    
    # Load models
    policy_model, ref_model, tokenizer = load_models_for_grpo(config, logger)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model = policy_model.to(device)
    ref_model = ref_model.to(device)
    
    # Create dataset
    logger.info("Loading dataset...")
    from torch.utils.data import DataLoader
    
    train_dataset = AudioCapsDataset(
        data_path=config.data_path,
        split="train",
        target_length=10,
    )
    
    def collate_fn(batch):
        audio_features = torch.stack([item['audio_features'] for item in batch])
        captions = [[item['caption']] for item in batch]  # List of reference lists
        return {
            'audio_features': audio_features,
            'captions': captions,
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        logger=logger,
    )
    
    # Training loop
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_reward = 0.0
    
    logger.info(f"Starting training for {config.num_steps} steps...")
    
    pbar = tqdm(total=config.num_steps, desc="GRPO Training")
    
    while global_step < config.num_steps:
        for batch in train_loader:
            if global_step >= config.num_steps:
                break
            
            # Training step
            metrics = trainer.train_step(batch, optimizer)
            
            # Gradient accumulation
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy_model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
            
            # Log metrics
            if global_step % 10 == 0:
                logger.log_metrics({
                    'grpo/policy_loss': metrics['policy_loss'],
                    'grpo/kl_loss': metrics['kl_loss'],
                    'grpo/mean_reward': metrics['mean_reward'],
                    'grpo/mean_advantage': metrics['mean_advantage'],
                }, step=global_step)
            
            # Save best model
            if metrics['mean_reward'] > best_reward:
                best_reward = metrics['mean_reward']
                policy_model.save_pretrained(str(output_dir / "best_model"))
                logger.info(f"Saved best model with reward: {best_reward:.4f}")
            
            global_step += 1
            pbar.update(1)
            pbar.set_postfix({
                'reward': f"{metrics['mean_reward']:.4f}",
                'loss': f"{metrics['policy_loss']:.4f}",
            })
    
    pbar.close()
    
    # Save final model
    policy_model.save_pretrained(str(output_dir / "final_model"))
    logger.info(f"GRPO training complete! Best reward: {best_reward:.4f}")
    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training for MedGemma Audio")
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model-path", type=str, help="Path to SFT checkpoint")
    parser.add_argument("--model-name", type=str, help="Base model name")
    parser.add_argument("--data-path", type=str, help="Path to processed data")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--run-name", type=str, help="Run name for logging")
    
    # GRPO args
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--reward-metric", type=str, choices=['cider', 'bleu', 'rouge'])
    parser.add_argument("--kl-coef", type=float)
    parser.add_argument("--clip-range", type=float)
    
    # Training args
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    
    # Misc
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
