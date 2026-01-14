"""
Evaluation metrics for audio captioning.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False


def compute_bleu_scores(
    predictions: List[str],
    references: List[List[str]],
    weights: tuple = (0.25, 0.25, 0.25, 0.25),
) -> Dict[str, float]:
    """
    Compute BLEU scores using NLTK.
    
    Args:
        predictions: List of predicted captions
        references: List of reference caption lists (each can have multiple refs)
        weights: Weights for n-gram precision
    
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    if not NLTK_AVAILABLE:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
    
    # Tokenize
    pred_tokens = [pred.lower().split() for pred in predictions]
    ref_tokens = [[ref.lower().split() for ref in refs] for refs in references]
    
    smoother = SmoothingFunction()
    
    # Compute individual BLEU scores
    scores = {}
    for n in range(1, 5):
        w = tuple([1.0 / n] * n + [0.0] * (4 - n))
        bleu_n = corpus_bleu(ref_tokens, pred_tokens, weights=w, smoothing_function=smoother.method1)
        scores[f"bleu_{n}"] = bleu_n
    
    return scores


def compute_caption_metrics(
    predictions: List[str],
    references: List[List[str]],
    use_coco_eval: bool = True,
) -> Dict[str, float]:
    """
    Compute comprehensive caption evaluation metrics.
    
    Args:
        predictions: List of predicted captions
        references: List of reference caption lists
        use_coco_eval: Whether to use pycocoevalcap metrics
    
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Basic BLEU scores
    bleu_scores = compute_bleu_scores(predictions, references)
    metrics.update(bleu_scores)
    
    # COCO evaluation metrics
    if use_coco_eval and COCO_EVAL_AVAILABLE:
        # Format for pycocoevalcap
        gts = {}
        res = {}
        for i, (pred, refs) in enumerate(zip(predictions, references)):
            gts[i] = [{"caption": ref} for ref in refs]
            res[i] = [{"caption": pred}]
        
        # Compute metrics
        scorers = [
            (Bleu(4), ["bleu_coco_1", "bleu_coco_2", "bleu_coco_3", "bleu_coco_4"]),
            (Rouge(), "rouge_l"),
            (Cider(), "cider"),
        ]
        
        # Meteor and SPICE can be slow, enable optionally
        # (Meteor(), "meteor"),
        # (Spice(), "spice"),
        
        for scorer, method in scorers:
            try:
                score, _ = scorer.compute_score(gts, res)
                if isinstance(method, list):
                    for m, s in zip(method, score):
                        metrics[m] = s
                else:
                    metrics[method] = score
            except Exception as e:
                print(f"Warning: Could not compute {method}: {e}")
    
    return metrics


def compute_mae_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute MAE reconstruction metrics.
    
    Args:
        reconstruction: Reconstructed features
        target: Original features
        mask: Binary mask (1 = masked)
    
    Returns:
        Dictionary with reconstruction metrics
    """
    with torch.no_grad():
        # MSE on masked patches
        mse = ((reconstruction - target) ** 2).mean(dim=-1)
        masked_mse = (mse * mask).sum() / mask.sum()
        
        # MSE on all patches
        full_mse = mse.mean()
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            reconstruction.view(-1, reconstruction.size(-1)),
            target.view(-1, target.size(-1)),
            dim=-1
        ).mean()
    
    return {
        "masked_mse": masked_mse.item(),
        "full_mse": full_mse.item(),
        "cosine_similarity": cos_sim.item(),
    }


class CaptionMetricsAccumulator:
    """Accumulator for computing metrics over batches."""
    
    def __init__(self):
        self.predictions = []
        self.references = []
        self.losses = []
    
    def add_batch(
        self,
        predictions: List[str],
        references: List[List[str]],
        loss: Optional[float] = None,
    ):
        """Add a batch of predictions and references."""
        self.predictions.extend(predictions)
        self.references.extend(references)
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self, use_coco_eval: bool = True) -> Dict[str, float]:
        """Compute metrics over all accumulated samples."""
        metrics = compute_caption_metrics(
            self.predictions,
            self.references,
            use_coco_eval=use_coco_eval,
        )
        
        if self.losses:
            metrics["avg_loss"] = np.mean(self.losses)
        
        metrics["num_samples"] = len(self.predictions)
        
        return metrics
    
    def reset(self):
        """Reset accumulator."""
        self.predictions = []
        self.references = []
        self.losses = []


def compute_reward_score(
    prediction: str,
    references: List[str],
    metric: str = "cider",
) -> float:
    """
    Compute reward score for a single prediction.
    
    Used for GRPO training.
    
    Args:
        prediction: Predicted caption
        references: Reference captions
        metric: Metric to use ('cider', 'bleu', 'rouge')
    
    Returns:
        Reward score
    """
    if not COCO_EVAL_AVAILABLE and metric in ["cider", "rouge"]:
        # Fallback to BLEU
        metric = "bleu"
    
    if metric == "bleu":
        if not NLTK_AVAILABLE:
            return 0.0
        smoother = SmoothingFunction()
        pred_tokens = prediction.lower().split()
        ref_tokens = [ref.lower().split() for ref in references]
        return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoother.method1)
    
    elif metric in ["cider", "rouge"]:
        gts = {0: [{"caption": ref} for ref in references]}
        res = {0: [{"caption": prediction}]}
        
        if metric == "cider":
            scorer = Cider()
        else:
            scorer = Rouge()
        
        score, _ = scorer.compute_score(gts, res)
        return score
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_batch_rewards(
    predictions: List[str],
    references: List[List[str]],
    metric: str = "cider",
) -> torch.Tensor:
    """
    Compute rewards for a batch of predictions.
    
    Args:
        predictions: List of predicted captions
        references: List of reference caption lists
        metric: Metric to use
    
    Returns:
        Tensor of reward scores
    """
    rewards = []
    for pred, refs in zip(predictions, references):
        reward = compute_reward_score(pred, refs, metric)
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)
