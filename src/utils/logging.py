"""
Logging utilities for training.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(
    log_dir: str = "logs",
    log_name: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_name: Name for the log file (defaults to timestamp)
        level: Logging level
        console: Whether to also log to console
    
    Returns:
        Configured logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"train_{timestamp}"
    
    log_file = log_path / f"{log_name}.log"
    
    # Configure logging
    logger = logging.getLogger("medgemma_audio")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logging to {log_file}")
    return logger


def get_logger(name: str = "medgemma_audio") -> logging.Logger:
    """Get existing logger or create new one."""
    return logging.getLogger(name)


class WandbLogger:
    """Weights & Biases logger wrapper."""
    
    def __init__(
        self,
        project: str = "medgemma-audio",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            name: Run name
            config: Configuration to log
            enabled: Whether to enable W&B logging
        """
        self.enabled = enabled and WANDB_AVAILABLE
        
        if self.enabled:
            wandb.init(
                project=project,
                name=name,
                config=config,
            )
            self.run = wandb.run
        else:
            self.run = None
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def log_table(self, name: str, data: Any, columns: Optional[list] = None):
        """Log a table."""
        if self.enabled:
            table = wandb.Table(data=data, columns=columns)
            wandb.log({name: table})
    
    def log_artifact(self, name: str, path: str, type: str = "model"):
        """Log an artifact."""
        if self.enabled:
            artifact = wandb.Artifact(name, type=type)
            artifact.add_dir(path) if os.path.isdir(path) else artifact.add_file(path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish the run."""
        if self.enabled:
            wandb.finish()


class TrainingLogger:
    """Combined logger for training with file, console, and W&B support."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        project: str = "medgemma-audio",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for log files
            project: W&B project name
            run_name: Name for this run
            config: Training configuration
            use_wandb: Whether to use W&B
        """
        self.logger = setup_logging(log_dir, run_name)
        self.wandb = WandbLogger(
            project=project,
            name=run_name,
            config=config,
            enabled=use_wandb,
        )
        self.step = 0
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics."""
        step = step or self.step
        
        # Log to file
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metrics_str}")
        
        # Log to W&B
        self.wandb.log(metrics, step=step)
        
        self.step = step + 1
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.logger.info(f"Hyperparameters: {params}")
        if self.wandb.enabled:
            wandb.config.update(params)
    
    def save_model(self, path: str, name: str = "model"):
        """Log saved model as artifact."""
        self.logger.info(f"Saved model to {path}")
        self.wandb.log_artifact(name, path, type="model")
    
    def finish(self):
        """Finish logging."""
        self.wandb.finish()
