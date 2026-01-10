"""Dataclass-based configuration management.

This module provides structured configuration using Python dataclasses
with validation and YAML serialization support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data paths and preprocessing.
    
    Attributes:
        images_path: Path to the directory containing images.
        captions_file: Path to the captions text file.
        glove_path: Path to GloVe embeddings file.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        word_count_threshold: Minimum word frequency for vocabulary.
        random_seed: Seed for reproducible data splits.
    """
    images_path: str
    captions_file: str
    glove_path: str
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    word_count_threshold: int = 5
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")
        if not 0 <= self.val_ratio < 1:
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")
        if not 0 <= self.test_ratio < 1:
            raise ValueError(f"test_ratio must be in [0, 1), got {self.test_ratio}")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model architecture.
    
    Attributes:
        feature_extractor: Name of the feature extraction backbone.
        feature_dim: Dimension of image features.
        embedding_dim: Dimension of word embeddings.
        hidden_dim: Hidden dimension for LSTM and dense layers.
        num_attention_heads: Number of attention heads.
        dropout_rate: Dropout rate for regularization.
        recurrent_dropout: Dropout rate for recurrent connections.
        image_size: Input image size as (height, width).
    """
    feature_extractor: str = "EfficientNetB3"
    feature_dim: int = 1536
    embedding_dim: int = 200
    hidden_dim: int = 256
    num_attention_heads: int = 4
    dropout_rate: float = 0.3
    recurrent_dropout: float = 0.2
    image_size: Tuple[int, int] = (300, 300)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training hyperparameters.
    
    Attributes:
        batch_size: Number of samples per batch.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate.
        lr_decay_factor: Factor to reduce LR on plateau.
        lr_patience: Epochs to wait before reducing LR.
        min_lr: Minimum learning rate.
        early_stopping_patience: Epochs to wait before early stopping.
        use_mixed_precision: Whether to use mixed precision training.
        artifacts_dir: Directory to save model artifacts.
    """
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-4
    lr_decay_factor: float = 0.5
    lr_patience: int = 3
    min_lr: float = 1e-6
    early_stopping_patience: int = 5
    use_mixed_precision: bool = True
    artifacts_dir: str = "../shared/artifacts"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")


@dataclass
class Config:
    """Root configuration container.
    
    Aggregates all sub-configurations for the training pipeline.
    
    Attributes:
        data: Data configuration.
        model: Model architecture configuration.
        training: Training hyperparameters.
    """
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> Config:
        """Create Config from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values.
            
        Returns:
            Config instance.
        """
        data_config = DataConfig(
            images_path=config_dict.get("images_path", ""),
            captions_file=config_dict.get("captions_file", ""),
            glove_path=config_dict.get("glove_path", ""),
            train_ratio=config_dict.get("train_ratio", 0.6),
            val_ratio=config_dict.get("val_ratio", 0.2),
            test_ratio=config_dict.get("test_ratio", 0.2),
            word_count_threshold=config_dict.get("word_count_threshold", 5),
            random_seed=config_dict.get("random_seed", 42),
        )
        
        model_config = ModelConfig(
            feature_extractor=config_dict.get("feature_extractor", "EfficientNetB3"),
            feature_dim=config_dict.get("feature_dim", 1536),
            embedding_dim=config_dict.get("embedding_dim", 200),
            hidden_dim=config_dict.get("hidden_dim", 256),
            num_attention_heads=config_dict.get("num_attention_heads", 4),
            dropout_rate=config_dict.get("dropout_rate", 0.3),
            recurrent_dropout=config_dict.get("recurrent_dropout", 0.2),
            image_size=tuple(config_dict.get("image_size", [300, 300])),
        )
        
        training_config = TrainingConfig(
            batch_size=config_dict.get("batch_size", 64),
            epochs=config_dict.get("epochs", 20),
            learning_rate=config_dict.get("learning_rate", 1e-4),
            lr_decay_factor=config_dict.get("lr_decay_factor", 0.5),
            lr_patience=config_dict.get("lr_patience", 3),
            min_lr=config_dict.get("min_lr", 1e-6),
            early_stopping_patience=config_dict.get("early_stopping_patience", 5),
            use_mixed_precision=config_dict.get("use_mixed_precision", True),
            artifacts_dir=config_dict.get("artifacts_dir", "../shared/artifacts"),
        )
        
        return cls(data=data_config, model=model_config, training=training_config)


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Config instance.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)
