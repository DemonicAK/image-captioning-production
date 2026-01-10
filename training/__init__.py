"""Image Captioning Training Package.

This package provides a production-grade training pipeline for
image captioning models with attention mechanisms.

Modules:
    configs: Configuration management with dataclasses.
    data: Data loading, preprocessing, and tokenization.
    features: Image feature extraction with registry pattern.
    models: Model architecture components.
    trainers: Training loop abstractions.
    evaluation: Metrics and inference decoders.
    callbacks: Custom training callbacks.
    utils: Utility functions.

Example:
    >>> from training.train import TrainingPipeline
    >>> pipeline = TrainingPipeline("config.yaml")
    >>> pipeline.run()
"""

__version__ = "1.0.0"
__author__ = "Image Captioning Team"

from training.configs import Config, load_config
from training.data import CaptionLoader, Tokenizer, DatasetBuilder
from training.features import get_feature_extractor
from training.models import build_caption_model
from training.trainers import Trainer

__all__ = [
    "Config",
    "load_config",
    "CaptionLoader",
    "Tokenizer",
    "DatasetBuilder",
    "get_feature_extractor",
    "build_caption_model",
    "Trainer",
]
