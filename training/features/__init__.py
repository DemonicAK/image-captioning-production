"""Feature extraction module."""

from training.features.base import BaseFeatureExtractor
from training.features.efficientnet import EfficientNetExtractor
from training.features.registry import FeatureExtractorRegistry, get_feature_extractor

__all__ = [
    "BaseFeatureExtractor",
    "EfficientNetExtractor",
    "FeatureExtractorRegistry",
    "get_feature_extractor",
]
