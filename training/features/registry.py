"""Feature extractor registry.

This module implements the registry pattern for feature extractors,
allowing dynamic registration and lookup of extractors by name.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple, Type

from training.features.base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureExtractorRegistry:
    """Registry for feature extractor classes.
    
    Implements the registry pattern for dynamic registration
    and retrieval of feature extractors.
    
    Example:
        >>> registry = FeatureExtractorRegistry()
        >>> registry.register("efficientnet", EfficientNetExtractor)
        >>> extractor = registry.get("efficientnet", image_size=(300, 300))
    """
    
    _instance: Optional["FeatureExtractorRegistry"] = None
    
    def __new__(cls) -> "FeatureExtractorRegistry":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._extractors = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the registry."""
        if not self._initialized:
            self._extractors: Dict[str, Type[BaseFeatureExtractor]] = {}
            self._initialized = True
            self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default feature extractors."""
        from training.features.efficientnet import (
            EfficientNetExtractor,
            InceptionV3Extractor,
        )
        
        self.register("EfficientNetB3", EfficientNetExtractor)
        self.register("efficientnetb3", EfficientNetExtractor)
        self.register("InceptionV3", InceptionV3Extractor)
        self.register("inceptionv3", InceptionV3Extractor)
    
    def register(
        self,
        name: str,
        extractor_class: Type[BaseFeatureExtractor],
    ) -> None:
        """Register a feature extractor class.
        
        Args:
            name: Name to register the extractor under.
            extractor_class: Feature extractor class to register.
            
        Raises:
            TypeError: If extractor_class is not a BaseFeatureExtractor subclass.
        """
        if not issubclass(extractor_class, BaseFeatureExtractor):
            raise TypeError(
                f"extractor_class must be a subclass of BaseFeatureExtractor, "
                f"got {type(extractor_class)}"
            )
        
        self._extractors[name] = extractor_class
        logger.debug(f"Registered feature extractor: {name}")
    
    def get(
        self,
        name: str,
        **kwargs,
    ) -> BaseFeatureExtractor:
        """Get a feature extractor instance by name.
        
        Args:
            name: Name of the registered extractor.
            **kwargs: Arguments to pass to the extractor constructor.
            
        Returns:
            Instantiated feature extractor.
            
        Raises:
            KeyError: If extractor name is not registered.
        """
        if name not in self._extractors:
            available = ", ".join(self._extractors.keys())
            raise KeyError(
                f"Unknown feature extractor: {name}. "
                f"Available: {available}"
            )
        
        extractor_class = self._extractors[name]
        return extractor_class(**kwargs)
    
    def list_available(self) -> list[str]:
        """List all registered extractor names.
        
        Returns:
            List of registered extractor names.
        """
        return list(self._extractors.keys())
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None


# Convenience function for getting extractors
def get_feature_extractor(
    name: str,
    image_size: Tuple[int, int] = (300, 300),
    batch_size: int = 32,
    weights: str = "imagenet",
    build: bool = True,
) -> BaseFeatureExtractor:
    """Factory function to get a feature extractor.
    
    Args:
        name: Name of the feature extractor.
        image_size: Input image size.
        batch_size: Batch size for extraction.
        weights: Pre-trained weights to use.
        build: Whether to call build_model() automatically.
        
    Returns:
        Configured feature extractor instance.
    
    Example:
        >>> extractor = get_feature_extractor("EfficientNetB3", image_size=(300, 300))
        >>> features = extractor.extract_features(keys, images_path)
    """
    registry = FeatureExtractorRegistry()
    extractor = registry.get(
        name,
        image_size=image_size,
        batch_size=batch_size,
        weights=weights,
    )
    
    if build:
        extractor.build_model()
    
    return extractor
