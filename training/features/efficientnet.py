"""EfficientNet-based feature extractor.

This module implements feature extraction using EfficientNet
pre-trained on ImageNet.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from training.features.base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class EfficientNetExtractor(BaseFeatureExtractor):
    """Feature extractor using EfficientNetB3.
    
    Extracts 1536-dimensional feature vectors from images
    using EfficientNetB3 pre-trained on ImageNet.
    
    Attributes:
        feature_dim: 1536 (EfficientNetB3 output dimension).
        name: "EfficientNetB3".
    
    Example:
        >>> extractor = EfficientNetExtractor(image_size=(300, 300))
        >>> extractor.build_model()
        >>> features = extractor.extract_features(keys, "/path/to/images/")
    """
    
    FEATURE_DIM = 1536
    NAME = "EfficientNetB3"
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (300, 300),
        batch_size: int = 32,
        weights: str = "imagenet",
    ) -> None:
        """Initialize EfficientNet extractor.
        
        Args:
            image_size: Input image size (default 300x300 for B3).
            batch_size: Batch size for extraction.
            weights: Pre-trained weights ("imagenet" or None).
        """
        super().__init__(image_size, batch_size, weights)
        self._model: tf.keras.Model | None = None
    
    @property
    def feature_dim(self) -> int:
        """Dimension of extracted features (1536 for B3)."""
        return self.FEATURE_DIM
    
    @property
    def name(self) -> str:
        """Name of the feature extractor."""
        return self.NAME
    
    def build_model(self) -> None:
        """Build EfficientNetB3 feature extraction model."""
        from tensorflow.keras.applications import EfficientNetB3
        
        self._model = EfficientNetB3(
            include_top=False,
            weights=self._weights,
            pooling="avg",
            input_shape=(*self._image_size, 3),
        )
        self._model.trainable = False
        
        logger.info(
            f"Built {self.NAME} model with "
            f"input_shape={self._image_size}, feature_dim={self.feature_dim}"
        )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for EfficientNet.
        
        Args:
            image: Input image array.
            
        Returns:
            Preprocessed image array.
        """
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input(image)
    
    def _load_and_preprocess_image(self, img_path: tf.Tensor) -> tf.Tensor:
        """TensorFlow function to load and preprocess image.
        
        Args:
            img_path: Path to image file.
            
        Returns:
            Preprocessed image tensor.
        """
        from tensorflow.keras.applications.efficientnet import preprocess_input
        
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self._image_size)
        image = preprocess_input(image)
        return image
    
    def _build_tf_dataset(self, image_paths: np.ndarray) -> tf.data.Dataset:
        """Build TensorFlow dataset for efficient batch processing.
        
        Args:
            image_paths: Array of image file paths.
            
        Returns:
            tf.data.Dataset yielding preprocessed images.
        """
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def extract_features(
        self,
        image_keys: List[str],
        images_path: str,
        verbose: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Extract features for a batch of images.
        
        Args:
            image_keys: List of image identifiers (without extension).
            images_path: Path to directory containing images.
            verbose: Verbosity level (0=silent, 1=progress).
            
        Returns:
            Dictionary mapping image_key to feature vector.
            
        Raises:
            RuntimeError: If model hasn't been built.
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        # Ensure images_path ends with separator
        if not images_path.endswith("/"):
            images_path = images_path + "/"
        
        # Build full paths
        image_paths = np.array(
            [f"{images_path}{key}.jpg" for key in image_keys],
            dtype=str,
        )
        
        # Create dataset and extract features
        dataset = self._build_tf_dataset(image_paths)
        features = self._model.predict(dataset, verbose=verbose)
        
        # Build result dictionary
        result = {
            image_keys[i]: features[i].astype(np.float32)
            for i in range(len(image_keys))
        }
        
        logger.info(f"Extracted features for {len(result)} images")
        return result


class InceptionV3Extractor(BaseFeatureExtractor):
    """Feature extractor using InceptionV3.
    
    Extracts 2048-dimensional feature vectors from images
    using InceptionV3 pre-trained on ImageNet.
    """
    
    FEATURE_DIM = 2048
    NAME = "InceptionV3"
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (299, 299),
        batch_size: int = 32,
        weights: str = "imagenet",
    ) -> None:
        """Initialize InceptionV3 extractor."""
        super().__init__(image_size, batch_size, weights)
    
    @property
    def feature_dim(self) -> int:
        """Dimension of extracted features (2048 for InceptionV3)."""
        return self.FEATURE_DIM
    
    @property
    def name(self) -> str:
        """Name of the feature extractor."""
        return self.NAME
    
    def build_model(self) -> None:
        """Build InceptionV3 feature extraction model."""
        from tensorflow.keras.applications import InceptionV3
        
        self._model = InceptionV3(
            include_top=False,
            weights=self._weights,
            pooling="avg",
            input_shape=(*self._image_size, 3),
        )
        self._model.trainable = False
        
        logger.info(
            f"Built {self.NAME} model with "
            f"input_shape={self._image_size}, feature_dim={self.feature_dim}"
        )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for InceptionV3."""
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input(image)
    
    def _load_and_preprocess_image(self, img_path: tf.Tensor) -> tf.Tensor:
        """TensorFlow function to load and preprocess image."""
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self._image_size)
        image = preprocess_input(image)
        return image
    
    def _build_tf_dataset(self, image_paths: np.ndarray) -> tf.data.Dataset:
        """Build TensorFlow dataset for efficient batch processing."""
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def extract_features(
        self,
        image_keys: List[str],
        images_path: str,
        verbose: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Extract features for a batch of images."""
        if self._model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        if not images_path.endswith("/"):
            images_path = images_path + "/"
        
        image_paths = np.array(
            [f"{images_path}{key}.jpg" for key in image_keys],
            dtype=str,
        )
        
        dataset = self._build_tf_dataset(image_paths)
        features = self._model.predict(dataset, verbose=verbose)
        
        result = {
            image_keys[i]: features[i].astype(np.float32)
            for i in range(len(image_keys))
        }
        
        logger.info(f"Extracted features for {len(result)} images")
        return result
