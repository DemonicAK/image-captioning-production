"""Encoder components for caption models.

This module provides image and text encoding layers
for the caption model architecture.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf


class ImageProjection(tf.keras.layers.Layer):
    """Image feature projection layer.
    
    Projects CNN features to a common embedding space
    with optional dropout and normalization.
    
    Attributes:
        projection_dim: Output dimension.
        dropout_rate: Dropout rate.
        use_layer_norm: Whether to apply layer normalization.
    
    Example:
        >>> proj = ImageProjection(projection_dim=256, dropout_rate=0.3)
        >>> projected = proj(image_features)  # (batch, 1536) -> (batch, 256)
    """
    
    def __init__(
        self,
        projection_dim: int = 256,
        dropout_rate: float = 0.3,
        use_layer_norm: bool = True,
        **kwargs,
    ) -> None:
        """Initialize image projection layer.
        
        Args:
            projection_dim: Output dimension.
            dropout_rate: Dropout rate for regularization.
            use_layer_norm: Whether to apply layer normalization.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self._projection_dim = projection_dim
        self._dropout_rate = dropout_rate
        self._use_layer_norm = use_layer_norm
        
        self._dense = tf.keras.layers.Dense(
            projection_dim,
            activation="relu",
            kernel_initializer="he_normal",
        )
        self._dropout = tf.keras.layers.Dropout(dropout_rate)
        self._layer_norm = (
            tf.keras.layers.LayerNormalization()
            if use_layer_norm
            else None
        )
    
    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """Project image features.
        
        Args:
            inputs: Image features of shape (batch, feature_dim).
            training: Whether in training mode.
            
        Returns:
            Projected features of shape (batch, projection_dim).
        """
        x = self._dense(inputs)
        x = self._dropout(x, training=training)
        if self._layer_norm is not None:
            x = self._layer_norm(x)
        return x
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "projection_dim": self._projection_dim,
            "dropout_rate": self._dropout_rate,
            "use_layer_norm": self._use_layer_norm,
        })
        return config


class TextEmbedding(tf.keras.layers.Layer):
    """Text embedding layer with optional pre-trained embeddings.
    
    Embeds word indices to dense vectors, optionally using
    pre-trained GloVe embeddings.
    
    Attributes:
        vocab_size: Size of vocabulary.
        embedding_dim: Embedding dimension.
        mask_zero: Whether to mask zero (padding) tokens.
    
    Example:
        >>> emb = TextEmbedding(vocab_size=5000, embedding_dim=200)
        >>> embedded = emb(word_indices)  # (batch, seq) -> (batch, seq, 200)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        embedding_matrix: Optional[np.ndarray] = None,
        trainable_embeddings: bool = False,
        mask_zero: bool = True,
        **kwargs,
    ) -> None:
        """Initialize text embedding layer.
        
        Args:
            vocab_size: Size of vocabulary.
            embedding_dim: Embedding dimension.
            embedding_matrix: Pre-trained embedding matrix (optional).
            trainable_embeddings: Whether embeddings are trainable.
            mask_zero: Whether to mask padding tokens.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._mask_zero = mask_zero
        self._trainable_embeddings = trainable_embeddings
        
        weights = [embedding_matrix] if embedding_matrix is not None else None
        
        self._embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            weights=weights,
            trainable=trainable_embeddings,
            mask_zero=mask_zero,
        )
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Embed word indices.
        
        Args:
            inputs: Word indices of shape (batch, seq_len).
            
        Returns:
            Embeddings of shape (batch, seq_len, embedding_dim).
        """
        return self._embedding(inputs)
    
    def compute_mask(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> Optional[tf.Tensor]:
        """Compute mask for padded sequences."""
        if self._mask_zero:
            return self._embedding.compute_mask(inputs)
        return None
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "vocab_size": self._vocab_size,
            "embedding_dim": self._embedding_dim,
            "mask_zero": self._mask_zero,
            "trainable_embeddings": self._trainable_embeddings,
        })
        return config


class LSTMEncoder(tf.keras.layers.Layer):
    """LSTM-based sequence encoder.
    
    Encodes text sequences using LSTM with optional
    bidirectional processing and dropout.
    
    Attributes:
        units: Number of LSTM units.
        return_sequences: Whether to return all timesteps.
        bidirectional: Whether to use bidirectional LSTM.
    """
    
    def __init__(
        self,
        units: int = 256,
        return_sequences: bool = True,
        bidirectional: bool = False,
        dropout: float = 0.3,
        recurrent_dropout: float = 0.2,
        **kwargs,
    ) -> None:
        """Initialize LSTM encoder.
        
        Args:
            units: Number of LSTM units.
            return_sequences: Whether to return all timesteps.
            bidirectional: Whether to use bidirectional LSTM.
            dropout: Input dropout rate.
            recurrent_dropout: Recurrent dropout rate.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self._units = units
        self._return_sequences = return_sequences
        self._bidirectional = bidirectional
        
        lstm = tf.keras.layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
        
        if bidirectional:
            self._lstm = tf.keras.layers.Bidirectional(lstm)
        else:
            self._lstm = lstm
    
    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """Encode sequence.
        
        Args:
            inputs: Input tensor of shape (batch, seq, dim).
            mask: Optional mask tensor.
            training: Whether in training mode.
            
        Returns:
            Encoded sequence.
        """
        return self._lstm(inputs, mask=mask, training=training)
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "units": self._units,
            "return_sequences": self._return_sequences,
            "bidirectional": self._bidirectional,
        })
        return config
