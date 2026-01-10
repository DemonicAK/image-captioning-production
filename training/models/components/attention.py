"""Attention mechanisms for caption models.

This module provides attention layer implementations
for cross-modal attention between image and text features.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf


class MultiHeadCrossAttention(tf.keras.layers.Layer):
    """Multi-head cross attention for image-text fusion.
    
    Implements cross-attention where image features attend
    to text sequence representations.
    
    Attributes:
        num_heads: Number of attention heads.
        key_dim: Dimension of key/query vectors.
        dropout_rate: Dropout rate for attention weights.
    
    Example:
        >>> attention = MultiHeadCrossAttention(num_heads=4, key_dim=256)
        >>> context = attention(image_features, text_sequence)
    """
    
    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int = 256,
        dropout_rate: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize multi-head cross attention.
        
        Args:
            num_heads: Number of attention heads.
            key_dim: Dimension of key/query vectors.
            dropout_rate: Dropout rate for attention weights.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._dropout_rate = dropout_rate
        
        self._attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
        )
        self._pooling = tf.keras.layers.GlobalAveragePooling1D()
    
    def call(
        self,
        query: tf.Tensor,
        key_value: tf.Tensor,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """Apply cross attention.
        
        Args:
            query: Query tensor (image features) of shape (batch, dim).
            key_value: Key/value tensor (text sequence) of shape (batch, seq, dim).
            training: Whether in training mode.
            
        Returns:
            Context vector of shape (batch, dim).
        """
        # Reshape query for attention: (batch, dim) -> (batch, 1, dim)
        query_expanded = tf.expand_dims(query, axis=1)
        
        # Apply multi-head attention
        attended = self._attention(
            query_expanded,
            key_value,
            training=training,
        )
        
        # Pool to get fixed-size context vector
        context = self._pooling(attended)
        
        return context
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "dropout_rate": self._dropout_rate,
        })
        return config


class BahdanauAttention(tf.keras.layers.Layer):
    """Bahdanau-style additive attention.
    
    Classic attention mechanism for sequence-to-sequence models.
    Computes attention weights using a learned alignment model.
    
    Attributes:
        units: Number of units in the attention alignment model.
    """
    
    def __init__(self, units: int = 256, **kwargs) -> None:
        """Initialize Bahdanau attention.
        
        Args:
            units: Number of units for the attention layers.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self._units = units
        
        self._W1 = tf.keras.layers.Dense(units)
        self._W2 = tf.keras.layers.Dense(units)
        self._V = tf.keras.layers.Dense(1)
    
    def call(
        self,
        query: tf.Tensor,
        values: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute attention context and weights.
        
        Args:
            query: Query tensor of shape (batch, query_dim).
            values: Value tensor of shape (batch, seq_len, value_dim).
            
        Returns:
            Tuple of (context_vector, attention_weights).
        """
        # Expand query: (batch, query_dim) -> (batch, 1, query_dim)
        query_expanded = tf.expand_dims(query, 1)
        
        # Score calculation
        score = self._V(
            tf.nn.tanh(self._W1(query_expanded) + self._W2(values))
        )
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context = attention_weights * values
        context = tf.reduce_sum(context, axis=1)
        
        return context, tf.squeeze(attention_weights, -1)
    
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({"units": self._units})
        return config
