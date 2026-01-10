"""Inference decoders for caption generation.

This module provides decoding strategies for generating
captions from trained models.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from training.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class BaseDecoder(ABC):
    """Abstract base class for caption decoders.
    
    Defines the interface for decoding captions
    from model predictions.
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int,
    ) -> None:
        """Initialize decoder.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding.
            max_length: Maximum caption length.
        """
        self._tokenizer = tokenizer
        self._max_length = max_length
    
    @property
    def tokenizer(self) -> Tokenizer:
        """Get tokenizer."""
        return self._tokenizer
    
    @property
    def max_length(self) -> int:
        """Get maximum length."""
        return self._max_length
    
    @abstractmethod
    def decode(
        self,
        model: tf.keras.Model,
        image_features: np.ndarray,
    ) -> str:
        """Decode caption from image features.
        
        Args:
            model: Trained caption model.
            image_features: Image feature vector.
            
        Returns:
            Generated caption string.
        """
        pass


class GreedyDecoder(BaseDecoder):
    """Greedy decoding strategy.
    
    Selects the most probable word at each step.
    Fast but may not find the globally optimal caption.
    
    Example:
        >>> decoder = GreedyDecoder(tokenizer, max_length=38)
        >>> caption = decoder.decode(model, image_features)
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int,
        repetition_penalty: bool = True,
    ) -> None:
        """Initialize greedy decoder.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding.
            max_length: Maximum caption length.
            repetition_penalty: Whether to penalize repetitions.
        """
        super().__init__(tokenizer, max_length)
        self._repetition_penalty = repetition_penalty
    
    def decode(
        self,
        model: tf.keras.Model,
        image_features: np.ndarray,
    ) -> str:
        """Decode caption using greedy search.
        
        Args:
            model: Trained caption model.
            image_features: Image feature vector of shape (1, feature_dim).
            
        Returns:
            Generated caption string.
        """
        wordtoix = self._tokenizer.word_to_index
        ixtoword = self._tokenizer.index_to_word
        
        # Start with startseq token
        in_text = ["startseq"]
        
        for _ in range(self._max_length):
            # Encode current sequence
            seq = [wordtoix.get(w, 0) for w in in_text]
            seq = tf.keras.preprocessing.sequence.pad_sequences(
                [seq],
                maxlen=self._max_length,
                padding="post",
            )
            
            # Predict next word
            preds = model.predict(
                [image_features, seq],
                verbose=0,
            )[0]
            
            # Get most probable word
            next_idx = int(np.argmax(preds))
            next_word = ixtoword.get(next_idx)
            
            # Stop conditions
            if next_word is None:
                break
            if next_word == "endseq":
                break
            
            # Repetition guard
            if self._repetition_penalty and next_word in in_text[-2:]:
                break
            
            in_text.append(next_word)
        
        # Remove startseq and join
        caption = " ".join(in_text[1:])
        return caption


class BeamSearchDecoder(BaseDecoder):
    """Beam search decoding strategy.
    
    Maintains multiple hypotheses and selects the
    best one at the end. More thorough but slower.
    
    Example:
        >>> decoder = BeamSearchDecoder(tokenizer, max_length=38, beam_width=5)
        >>> caption = decoder.decode(model, image_features)
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int,
        beam_width: int = 3,
        length_penalty: float = 0.7,
    ) -> None:
        """Initialize beam search decoder.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding.
            max_length: Maximum caption length.
            beam_width: Number of beams to maintain.
            length_penalty: Penalty factor for length normalization.
        """
        super().__init__(tokenizer, max_length)
        self._beam_width = beam_width
        self._length_penalty = length_penalty
    
    @property
    def beam_width(self) -> int:
        """Get beam width."""
        return self._beam_width
    
    def decode(
        self,
        model: tf.keras.Model,
        image_features: np.ndarray,
    ) -> str:
        """Decode caption using beam search.
        
        Args:
            model: Trained caption model.
            image_features: Image feature vector of shape (1, feature_dim).
            
        Returns:
            Generated caption string.
        """
        wordtoix = self._tokenizer.word_to_index
        ixtoword = self._tokenizer.index_to_word
        
        start_token = wordtoix.get("startseq", 1)
        end_token = wordtoix.get("endseq", 2)
        
        # Initialize beams: (sequence, log_probability)
        beams = [([start_token], 0.0)]
        completed = []
        
        for _ in range(self._max_length):
            new_beams = []
            
            for seq, score in beams:
                # Check if already completed
                if seq[-1] == end_token:
                    completed.append((seq, score))
                    continue
                
                # Prepare input
                padded = tf.keras.preprocessing.sequence.pad_sequences(
                    [seq],
                    maxlen=self._max_length,
                    padding="post",
                )
                
                # Get predictions
                preds = model.predict(
                    [image_features, padded],
                    verbose=0,
                )[0]
                
                # Clip for numerical stability
                preds = np.clip(preds, 1e-9, 1.0)
                
                # Get top-k candidates
                top_indices = np.argsort(preds)[-self._beam_width:]
                
                for idx in top_indices:
                    new_seq = seq + [int(idx)]
                    new_score = score + np.log(preds[idx])
                    new_beams.append((new_seq, new_score))
            
            if not new_beams:
                break
            
            # Keep top beams (length-normalized)
            beams = sorted(
                new_beams,
                key=lambda x: x[1] / (len(x[0]) ** self._length_penalty),
                reverse=True,
            )[:self._beam_width]
        
        # Select best sequence
        all_seqs = completed if completed else beams
        best_seq = max(
            all_seqs,
            key=lambda x: x[1] / (len(x[0]) ** self._length_penalty),
        )[0]
        
        # Decode to words
        words = []
        for idx in best_seq:
            word = ixtoword.get(idx)
            if word is None:
                continue
            if word == "endseq":
                break
            if word != "startseq":
                words.append(word)
        
        return " ".join(words)
