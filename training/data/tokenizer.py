"""Tokenization and vocabulary management.

This module provides vocabulary building and word-to-index mapping
functionality for training caption models.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Tokenizer:
    """Tokenizer for converting text to sequences of indices.
    
    Builds vocabulary from captions, creates word-to-index mappings,
    and handles GloVe embedding loading.
    
    Attributes:
        vocab_size: Size of vocabulary (including padding).
        word_to_index: Mapping from words to indices.
        index_to_word: Mapping from indices to words.
    
    Example:
        >>> tokenizer = Tokenizer()
        >>> tokenizer.fit(["a dog runs", "a cat sits"])
        >>> tokenizer.encode("a dog")
        [1, 2]
    """
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(
        self,
        min_word_count: int = 5,
        max_vocab_size: Optional[int] = None,
    ) -> None:
        """Initialize the tokenizer.
        
        Args:
            min_word_count: Minimum word frequency for vocabulary.
            max_vocab_size: Maximum vocabulary size (None for unlimited).
        """
        self._min_word_count = min_word_count
        self._max_vocab_size = max_vocab_size
        self._word_to_index: Dict[str, int] = {}
        self._index_to_word: Dict[int, str] = {}
        self._word_counts: Counter = Counter()
        self._is_fitted = False
    
    @property
    def vocab_size(self) -> int:
        """Size of vocabulary including padding token."""
        return len(self._word_to_index) + 1  # +1 for padding at index 0
    
    @property
    def word_to_index(self) -> Dict[str, int]:
        """Word to index mapping."""
        return self._word_to_index.copy()
    
    @property
    def index_to_word(self) -> Dict[int, str]:
        """Index to word mapping."""
        return self._index_to_word.copy()
    
    @property
    def is_fitted(self) -> bool:
        """Whether tokenizer has been fitted."""
        return self._is_fitted
    
    def fit(self, captions: List[str]) -> "Tokenizer":
        """Build vocabulary from captions.
        
        Args:
            captions: List of caption strings.
            
        Returns:
            Self for method chaining.
        """
        # Count word frequencies
        self._word_counts = Counter()
        for caption in captions:
            self._word_counts.update(caption.split())
        
        # Filter by minimum count
        vocab = [
            word for word, count in self._word_counts.items()
            if count >= self._min_word_count
        ]
        
        # Optionally limit vocabulary size
        if self._max_vocab_size is not None:
            vocab = sorted(
                vocab,
                key=lambda w: self._word_counts[w],
                reverse=True
            )[:self._max_vocab_size]
        
        # Build mappings (index 0 reserved for padding)
        self._word_to_index = {}
        self._index_to_word = {}
        
        for idx, word in enumerate(sorted(vocab), start=1):
            self._word_to_index[word] = idx
            self._index_to_word[idx] = word
        
        self._is_fitted = True
        logger.info(f"Built vocabulary with {self.vocab_size} tokens")
        
        return self
    
    def encode(self, text: str) -> List[int]:
        """Encode text to sequence of indices.
        
        Args:
            text: Input text string.
            
        Returns:
            List of token indices.
            
        Raises:
            RuntimeError: If tokenizer hasn't been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted before encoding")
        
        return [
            self._word_to_index[word]
            for word in text.split()
            if word in self._word_to_index
        ]
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Decode sequence of indices to text.
        
        Args:
            indices: List of token indices.
            skip_special: Whether to skip special tokens like startseq/endseq.
            
        Returns:
            Decoded text string.
        """
        words = []
        special_tokens = {"startseq", "endseq"} if skip_special else set()
        
        for idx in indices:
            if idx == 0:  # Padding
                continue
            word = self._index_to_word.get(idx)
            if word is None:
                continue
            if word in special_tokens:
                continue
            if word == "endseq":
                break
            words.append(word)
        
        return " ".join(words)
    
    def save(self, directory: str | Path) -> None:
        """Save tokenizer state to directory.
        
        Args:
            directory: Directory to save files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        with open(directory / "wordtoix.json", "w") as f:
            json.dump(self._word_to_index, f)
        
        # Save with string keys for JSON compatibility
        index_to_word_str = {str(k): v for k, v in self._index_to_word.items()}
        with open(directory / "ixtoword.json", "w") as f:
            json.dump(index_to_word_str, f)
        
        logger.info(f"Saved tokenizer to {directory}")
    
    @classmethod
    def load(cls, directory: str | Path) -> "Tokenizer":
        """Load tokenizer state from directory.
        
        Args:
            directory: Directory containing saved files.
            
        Returns:
            Loaded Tokenizer instance.
        """
        directory = Path(directory)
        
        with open(directory / "wordtoix.json", "r") as f:
            word_to_index = json.load(f)
        
        with open(directory / "ixtoword.json", "r") as f:
            index_to_word_str = json.load(f)
        
        tokenizer = cls()
        tokenizer._word_to_index = word_to_index
        tokenizer._index_to_word = {int(k): v for k, v in index_to_word_str.items()}
        tokenizer._is_fitted = True
        
        logger.info(f"Loaded tokenizer from {directory}")
        return tokenizer


class GloVeEmbeddings:
    """GloVe word embeddings loader and manager.
    
    Handles loading pre-trained GloVe embeddings and creating
    embedding matrices for model training.
    
    Example:
        >>> glove = GloVeEmbeddings("/path/to/glove.6B.200d.txt", dim=200)
        >>> matrix = glove.create_embedding_matrix(tokenizer)
    """
    
    def __init__(
        self,
        glove_path: str | Path,
        embedding_dim: int = 200,
    ) -> None:
        """Initialize GloVe embeddings loader.
        
        Args:
            glove_path: Path to GloVe embeddings file.
            embedding_dim: Dimension of embeddings.
        """
        self._glove_path = Path(glove_path)
        self._embedding_dim = embedding_dim
        self._embeddings: Dict[str, np.ndarray] = {}
        self._is_loaded = False
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of embeddings."""
        return self._embedding_dim
    
    @property
    def is_loaded(self) -> bool:
        """Whether embeddings have been loaded."""
        return self._is_loaded
    
    def load(self) -> "GloVeEmbeddings":
        """Load embeddings from file.
        
        Returns:
            Self for method chaining.
            
        Raises:
            FileNotFoundError: If GloVe file doesn't exist.
        """
        if not self._glove_path.exists():
            raise FileNotFoundError(f"GloVe file not found: {self._glove_path}")
        
        # Count lines for progress bar
        with open(self._glove_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        
        self._embeddings = {}
        with open(self._glove_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="Loading GloVe"):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype=np.float32)
                if len(vector) == self._embedding_dim:
                    self._embeddings[word] = vector
        
        self._is_loaded = True
        logger.info(f"Loaded {len(self._embeddings)} GloVe embeddings")
        
        return self
    
    def create_embedding_matrix(self, tokenizer: Tokenizer) -> np.ndarray:
        """Create embedding matrix for tokenizer vocabulary.
        
        Args:
            tokenizer: Fitted tokenizer instance.
            
        Returns:
            Embedding matrix of shape (vocab_size, embedding_dim).
            
        Raises:
            RuntimeError: If embeddings haven't been loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("GloVe embeddings must be loaded first")
        
        vocab_size = tokenizer.vocab_size
        embedding_matrix = np.zeros((vocab_size, self._embedding_dim), dtype=np.float32)
        
        found = 0
        for word, idx in tokenizer.word_to_index.items():
            vector = self._embeddings.get(word)
            if vector is not None:
                embedding_matrix[idx] = vector
                found += 1
        
        coverage = found / (vocab_size - 1) * 100  # -1 for padding
        logger.info(
            f"Created embedding matrix: {found}/{vocab_size-1} words "
            f"({coverage:.1f}% coverage)"
        )
        
        return embedding_matrix
