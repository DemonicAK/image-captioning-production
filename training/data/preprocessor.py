"""Text preprocessing utilities.

This module provides text cleaning and normalization functionality
for caption preprocessing.
"""

from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from typing import List


class BasePreprocessor(ABC):
    """Abstract base class for text preprocessors."""
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocess a single text string.
        
        Args:
            text: Input text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        pass
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            List of preprocessed texts.
        """
        return [self.preprocess(text) for text in texts]


class TextPreprocessor(BasePreprocessor):
    """Text preprocessor for image captions.
    
    Performs the following transformations:
    - Lowercase conversion
    - Punctuation removal
    - Whitespace normalization
    - Optional additional cleaning
    
    Attributes:
        remove_punctuation: Whether to remove punctuation.
        lowercase: Whether to convert to lowercase.
        normalize_whitespace: Whether to normalize whitespace.
    
    Example:
        >>> preprocessor = TextPreprocessor()
        >>> preprocessor.preprocess("A Dog is Running!")
        'a dog is running'
    """
    
    def __init__(
        self,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        normalize_whitespace: bool = True,
    ) -> None:
        """Initialize the text preprocessor.
        
        Args:
            remove_punctuation: Whether to remove punctuation.
            lowercase: Whether to convert to lowercase.
            normalize_whitespace: Whether to normalize whitespace.
        """
        self._remove_punctuation = remove_punctuation
        self._lowercase = lowercase
        self._normalize_whitespace = normalize_whitespace
        self._punctuation_table = str.maketrans("", "", string.punctuation)
    
    @property
    def remove_punctuation(self) -> bool:
        """Whether punctuation is removed."""
        return self._remove_punctuation
    
    @property
    def lowercase(self) -> bool:
        """Whether text is lowercased."""
        return self._lowercase
    
    @property
    def normalize_whitespace(self) -> bool:
        """Whether whitespace is normalized."""
        return self._normalize_whitespace
    
    def preprocess(self, text: str) -> str:
        """Preprocess a single text string.
        
        Args:
            text: Input text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        if not text:
            return ""
        
        result = text
        
        if self._lowercase:
            result = result.lower()
        
        if self._remove_punctuation:
            result = result.translate(self._punctuation_table)
        
        if self._normalize_whitespace:
            result = " ".join(result.split())
        
        return result


class CaptionPreprocessor(TextPreprocessor):
    """Specialized preprocessor for image captions.
    
    Extends TextPreprocessor with caption-specific functionality
    like adding sequence tokens.
    
    Attributes:
        start_token: Token to add at the beginning of sequences.
        end_token: Token to add at the end of sequences.
    """
    
    START_TOKEN = "startseq"
    END_TOKEN = "endseq"
    
    def __init__(
        self,
        start_token: str = START_TOKEN,
        end_token: str = END_TOKEN,
        **kwargs,
    ) -> None:
        """Initialize caption preprocessor.
        
        Args:
            start_token: Token to add at sequence start.
            end_token: Token to add at sequence end.
            **kwargs: Additional arguments for TextPreprocessor.
        """
        super().__init__(**kwargs)
        self._start_token = start_token
        self._end_token = end_token
    
    @property
    def start_token(self) -> str:
        """Start sequence token."""
        return self._start_token
    
    @property
    def end_token(self) -> str:
        """End sequence token."""
        return self._end_token
    
    def add_sequence_tokens(self, text: str) -> str:
        """Add start and end tokens to text.
        
        Args:
            text: Input text.
            
        Returns:
            Text with sequence tokens added.
        """
        return f"{self._start_token} {text} {self._end_token}"
    
    def preprocess_with_tokens(self, text: str) -> str:
        """Preprocess text and add sequence tokens.
        
        Args:
            text: Input text.
            
        Returns:
            Preprocessed text with sequence tokens.
        """
        processed = self.preprocess(text)
        return self.add_sequence_tokens(processed)
