"""Evaluation metrics for image captioning.

This module provides metrics for evaluating generated
captions against reference captions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BLEUScore:
    """Container for BLEU scores.
    
    Attributes:
        bleu1: BLEU-1 (unigram precision).
        bleu2: BLEU-2 (bigram precision).
        bleu3: BLEU-3 (trigram precision).
        bleu4: BLEU-4 (4-gram precision).
    """
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"BLEU-1: {self.bleu1:.4f}, "
            f"BLEU-2: {self.bleu2:.4f}, "
            f"BLEU-3: {self.bleu3:.4f}, "
            f"BLEU-4: {self.bleu4:.4f}"
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "bleu1": self.bleu1,
            "bleu2": self.bleu2,
            "bleu3": self.bleu3,
            "bleu4": self.bleu4,
        }


class CaptionEvaluator:
    """Evaluator for image captioning models.
    
    Computes BLEU scores and other metrics for
    evaluating caption quality.
    
    Example:
        >>> evaluator = CaptionEvaluator()
        >>> scores = evaluator.evaluate(predictions, references)
        >>> print(scores)
        BLEU-1: 0.6543, BLEU-2: 0.4523, ...
    """
    
    def __init__(
        self,
        smoothing: bool = True,
    ) -> None:
        """Initialize evaluator.
        
        Args:
            smoothing: Whether to use smoothing for BLEU.
        """
        self._smoothing = smoothing
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> BLEUScore:
        """Evaluate predictions against references.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference caption lists
                (multiple references per image).
            
        Returns:
            BLEUScore with computed metrics.
        """
        from nltk.translate.bleu_score import (
            corpus_bleu,
            SmoothingFunction,
        )
        
        # Tokenize predictions and references
        tokenized_preds = [pred.split() for pred in predictions]
        tokenized_refs = [
            [ref.split() for ref in refs]
            for refs in references
        ]
        
        # Remove startseq/endseq tokens if present
        tokenized_preds = [
            [w for w in pred if w not in ("startseq", "endseq")]
            for pred in tokenized_preds
        ]
        tokenized_refs = [
            [
                [w for w in ref if w not in ("startseq", "endseq")]
                for ref in refs
            ]
            for refs in tokenized_refs
        ]
        
        # Compute BLEU scores
        smooth = SmoothingFunction().method1 if self._smoothing else None
        
        bleu1 = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            weights=(1.0, 0, 0, 0),
            smoothing_function=smooth,
        )
        
        bleu2 = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smooth,
        )
        
        bleu3 = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=smooth,
        )
        
        bleu4 = corpus_bleu(
            tokenized_refs,
            tokenized_preds,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth,
        )
        
        return BLEUScore(
            bleu1=bleu1,
            bleu2=bleu2,
            bleu3=bleu3,
            bleu4=bleu4,
        )
    
    def evaluate_batch(
        self,
        model,
        decoder,
        features: Dict[str, np.ndarray],
        references: Dict[str, List[str]],
        sample_size: Optional[int] = None,
    ) -> Tuple[BLEUScore, List[Tuple[str, str, List[str]]]]:
        """Evaluate model on a batch of images.
        
        Args:
            model: Caption model.
            decoder: Caption decoder (greedy or beam search).
            features: Dictionary of image features.
            references: Dictionary of reference captions.
            sample_size: Number of samples to evaluate (None for all).
            
        Returns:
            Tuple of (BLEUScore, list of (image_id, prediction, references)).
        """
        keys = list(features.keys())
        if sample_size:
            keys = keys[:sample_size]
        
        predictions = []
        ref_list = []
        results = []
        
        for key in keys:
            if key not in references:
                continue
            
            feature = features[key]
            if len(feature.shape) == 1:
                feature = np.expand_dims(feature, 0)
            
            pred = decoder.decode(model, feature)
            refs = references[key]
            
            predictions.append(pred)
            ref_list.append(refs)
            results.append((key, pred, refs))
        
        scores = self.evaluate(predictions, ref_list)
        
        return scores, results
