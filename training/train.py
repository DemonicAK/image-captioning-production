"""Training entrypoint for image captioning model.

This script orchestrates the full training pipeline:
1. Load configuration from YAML
2. Load and preprocess data
3. Extract image features
4. Build and train model
5. Save artifacts

Usage:
    python -m training.train --config training/config.yaml

Configuration:
    Edit training/config.yaml to customize paths and hyperparameters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.configs import load_config, Config
from training.data import CaptionLoader, Tokenizer, DatasetBuilder
from training.data.tokenizer import GloVeEmbeddings
from training.features import get_feature_extractor
from training.models import build_caption_model
from training.trainers import Trainer
from training.utils import (
    setup_logging,
    get_logger,
    setup_mixed_precision,
    setup_gpu,
    ensure_dir,
)

logger = get_logger(__name__)


class TrainingPipeline:
    """Orchestrates the complete training pipeline.
    
    Handles all stages from data loading to model saving
    in a clean, modular way.
    
    Example:
        >>> pipeline = TrainingPipeline("config.yaml")
        >>> pipeline.run()
    """
    
    def __init__(self, config_path: str | Path) -> None:
        """Initialize training pipeline.
        
        Args:
            config_path: Path to YAML configuration file.
        """
        self._config_path = Path(config_path)
        self._config: Config | None = None
        self._artifacts_dir: Path | None = None
        
        # Components (initialized lazily)
        self._tokenizer: Tokenizer | None = None
        self._embedding_matrix = None
        self._max_length: int = 0
    
    def _setup(self) -> None:
        """Setup training environment."""
        setup_logging()
        logger.info("Setting up training environment")
        
        # Load configuration
        self._config = load_config(self._config_path)
        logger.info(f"Loaded configuration from {self._config_path}")
        
        # Setup artifacts directory
        self._artifacts_dir = Path(self._config.training.artifacts_dir)
        if not self._artifacts_dir.is_absolute():
            # Make relative to training directory
            self._artifacts_dir = (
                Path(__file__).parent.parent / 
                self._config.training.artifacts_dir
            )
        ensure_dir(self._artifacts_dir)
        logger.info(f"Artifacts will be saved to {self._artifacts_dir}")
        
        # Setup GPU and mixed precision
        setup_gpu(memory_growth=True)
        if self._config.training.use_mixed_precision:
            setup_mixed_precision()
    
    def _load_data(self):
        """Load and preprocess caption data."""
        logger.info("Loading caption data")
        cfg = self._config.data
        
        # Load captions
        loader = CaptionLoader()
        descriptions = loader.load(cfg.captions_file)
        
        # Create splits
        splits = loader.create_splits(
            descriptions,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            seed=cfg.random_seed,
        )
        
        # Build tokenizer
        self._tokenizer = Tokenizer(min_word_count=cfg.word_count_threshold)
        self._tokenizer.fit(splits.all_train_captions)
        
        # Save tokenizer
        self._tokenizer.save(self._artifacts_dir)
        
        # Get max length
        all_descriptions = {**splits.train, **splits.val, **splits.test}
        self._max_length = loader.get_max_caption_length(all_descriptions)
        logger.info(f"Max caption length: {self._max_length}")
        
        return splits
    
    def _load_embeddings(self):
        """Load GloVe embeddings."""
        logger.info("Loading GloVe embeddings")
        cfg = self._config
        
        glove = GloVeEmbeddings(
            glove_path=cfg.data.glove_path,
            embedding_dim=cfg.model.embedding_dim,
        )
        glove.load()
        
        self._embedding_matrix = glove.create_embedding_matrix(self._tokenizer)
    
    def _extract_features(self, image_keys):
        """Extract image features."""
        logger.info("Extracting image features")
        cfg = self._config
        
        extractor = get_feature_extractor(
            name=cfg.model.feature_extractor,
            image_size=cfg.model.image_size,
            batch_size=32,
            build=True,
        )
        
        features = extractor.extract_features(
            image_keys=image_keys,
            images_path=cfg.data.images_path,
            verbose=1,
        )
        
        # Save features
        features_path = self._artifacts_dir / "features.npy"
        extractor.save_features(features, features_path)
        logger.info(f"Saved features to {features_path}")
        
        return features
    
    def _build_datasets(self, splits, features):
        """Build training datasets."""
        logger.info("Building datasets")
        cfg = self._config
        
        builder = DatasetBuilder(
            max_length=self._max_length,
            feature_dim=cfg.model.feature_dim,
            batch_size=cfg.training.batch_size,
        )
        
        # Build samples
        train_samples = builder.build_samples(
            splits.train,
            features,
            self._tokenizer,
        )
        val_samples = builder.build_samples(
            splits.val,
            features,
            self._tokenizer,
        )
        
        # Create datasets
        train_ds = builder.create_dataset(
            train_samples,
            features,
            shuffle=True,
            repeat=True,
        )
        val_ds = builder.create_dataset(
            val_samples,
            features,
            shuffle=False,
            repeat=True,
        )
        
        # Compute steps
        steps_per_epoch = builder.compute_steps_per_epoch(len(train_samples))
        val_steps = builder.compute_steps_per_epoch(len(val_samples))
        
        return train_ds, val_ds, steps_per_epoch, val_steps
    
    def _build_model(self):
        """Build and compile the model."""
        logger.info("Building model")
        cfg = self._config
        
        model = build_caption_model(
            vocab_size=self._tokenizer.vocab_size,
            embedding_matrix=self._embedding_matrix,
            max_length=self._max_length,
            feature_dim=cfg.model.feature_dim,
            embedding_dim=cfg.model.embedding_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_attention_heads=cfg.model.num_attention_heads,
            dropout_rate=cfg.model.dropout_rate,
            recurrent_dropout=cfg.model.recurrent_dropout,
            learning_rate=cfg.training.learning_rate,
        )
        
        model.summary()
        return model
    
    def _train(self, model, train_ds, val_ds, steps_per_epoch, val_steps):
        """Train the model."""
        logger.info("Starting training")
        
        trainer = Trainer(
            model=model,
            config=self._config,
            artifacts_dir=str(self._artifacts_dir),
        )
        
        result = trainer.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
        )
        
        # Save final model
        model_path = trainer.save_model()
        result.model_path = model_path
        
        return result
    
    def run(self) -> None:
        """Execute the full training pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Image Captioning Training Pipeline")
        logger.info("=" * 60)
        
        # Setup
        self._setup()
        
        # Load data
        splits = self._load_data()
        
        # Load embeddings
        self._load_embeddings()
        
        # Get all image keys
        all_keys = sorted(
            set(splits.train_keys) | 
            set(splits.val_keys) | 
            set(splits.test_keys)
        )
        
        # Extract features
        features = self._extract_features(all_keys)
        
        # Build datasets
        train_ds, val_ds, steps_per_epoch, val_steps = self._build_datasets(
            splits, features
        )
        
        # Build model
        model = self._build_model()
        
        # Train
        result = self._train(
            model, train_ds, val_ds, steps_per_epoch, val_steps
        )
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Best epoch: {result.best_epoch}")
        logger.info(f"Final train loss: {result.final_train_loss:.4f}")
        if result.final_val_loss:
            logger.info(f"Final val loss: {result.final_val_loss:.4f}")
        logger.info(f"Model saved to: {result.model_path}")
        logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train image captioning model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to configuration YAML file",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    pipeline = TrainingPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
