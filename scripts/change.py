"""Model conversion script.

This script rebuilds the caption model with the correct architecture
(mask_zero=False for embedding layer) and transfers weights from
an existing saved model. This is useful when you need to:
- Fix compatibility issues between training and inference
- Update model architecture while preserving learned weights
- Convert models between different Keras save formats

Usage:
    python scripts/change.py [--input MODEL_PATH] [--output OUTPUT_PATH]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default paths
DEFAULT_ARTIFACTS_DIR = project_root / "shared" / "artifacts"
DEFAULT_INPUT_MODEL = DEFAULT_ARTIFACTS_DIR / "image_caption_model_final.keras"
DEFAULT_OUTPUT_MODEL = DEFAULT_ARTIFACTS_DIR / "image_caption_model_final_0.keras"
DEFAULT_VOCAB_PATH = DEFAULT_ARTIFACTS_DIR / "wordtoix.json"


def load_vocabulary(vocab_path: Path) -> int:
    """Load vocabulary and return vocab size.
    
    Args:
        vocab_path: Path to wordtoix.json file.
        
    Returns:
        Vocabulary size (number of words + 1 for padding).
    """
    import json
    
    with open(vocab_path, "r") as f:
        word_to_ix = json.load(f)
    
    vocab_size = len(word_to_ix) + 1  # +1 for padding/unknown token
    logger.info(f"Loaded vocabulary with {len(word_to_ix)} words (vocab_size={vocab_size})")
    return vocab_size


def build_caption_model(
    vocab_size: int,
    max_length: int = 38,
    feature_dim: int = 1536,
    embedding_dim: int = 200,
    hidden_dim: int = 256,
    num_attention_heads: int = 4,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.2,
) -> tf.keras.Model:
    """Build caption model with functional API.
    
    This builds the same architecture as the training model
    but with mask_zero=False for inference compatibility.
    
    Args:
        vocab_size: Size of vocabulary.
        max_length: Maximum sequence length.
        feature_dim: Dimension of image features (EfficientNetB3 = 1536).
        embedding_dim: Word embedding dimension.
        hidden_dim: Hidden dimension for LSTM and attention.
        num_attention_heads: Number of attention heads.
        dropout_rate: Dropout rate.
        recurrent_dropout: LSTM recurrent dropout.
        
    Returns:
        Compiled Keras model.
    """
    # Inputs
    image_input = tf.keras.Input(shape=(feature_dim,), name="image_features")
    text_input = tf.keras.Input(shape=(max_length,), name="text_input")
    
    # Image branch - project to hidden dimension
    img_proj = tf.keras.layers.Dense(hidden_dim, activation="relu")(image_input)
    img_proj = tf.keras.layers.Dropout(dropout_rate)(img_proj)
    img_proj = tf.keras.layers.LayerNormalization()(img_proj)
    
    # Text branch - embedding with mask_zero=False for inference
    # Note: mask_zero=False avoids issues with some layers during inference
    text_emb = tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=False,  # Important: set to False for inference compatibility
        trainable=False,
    )(text_input)
    
    # LSTM encoder
    lstm_out = tf.keras.layers.LSTM(
        hidden_dim,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
    )(text_emb)
    
    # Attention mechanism
    query = tf.keras.layers.Reshape((1, hidden_dim))(img_proj)
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=hidden_dim,
    )(query, lstm_out)
    context = tf.keras.layers.GlobalAveragePooling1D()(attention)
    
    # Fusion layer
    merged = tf.keras.layers.Concatenate()([context, img_proj])
    merged = tf.keras.layers.Dense(hidden_dim, activation="relu")(merged)
    merged = tf.keras.layers.Dropout(0.5)(merged)
    
    # Output layer with explicit float32 for mixed precision compatibility
    output = tf.keras.layers.Dense(
        vocab_size,
        activation="softmax",
        dtype="float32",
    )(merged)
    
    # Build model
    model = tf.keras.Model(
        inputs=[image_input, text_input],
        outputs=output,
        name="caption_model",
    )
    
    return model


def convert_model(
    input_path: Path,
    output_path: Path,
    vocab_size: int,
    max_length: int = 38,
) -> None:
    """Convert model by rebuilding architecture and transferring weights.
    
    Args:
        input_path: Path to source model.
        output_path: Path to save converted model.
        vocab_size: Vocabulary size.
        max_length: Maximum sequence length.
    """
    logger.info(f"Loading source model from: {input_path}")
    
    # Clear any existing session
    tf.keras.backend.clear_session()
    
    # Load the source model
    source_model = tf.keras.models.load_model(input_path, compile=False)
    logger.info(f"Source model loaded: {source_model.count_params():,} parameters")
    
    # Build new model with same architecture
    logger.info("Building new model with updated architecture...")
    new_model = build_caption_model(
        vocab_size=vocab_size,
        max_length=max_length,
    )
    logger.info(f"New model built: {new_model.count_params():,} parameters")
    
    # Transfer weights
    logger.info("Transferring weights...")
    try:
        # Try to load weights directly (works if architectures match)
        new_model.set_weights(source_model.get_weights())
        logger.info("Weights transferred successfully using set_weights()")
    except ValueError as e:
        logger.warning(f"Direct weight transfer failed: {e}")
        logger.info("Attempting layer-by-layer weight transfer...")
        
        # Layer-by-layer transfer for more flexibility
        for src_layer, dst_layer in zip(source_model.layers, new_model.layers):
            try:
                weights = src_layer.get_weights()
                if weights:
                    dst_layer.set_weights(weights)
                    logger.debug(f"Transferred weights for layer: {src_layer.name}")
            except Exception as layer_error:
                logger.warning(
                    f"Could not transfer weights for layer {src_layer.name}: {layer_error}"
                )
    
    # Verify weight transfer by comparing a few layers
    logger.info("Verifying weight transfer...")
    for i, (src_layer, dst_layer) in enumerate(
        zip(source_model.layers[:5], new_model.layers[:5])
    ):
        src_weights = src_layer.get_weights()
        dst_weights = dst_layer.get_weights()
        if src_weights:
            for sw, dw in zip(src_weights, dst_weights):
                if not np.allclose(sw, dw):
                    logger.warning(f"Weight mismatch detected in layer {i}: {src_layer.name}")
                    break
    
    # Compile the new model (optional, for inference we often don't need this)
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    
    # Save the converted model
    logger.info(f"Saving converted model to: {output_path}")
    new_model.save(output_path)
    logger.info("Model conversion completed successfully!")
    
    # Print model summary
    logger.info("\nModel Summary:")
    new_model.summary(print_fn=logger.info)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert/rebuild caption model with updated architecture."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_MODEL,
        help=f"Path to input model (default: {DEFAULT_INPUT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_MODEL,
        help=f"Path to output model (default: {DEFAULT_OUTPUT_MODEL})",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help=f"Path to vocabulary file (default: {DEFAULT_VOCAB_PATH})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=38,
        help="Maximum sequence length (default: 38)",
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input.exists():
        logger.error(f"Input model not found: {args.input}")
        sys.exit(1)
    
    if not args.vocab.exists():
        logger.error(f"Vocabulary file not found: {args.vocab}")
        sys.exit(1)
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    vocab_size = load_vocabulary(args.vocab)
    
    # Convert model
    convert_model(
        input_path=args.input,
        output_path=args.output,
        vocab_size=vocab_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
