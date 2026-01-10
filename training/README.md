# Training Module

Production-grade training pipeline for image captioning models with attention mechanisms.

## Architecture

```
training/
├── __init__.py              # Package exports
├── train.py                 # Main entrypoint (TrainingPipeline class)
├── config.yaml              # Default configuration
├── README.md
│
├── configs/                 # Configuration management
│   ├── __init__.py
│   └── training_config.py   # Dataclass-based configs with validation
│
├── data/                    # Data processing
│   ├── __init__.py
│   ├── preprocessor.py      # Text cleaning (TextPreprocessor, CaptionPreprocessor)
│   ├── caption_loader.py    # Caption loading and splitting (CaptionLoader, DataSplit)
│   ├── tokenizer.py         # Vocabulary and embeddings (Tokenizer, GloVeEmbeddings)
│   └── dataset.py           # tf.data pipelines (DatasetBuilder, TrainingSample)
│
├── features/                # Image feature extraction
│   ├── __init__.py
│   ├── base.py              # Abstract base class (BaseFeatureExtractor)
│   ├── efficientnet.py      # EfficientNet/InceptionV3 extractors
│   └── registry.py          # Registry pattern for extractor lookup
│
├── models/                  # Model architecture
│   ├── __init__.py
│   ├── caption_model.py     # Main model (CaptionModel, build_caption_model)
│   └── components/
│       ├── __init__.py
│       ├── attention.py     # MultiHeadCrossAttention, BahdanauAttention
│       └── encoders.py      # ImageProjection, TextEmbedding, LSTMEncoder
│
├── trainers/                # Training logic
│   ├── __init__.py
│   └── trainer.py           # Trainer, DistributedTrainer, CallbackFactory
│
├── evaluation/              # Evaluation and inference
│   ├── __init__.py
│   ├── metrics.py           # BLEUScore, CaptionEvaluator
│   └── inference.py         # GreedyDecoder, BeamSearchDecoder
│
├── callbacks/               # Custom callbacks
│   ├── __init__.py
│   └── metrics.py           # MetricsCallback, ProgressCallback
│
└── utils/                   # Utilities
    ├── __init__.py
    ├── io.py                # File I/O (save_json, load_json, ensure_dir)
    ├── logging.py           # Logging setup
    └── mixed_precision.py   # GPU and mixed precision configuration
```

## Quick Start

### 1. Configure Paths

Edit `config.yaml` to set your data paths:

```yaml
images_path: "/path/to/flickr8k/Images/"
captions_file: "/path/to/flickr8k/captions.txt"
glove_path: "/path/to/glove.6B.200d.txt"
```

### 2. Run Training

```bash
# From repository root
python -m training.train --config training/config.yaml
```

### 3. Artifacts

Models and tokenizer files are saved to `shared/artifacts/`:
- `image_caption_model_final.keras` - Trained model
- `checkpoint.keras` - Best checkpoint
- `wordtoix.json` - Word to index mapping
- `ixtoword.json` - Index to word mapping
- `features.npy` - Extracted image features

## Key Design Patterns

### 1. **Dataclass Configuration**
Type-safe configuration with validation:
```python
from training.configs import load_config
config = load_config("config.yaml")
print(config.model.feature_dim)  # 1536
```

### 2. **Registry Pattern (Feature Extractors)**
Extensible feature extractor registration:
```python
from training.features import get_feature_extractor
extractor = get_feature_extractor("EfficientNetB3", image_size=(300, 300))
```

### 3. **Abstract Base Classes**
Clean interfaces for extensibility:
```python
from training.features.base import BaseFeatureExtractor

class MyCustomExtractor(BaseFeatureExtractor):
    def build_model(self): ...
    def extract_features(self, ...): ...
```

### 4. **Factory Pattern (Callbacks)**
Standardized callback creation:
```python
from training.trainers.trainer import CallbackFactory
checkpoint = CallbackFactory.model_checkpoint("model.keras")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_extractor` | `EfficientNetB3` | CNN backbone |
| `feature_dim` | `1536` | Feature dimension |
| `embedding_dim` | `200` | GloVe embedding size |
| `hidden_dim` | `256` | LSTM/attention dimension |
| `batch_size` | `64` | Training batch size |
| `epochs` | `20` | Maximum epochs |
| `learning_rate` | `0.0001` | Initial learning rate |
| `use_mixed_precision` | `true` | Enable float16 training |

## Extending the Pipeline

### Add a New Feature Extractor

```python
# training/features/my_extractor.py
from training.features.base import BaseFeatureExtractor

class MyExtractor(BaseFeatureExtractor):
    FEATURE_DIM = 2048
    NAME = "MyModel"
    
    def build_model(self):
        # Build your model
        pass
    
    def extract_features(self, image_keys, images_path, verbose=1):
        # Extract features
        pass

# Register it
from training.features.registry import FeatureExtractorRegistry
registry = FeatureExtractorRegistry()
registry.register("MyModel", MyExtractor)
```

### Custom Training Callback

```python
from training.callbacks.metrics import MetricsCallback

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Custom logic
        pass
```

## Requirements

- Python >= 3.9
- TensorFlow >= 2.10
- NumPy
- PyYAML
- tqdm
- NLTK (for BLEU evaluation)
