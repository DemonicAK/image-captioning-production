# Image Captioning Production – Developer Guide

Complete technical reference for working with the image captioning codebase.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Experiment Tracking](#experiment-tracking)
3. [Training Pipeline](#training-pipeline)
4. [Testing](#testing)
5. [Configuration](#configuration)
6. [Extending the System](#extending-the-system)
7. [Kaggle Setup](#kaggle-setup)

---

## Architecture Overview

```
training/
├── configs/                 # Configuration management
│   └── training_config.py   # Dataclass-based configs with validation
├── data/                    # Data processing
│   ├── preprocessor.py      # Text cleaning
│   ├── caption_loader.py    # Caption loading and splitting
│   ├── tokenizer.py         # Vocabulary and GloVe embeddings
│   └── dataset.py           # tf.data pipelines
├── features/                # Image feature extraction
│   ├── base.py              # Abstract base class
│   ├── efficientnet.py      # EfficientNet/InceptionV3 extractors
│   └── registry.py          # Registry pattern
├── models/                  # Model architecture
│   ├── caption_model.py     # Main model builder
│   └── components/
│       ├── attention.py     # Multi-head attention
│       └── encoders.py      # Image projection, text embedding, LSTM
├── trainers/                # Training logic
│   └── trainer.py           # Trainer, DistributedTrainer, CallbackFactory
├── evaluation/              # Evaluation and inference
│   ├── metrics.py           # BLEU score computation
│   └── inference.py         # Greedy/beam search decoders
├── callbacks/               # Custom callbacks
│   └── metrics.py           # MetricsCallback, ProgressCallback
└── utils/                   # Utilities
    ├── experiment_tracking.py # Experiment tracking
    ├── io.py, logging.py, mixed_precision.py

inference/
├── main.py                  # FastAPI application
├── model_registry.py        # Thread-safe model management
├── caption_generator.py     # Caption generation algorithms
└── preprocessing.py         # Image preprocessing
```

---

## Experiment Tracking

### Core Module
**Location:** `training/utils/experiment_tracking.py`

### What Gets Tracked
1. **Run Metadata (`RunMetadata`)**
   - Unique run ID (timestamp + UUID)
   - Start/end time, status (running/completed/failed)
   - Git info: commit hash, branch, dirty status, remote URL
   - Environment: Python version, TensorFlow version, GPU availability

2. **Configuration Snapshot**
   - Complete config dict with MD5 hash (`_config_hash`)
   - Saved to `config_snapshot.json`

3. **Per-Epoch Metrics**
   - Train/val loss, learning rate, epoch time
   - Custom metrics (accuracy, BLEU interim, etc.)
   - Appended to `metrics.csv` after each epoch

4. **BLEU Scores**
   - Final evaluation scores (BLEU-1/2/3/4)
   - Logged via `log_bleu_scores()`

5. **TensorBoard Logs**
   - Histograms, scalars, model graph
   - Saved to `logs/` directory

6. **Artifacts**
   - JSON, NumPy arrays, or custom data
   - Saved via `log_artifact(name, data, artifact_type)`

### Reproducibility
```python
from training.utils.experiment_tracking import set_global_seed

# Sets seeds for Python random, NumPy, TensorFlow
# Also sets PYTHONHASHSEED and TF_DETERMINISTIC_OPS
set_global_seed(42)
```

### Usage Example
```python
from training.utils import ExperimentTracker, set_global_seed, get_git_info

# Set reproducibility seed
set_global_seed(42)

# Initialize tracker
tracker = ExperimentTracker(
    output_dir="experiments/run_001",
    run_name="efficientnet_baseline",
    seed=42,
)

# Log configuration
config_dict = {
    "model": {"feature_extractor": "EfficientNetB3", "hidden_dim": 256},
    "training": {"batch_size": 64, "epochs": 20, "learning_rate": 1e-4},
}
tracker.log_config(config_dict)

# During training: log metrics after each epoch
for epoch in range(20):
    # ... training code ...
    tracker.log_metrics(
        epoch=epoch + 1,
        train_loss=2.5,
        val_loss=2.8,
        learning_rate=1e-4,
        epoch_time_seconds=120.5,
        # Custom metrics:
        train_accuracy=0.85,
    )

# After training: log BLEU scores
tracker.log_bleu_scores(
    bleu1=0.65, bleu2=0.45, bleu3=0.35, bleu4=0.25,
    split="test",  # or "val"
)

# Log custom artifacts
tracker.log_artifact("predictions", predictions_dict, artifact_type="json")
tracker.log_artifact("embeddings", embedding_matrix, artifact_type="numpy")

# Finalize (writes run_summary.json)
summary_path = tracker.finalize()
# Or if training failed:
# tracker.finalize(error="CUDA out of memory")
```

### Trainer Integration
The `Trainer` class automatically creates and manages an `ExperimentTracker`:

```python
from training.trainers import Trainer
from training.configs import load_config

config = load_config("training/config.yaml")
trainer = Trainer(
    model=model,
    config=config,
    artifacts_dir="shared/artifacts",
    run_name="my_experiment",
)

# Tracker is auto-created and config is logged
# Metrics are logged automatically via TFMetricsLoggingCallback

result = trainer.train(train_ds, val_ds, steps_per_epoch=100)

# Log BLEU scores after training
trainer.log_bleu_scores(bleu1=0.65, bleu2=0.45, bleu3=0.35, bleu4=0.25)

# Finalize
summary_path = trainer.finalize()
```

### Output Files
After tracking, `ExperimentTracker` generates:

```
experiments/run_001/
├── config_snapshot.json       # Config with _config_hash
├── metrics.csv                # Epoch-wise metrics
├── run_summary.json           # Complete run metadata + final metrics
├── logs/                      # TensorBoard logs
│   └── events.out.tfevents.*
├── checkpoints/               # Model checkpoints
│   └── checkpoint.keras
└── artifacts/                 # Custom artifacts
    ├── predictions.json
    └── embeddings.npy
```

**`run_summary.json` structure:**
```json
{
  "metadata": {
    "run_id": "20240115_143022_abc12345",
    "run_name": "efficientnet_baseline",
    "start_time": "2024-01-15T14:30:22",
    "end_time": "2024-01-15T18:45:10",
    "status": "completed",
    "git_info": {
      "commit": "a1b2c3d4",
      "branch": "main",
      "dirty": false,
      "remote_url": "https://github.com/DemonicAK/image-captioning-production.git"
    },
    "environment": {
      "python_version": "3.10.12",
      "tensorflow_version": "2.15.0",
      "gpu_available": true,
      "gpu_devices": ["/physical_device:GPU:0"]
    }
  },
  "config": {...},
  "final_metrics": {
    "epoch": 20,
    "train_loss": 2.1,
    "val_loss": 2.3
  },
  "best_metrics": {
    "epoch": 18,
    "train_loss": 2.0,
    "val_loss": 2.2
  },
  "bleu_scores": {
    "test_bleu1": 0.65,
    "test_bleu2": 0.45,
    "test_bleu3": 0.35,
    "test_bleu4": 0.25
  },
  "total_epochs": 20
}
```

---

## Training Pipeline

### Entrypoint
**File:** `training/train.py`

**Class:** `TrainingPipeline` – orchestrates data loading, feature extraction, model building, and training

### Run Command
```bash
python -m training.train --config training/config.yaml
```

### Workflow
1. Load config from `training/config.yaml`
2. Setup GPU and mixed precision
3. Load captions and create train/val/test splits
4. Build tokenizer and save to artifacts
5. Load GloVe embeddings
6. Extract image features using EfficientNetB3/InceptionV3
7. Build tf.data pipelines
8. Build attention-based caption model
9. Train with callbacks (checkpoint, early stopping, LR scheduler, TensorBoard)
10. Save final model and artifacts

### Output Artifacts
**Default location:** `shared/artifacts/`

```
shared/artifacts/
├── image_caption_model_final.keras  # Final trained model
├── checkpoint.keras                 # Best checkpoint (based on val_loss)
├── wordtoix.json                    # Word → index mapping
├── ixtoword.json                    # Index → word mapping
├── features.npy                     # Pre-extracted image features
├── config_snapshot.json             # Config with hash
├── metrics.csv                      # Per-epoch metrics
├── run_summary.json                 # Complete run summary
└── logs/                            # TensorBoard logs
    └── events.out.tfevents.*
```

### TensorBoard Visualization
```bash
tensorboard --logdir shared/artifacts/logs
# Open http://localhost:6006
```

### Pipeline Stages
```python
from training.train import TrainingPipeline

pipeline = TrainingPipeline("training/config.yaml")

# Internally calls:
# 1. _setup() - Load config, setup GPU/mixed precision
# 2. _load_data() - Load captions, create splits, build tokenizer
# 3. _load_embeddings() - Load GloVe and create embedding matrix
# 4. _extract_features() - Extract CNN features from images
# 5. _build_datasets() - Create tf.data pipelines
# 6. _build_model() - Build attention-based caption model
# 7. _train() - Train with Trainer class

pipeline.run()
```

---

## Testing

### Test Structure
**Framework:** pytest with custom markers

**Configuration:** `pytest.ini`

```ini
[pytest]
testpaths = tests
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
```

### Test Suites

| File | Coverage |
|------|----------|
| `test_config.py` | Config validation, dataclass constraints |
| `test_tokenizer.py` | Vocabulary building, GloVe embeddings |
| `test_dataset.py` | tf.data pipeline, sample generation |
| `test_decoder.py` | Greedy/beam search decoding |
| `test_experiment_tracking.py` | Experiment tracking, metadata |
| `test_integration.py` | FastAPI endpoints, caption generation |
| `test_performance.py` | Memory usage, latency benchmarks |

### Setup Test Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or: . .venv/bin/activate

# Install test dependencies
pip install -r requirements-test.txt

# Install TensorFlow (required for full suite)
pip install tensorflow  # or tensorflow-gpu for GPU
```

### Run Tests
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=training --cov=inference --cov-report=html
# Open htmlcov/index.html in browser

# Quick run (config validation only, no TensorFlow needed)
pytest tests/test_config.py -q

# Integration tests only
pytest -m integration -q

# Skip slow tests
pytest -m "not slow" -q

# Performance tests
pytest -m performance -v

# Specific test file
pytest tests/test_experiment_tracking.py -v

# Specific test function
pytest tests/test_config.py::TestDataConfigValidation::test_valid_config -v
```

### Dependencies Note
The full test suite requires TensorFlow because `tests/conftest.py` imports training modules that depend on `tf.data` and `tf.keras`. 

**Workaround for TensorFlow-free testing:**
```bash
# Only run tests that don't import training modules
pytest tests/test_config.py tests/test_experiment_tracking.py -k "not training"
```

### Test Fixtures
Common fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def valid_data_config():
    """Create valid DataConfig for testing."""
    return DataConfig(
        images_path="/path/images",
        captions_file="/path/captions.txt",
        glove_path="/path/glove.txt",
    )

@pytest.fixture
def sample_tokenizer():
    """Create tokenizer with sample vocabulary."""
    tokenizer = Tokenizer(min_word_count=1)
    captions = ["a dog runs", "a cat sits", "a bird flies"]
    tokenizer.fit(captions)
    return tokenizer
```

---

## Configuration

### Config File Structure
**File:** `training/config.yaml`

```yaml
# Data Configuration
images_path: "/kaggle/input/flickr8k/Images/"
captions_file: "/kaggle/input/flickr8k/captions.txt"
glove_path: "/kaggle/input/glove6b200d/glove.6B.200d.txt"

train_ratio: 0.6
val_ratio: 0.2
test_ratio: 0.2
word_count_threshold: 5  # Min word frequency
random_seed: 42

# Model Configuration
feature_extractor: "EfficientNetB3"  # or "InceptionV3"
feature_dim: 1536  # 1536 for EfficientNetB3, 2048 for InceptionV3
image_size: [300, 300]  # [299, 299] for InceptionV3

embedding_dim: 200  # GloVe dimension
hidden_dim: 256     # LSTM/attention dimension
num_attention_heads: 4

dropout_rate: 0.3
recurrent_dropout: 0.2

# Training Configuration
batch_size: 64
epochs: 20
learning_rate: 0.0001

lr_decay_factor: 0.5  # Reduce LR by this factor on plateau
lr_patience: 3        # Epochs to wait before reducing LR
min_lr: 0.000001

early_stopping_patience: 5

use_mixed_precision: true  # float16 for faster training

artifacts_dir: "../shared/artifacts"
```

### Dataclass-Based Config
```python
from training.configs import load_config, Config

# Load and validate config
config = load_config("training/config.yaml")

# Access nested configs
print(config.data.images_path)
print(config.model.feature_dim)
print(config.training.batch_size)

# Validation happens automatically
# Invalid values raise ValueError at load time
```

### Config Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `feature_extractor` | `EfficientNetB3`, `InceptionV3` | `EfficientNetB3` | CNN backbone |
| `feature_dim` | int | `1536` | Must match extractor output |
| `image_size` | [height, width] | `[300, 300]` | Input image size |
| `embedding_dim` | int | `200` | GloVe embedding size (50/100/200/300) |
| `hidden_dim` | int | `256` | LSTM/attention hidden size |
| `num_attention_heads` | int | `4` | Multi-head attention heads |
| `batch_size` | int | `64` | Training batch size |
| `epochs` | int | `20` | Maximum training epochs |
| `learning_rate` | float | `0.0001` | Initial learning rate |
| `use_mixed_precision` | bool | `true` | Enable float16 training |

---

## Extending the System

### Add Custom Feature Extractor
```python
# training/features/my_extractor.py
from training.features.base import BaseFeatureExtractor
import tensorflow as tf

class MyCustomExtractor(BaseFeatureExtractor):
    FEATURE_DIM = 2048
    NAME = "MyModel"
    
    def build_model(self) -> tf.keras.Model:
        """Build feature extraction model."""
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        base.trainable = False
        return base
    
    def extract_features(
        self,
        image_keys: List[str],
        images_path: str,
        verbose: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Extract features from images."""
        # Implementation here
        pass

# Register it
from training.features.registry import FeatureExtractorRegistry
registry = FeatureExtractorRegistry()
registry.register("MyModel", MyCustomExtractor)

# Use in config.yaml
# feature_extractor: "MyModel"
# feature_dim: 2048
```

### Add Custom Training Callback
```python
import tensorflow as tf

class CustomMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
    
    def on_epoch_end(self, epoch, logs=None):
        # Log custom metrics
        self.tracker.log_metrics(
            epoch=epoch + 1,
            custom_metric=logs.get("custom_metric"),
        )

# Use in training
from training.trainers import Trainer

trainer = Trainer(model, config, artifacts_dir="artifacts/")
callback = CustomMetricsCallback(trainer.tracker)
trainer.train(train_ds, val_ds, extra_callbacks=[callback])
```

### Add Custom Decoder Algorithm
```python
# training/evaluation/inference.py
from training.evaluation.inference import BaseDecoder

class TopKSamplingDecoder(BaseDecoder):
    def __init__(self, model, tokenizer, max_length, k=10):
        super().__init__(model, tokenizer, max_length)
        self.k = k
    
    def decode(self, image_features):
        """Decode with top-k sampling."""
        # Implementation here
        pass

# Use in inference
from inference.caption_generator import CaptionService

service = CaptionService(model_bundle)
service.register_algorithm("topk", TopKSamplingDecoder)
caption = service.generate_caption(features, algorithm="topk")
```

---

## Kaggle Setup

### Prerequisites
1. Kaggle account with notebook access
2. GPU enabled in notebook settings
3. Add datasets as input sources:
   - `adityajn105/flickr8k`
   - `incorpes/glove6b200d`

### Quick Setup
```python
# Cell 1: Install & clone
!pip install -q pyyaml tqdm nltk
!git clone https://github.com/DemonicAK/image-captioning-production.git
%cd image-captioning-production

# Cell 2: Setup Kaggle config
from training.kaggle_utils import create_kaggle_config, setup_kaggle_training

create_kaggle_config()  # Auto-detects Kaggle paths
env_info = setup_kaggle_training(verbose=True)

# Cell 3: Train
from training.train import TrainingPipeline

pipeline = TrainingPipeline("training/config.kaggle.yaml")
pipeline.run()

# Cell 4: Download outputs
import os
print("\n✓ Training complete! Files in /kaggle/working/:")
for f in sorted(os.listdir("/kaggle/working")):
    size = os.path.getsize(f"/kaggle/working/{f}") / 1e6
    print(f"  {f}: {size:.1f} MB")
```

### Kaggle Config Auto-Generation
```python
from training.kaggle_utils import create_kaggle_config

# Detects Kaggle paths and creates config.kaggle.yaml
create_kaggle_config(
    output_path="training/config.kaggle.yaml",
    flickr_dataset="adityajn105/flickr8k",  # Optional override
    glove_dataset="incorpes/glove6b200d",
)
```

### Performance on Kaggle
- Feature extraction: 30-45 minutes
- Training per epoch: 12-15 minutes  
- Total (20 epochs): 4-6 hours
- Memory: ~10GB GPU RAM (reduce batch_size if needed)

### Troubleshooting
**ImportError: No module named 'training'**
```bash
%cd image-captioning-production
```

**CUDA out of memory**
```yaml
# In config.kaggle.yaml
batch_size: 32  # Reduce from 64
```

**Timeout (9-hour limit)**
```yaml
# Reduce epochs
epochs: 10
```

---

## MLOps Runbook Quick Reference

### Training
```bash
python -m training.train --config training/config.yaml
tensorboard --logdir shared/artifacts/logs
```

### Testing
```bash
pytest                              # All tests
pytest tests/test_config.py -q      # Config only (fast)
pytest -m integration -q            # Integration tests
pytest -m "not slow" -q             # Skip slow tests
```

### Inference
```bash
# Local
cd inference && uvicorn main:app --host 0.0.0.0 --port 8000

# Docker
docker-compose up inference

# API call
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg" \
  -F "algorithm=beam"
```

---

## Architecture Patterns

### Design Patterns Used
1. **Registry Pattern** – Feature extractors, decoders
2. **Factory Pattern** – Callback creation
3. **Strategy Pattern** – Decoding algorithms
4. **Singleton Pattern** – Model registry (inference)
5. **Abstract Base Class** – Feature extractors, decoders
6. **Dependency Injection** – Trainer receives config/tracker

### Key Design Decisions
- **Dataclass configs** for type safety and validation
- **Lazy model loading** in inference (no import-time side effects)
- **Thread-safe singleton** model registry
- **Separation of concerns** – data/features/models/training/inference
- **Experiment tracking built-in** – no manual logging needed
- **Backward compatibility** – legacy API functions preserved

---

## FAQ

**Q: Can I use a different dataset?**
A: Yes, just update paths in `config.yaml`. Captions file should have format: `image_id.jpg,caption text`

**Q: How do I resume training?**
A: Load checkpoint: `model.load_weights("shared/artifacts/checkpoint.keras")` before calling `trainer.train()`

**Q: Can I train on CPU?**
A: Yes, but very slow. Set `use_mixed_precision: false` in config.

**Q: How do I deploy the model?**
A: Use Docker: `docker-compose up inference` or Kubernetes with manifests in `k8s/`

**Q: Where are training logs?**
A: `shared/artifacts/logs/` for TensorBoard, `shared/artifacts/metrics.csv` for raw metrics

**Q: How do I change the feature extractor?**
A: In `config.yaml`: set `feature_extractor: "InceptionV3"` and `feature_dim: 2048`, `image_size: [299, 299]`

**Q: Can I use my own word embeddings?**
A: Yes, modify `GloVeEmbeddings` class or provide custom embedding matrix to `build_caption_model()`

**Q: How do I add a new decoder algorithm?**
A: Extend `BaseDecoder` and register in `CaptionService.register_algorithm()`

---

## Performance Tips

1. **Enable mixed precision**: `use_mixed_precision: true` (2-3x speedup on modern GPUs)
2. **Increase batch size**: Limited by GPU memory, try 128 or 256
3. **Pre-extract features**: Run feature extraction once, reuse in multiple training runs
4. **Use persistent workers**: Set `num_parallel_calls=tf.data.AUTOTUNE` in dataset pipeline
5. **Profile with TensorBoard**: Analyze performance bottlenecks
6. **Reduce validation frequency**: Validate every N epochs instead of every epoch

---

## License

MIT License - See LICENSE file for details
