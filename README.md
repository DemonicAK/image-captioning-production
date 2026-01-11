# Image Captioning Production

Production-grade image captioning system with attention-based encoder-decoder architecture (TensorFlow).

## Overview

- **Training**: Attention-based caption model with EfficientNetB3/InceptionV3 feature extraction, experiment tracking, TensorBoard logging, and model versioning
- **Inference**: FastAPI REST API with lazy model loading, health/metrics endpoints, beam search & greedy decoding
- **Testing**: Unit, integration, and performance tests with pytest markers
- **MLOps**: Automated experiment tracking, model versioning with SHA256 integrity checks, deployment manifest generation

## Quick Start

```bash
# Training (outputs to shared/artifacts/)
python -m training.train --config training/config.yaml

# Inference API
docker-compose up inference
# OR: cd inference && uvicorn main:app --host 0.0.0.0 --port 8000

# Generate caption
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "algorithm=beam"
```

## Key Features

### Training Pipeline
- Dataclass-based config with validation
- Automated experiment tracking (metrics, configs, git info, TensorBoard)
- Feature extractor registry (EfficientNet/Inception)
- Mixed precision training, early stopping, LR scheduling
- Kaggle notebook compatible

### Inference Service
- Thread-safe model registry with lazy loading
- `/predict`, `/health`, `/ready`, `/metrics` endpoints
- Greedy and beam search decoding
- Request logging with timing

### MLOps
- **Experiment Tracking**: Automated logging of configs, metrics, BLEU scores, git state, environment info → `run_summary.json`, `metrics.csv`, TensorBoard logs
- **Testing**: Pytest with markers (`slow`, `integration`, `performance`), coverage reports

## Project Structure

```
image-captioning-production/
├── training/          # Training pipeline (configs, data, models, trainers, utils)
├── inference/         # FastAPI service (main, registry, caption_generator, preprocessing)
├── tests/             # Test suite (unit, integration, performance)
└── shared/artifacts/  # Model artifacts (models, tokenizers, logs, metrics)
```

## Documentation

**For developers & contributors:** See [DOCS.md](DOCS.md) for:
- Architecture deep-dive
- MLOps runbook (training, tracking, testing)
- Configuration options
- Extending the pipeline
- Kaggle setup

## Requirements

- Python ≥ 3.9
- TensorFlow ≥ 2.10
- FastAPI, NumPy, PyYAML, tqdm, NLTK

## License

MIT
