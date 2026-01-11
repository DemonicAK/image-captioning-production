# Kaggle Setup Guide

## Quick Answer

**Yes, you can run this on Kaggle!** The code is fully compatible after cloning. Just follow the steps below.

## Prerequisites

1. **Kaggle Account** with notebook access
2. **GPU** enabled in notebook settings (required)
3. **Add these datasets** to your notebook as input sources:
   - `adityajn105/flickr8k`
   - `incorpes/glove6b200d`

## Step-by-Step Setup

### Step 1: Clone Repository

```bash
# In first cell of Kaggle notebook
!git clone https://github.com/DemonicAK/image-captioning-production.git
%cd image-captioning-production
```

### Step 2: Install Dependencies

```bash
!pip install -q pyyaml tqdm nltk
```

(TensorFlow and NumPy are pre-installed on Kaggle)

### Step 3: Create Kaggle Configuration

```python
from training.kaggle_utils import create_kaggle_config, setup_kaggle_training

# Auto-detect Kaggle environment and create config
create_kaggle_config()

# Verify setup
env_info = setup_kaggle_training(verbose=True)
```

### Step 4: Run Training

```python
from training.train import TrainingPipeline

pipeline = TrainingPipeline("training/config.kaggle.yaml")
pipeline.run()
```

## Complete Kaggle Notebook Example

```python
# Cell 1: Install & Setup
!pip install -q pyyaml tqdm nltk
!git clone https://github.com/DemonicAK/image-captioning-production.git
%cd image-captioning-production

# Cell 2: Create Kaggle config
from training.kaggle_utils import create_kaggle_config, setup_kaggle_training

# Auto-configures for Kaggle paths
create_kaggle_config()
env_info = setup_kaggle_training(verbose=True)

# Cell 3: Train
from training.train import TrainingPipeline

pipeline = TrainingPipeline("training/config.kaggle.yaml")
pipeline.run()

# Cell 4: Check outputs
import os
print("\n✓ Training Complete! Generated files:")
for f in sorted(os.listdir("/kaggle/working")):
    size = os.path.getsize(f"/kaggle/working/{f}") / 1e6
    print(f"  {f}: {size:.1f} MB")
```

## Output Files

After training, all artifacts are saved in `/kaggle/working/`:

| File | Purpose |
|------|---------|
| `image_caption_model_final.keras` | Trained model (download to use later) |
| `checkpoint.keras` | Best model checkpoint |
| `wordtoix.json` | Word→Index vocabulary mapping |
| `ixtoword.json` | Index→Word vocabulary mapping |
| `features.npy` | Pre-extracted image features |
| `logs/` | TensorBoard training logs |

## Customization

### Change Training Parameters

Edit `training/config.kaggle.yaml`:

```yaml
batch_size: 32          # Reduce if running out of memory
epochs: 10              # Fewer epochs for faster testing
learning_rate: 0.0001   # Adjust as needed
```

### Use Different Datasets

```python
from training.kaggle_utils import create_kaggle_config

# Example: Use different Flickr dataset
create_kaggle_config(
    output_path="training/config.kaggle.yaml",
    flickr_dataset="your-dataset-name",
    glove_dataset="another-glove-dataset"
)
```

### Memory Optimization

If running out of memory:

```python
# Reduce batch size in config.kaggle.yaml
batch_size: 32

# Or manually in notebook
from training.kaggle_utils import KaggleNotebookHelper
helper = KaggleNotebookHelper()
helper.setup_memory()  # Enable GPU memory growth
```

## Troubleshooting

### ImportError: No module named 'training'

Make sure you're in the correct directory:
```bash
%cd image-captioning-production
```

### FileNotFoundError: Dataset not found

Verify datasets are added to notebook:
1. Click "Input" in notebook sidebar
2. Check if `adityajn105/flickr8k` and `incorpes/glove6b200d` are listed
3. If not, add them via "Input" → "Search datasets"

### CUDA out of memory

Reduce `batch_size` in `config.kaggle.yaml`:
```yaml
batch_size: 32  # Default is 64
```

### Timeout (9 hour limit)

Kaggle notebooks have a 9-hour runtime limit. This should be enough for 20 epochs on a GPU. If you timeout:
- Reduce `epochs` in config
- Save checkpoint and resume training (manual)

## Performance Expectations

On Kaggle GPU (P100 or K80):

| Metric | Expected Time |
|--------|----------------|
| Feature extraction | 30-45 minutes |
| Training per epoch | 12-15 minutes |
| Total (20 epochs) | 4-6 hours |

## Next Steps

After training on Kaggle:

1. **Download the model**: 
   - Right-click on files in `/kaggle/working/` → Download

2. **Use in inference** (inference/ folder):
   ```python
   from inference.predict import ImageCaptioner
   
   captioner = ImageCaptioner("path/to/model.keras")
   caption = captioner.caption("path/to/image.jpg")
   ```

3. **Deploy to Kubernetes**:
   - See k8s/ folder for deployment configs

## Have Issues?

Check these common fixes:

- ✓ GPU enabled? (Settings → Accelerator → GPU)
- ✓ Datasets added? (Input → Search datasets)
- ✓ In correct directory? (`%cd image-captioning-production`)
- ✓ Dependencies installed? (`!pip install -q pyyaml tqdm nltk`)
