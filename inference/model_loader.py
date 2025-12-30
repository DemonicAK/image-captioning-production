from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
import json

# FEATURE_DIM=1536
max_length = 38

feature_extractor = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
feature_extractor.trainable = False 

supermodel = load_model("/app/artifacts/image_caption_model_final.keras",  compile=False)

with open("/app/artifacts/wordtoix.json") as f:
    wordtoix = json.load(f)

with open("/app/artifacts/ixtoword.json") as f:
    ixtoword = {int(k): v for k, v in  json.load(f).items()}

print(len(wordtoix), len(ixtoword))