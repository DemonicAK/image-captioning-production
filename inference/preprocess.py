import numpy as np
import tensorflow as tf
from inference.model_loader import feature_extractor
from tensorflow.keras.applications.efficientnet import preprocess_input

class ImageFeatureEncoder:
    def __init__(
        self,
        feature_extractor,
        image,
        image_size=(300, 300),
    ):
        self.feature_extractor = feature_extractor
        self.image = image
        self.image_size = image_size

    def _load_and_preprocess_image(self, image):
        image = image.resize(self.image_size)
        image = np.array(image, dtype=np.float32)
        image = preprocess_input(image)
        # Add batch dim
        image = tf.expand_dims(image, axis=0) # (1, 300, 300, 3)
        return image


    def encode(self, image):
        image = self._load_and_preprocess_image(image)
        features = self.feature_extractor(
            image,
            training=False
        )
        return features

def image_preprocessor(image):
    encoder = ImageFeatureEncoder(
        feature_extractor=feature_extractor,
        image=image
    )
    features = encoder.encode(image)
    return features