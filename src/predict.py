# src/predict.py

import tensorflow as tf
import numpy as np
import cv2
import yaml

# Load configurations from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, tuple(config["input_shape"][:2]))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_emotion(image_path):
    model = tf.keras.models.load_model(config["model_save_path"])
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    predict_emotion(image_path)
