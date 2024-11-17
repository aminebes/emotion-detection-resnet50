# src/model.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import yaml

# Load configurations from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def build_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=config["input_shape"])
    x = Flatten()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dense(config["num_classes"], activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model
