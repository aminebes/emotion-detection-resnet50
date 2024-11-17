# src/train.py

import tensorflow as tf
from data_preprocessing import create_data_generators
from model import build_model
import yaml

# Load configurations from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def train_model():
    train_generator, val_generator, _ = create_data_generators()
    model = build_model()
    
    # Early stopping and model checkpoint
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(filepath=config["model_save_path"], save_best_only=True)
    ]
    
    # Train the model
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config["num_epochs"],
        callbacks=callbacks
    )

if __name__ == "__main__":
    train_model()
