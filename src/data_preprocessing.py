# src/data_preprocessing.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

# Load configurations from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def create_data_generators():
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=config["augmentation"]["rotation_range"],
        width_shift_range=config["augmentation"]["width_shift_range"],
        height_shift_range=config["augmentation"]["height_shift_range"],
        shear_range=config["augmentation"]["shear_range"],
        zoom_range=config["augmentation"]["zoom_range"],
        horizontal_flip=config["augmentation"]["horizontal_flip"],
        fill_mode=config["augmentation"]["fill_mode"],
        rescale=1.0/255
    )

    # Data generators for training, validation, and test sets
    train_generator = train_datagen.flow_from_directory(
        config["train_data_dir"],
        target_size=config["input_shape"][:2],
        batch_size=config["train_batch_size"],
        class_mode='categorical'
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)
    val_generator = val_datagen.flow_from_directory(
        config["val_data_dir"],
        target_size=config["input_shape"][:2],
        batch_size=config["val_batch_size"],
        class_mode='categorical'
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_datagen.flow_from_directory(
        config["test_data_dir"],
        target_size=config["input_shape"][:2],
        batch_size=config["test_batch_size"],
        class_mode='categorical'
    )

    return train_generator, val_generator, test_generator
