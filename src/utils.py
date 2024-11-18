# src/utils.py

import yaml
import json
import os
import logging
from typing import Any, Dict, List
import numpy as np
import random
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load the configuration from a YAML file.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        SystemExit: If the file is not found or parsing fails.
    """
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully from {file_path}")
        return config
    except FileNotFoundError as e:
        logging.error(f"Configuration file '{file_path}' not found. Exiting.")
        raise SystemExit(e)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file '{file_path}': {e}. Exiting.")
        raise SystemExit(e)


def validate_config(
    config: Dict[str, Any],
    required_keys: List[str] = ["paths", "data_splits", "yolo_splits", "model", "training", "augmentation", "extensions"]
) -> None:
    """
    Validate the configuration for required keys, structure, and value ranges.

    Args:
        config (Dict[str, Any]): Configuration dictionary to validate.
        required_keys (List[str]): List of top-level keys that must be present.

    Raises:
        ValueError: If required keys are missing or if the structure or values are invalid.
    """
    # Check for missing top-level keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Configuration validation failed. Missing keys: {missing_keys}")

    # Validate 'paths'
    paths_required = ["data", "train_data", "val_data", "test_data", "video", "performance", "results", "model_save"]
    for key in paths_required:
        if key not in config["paths"]:
            raise ValueError(f"Missing required path key: {key}")
        if key != "model_save" and not os.path.isdir(config["paths"][key]):
            raise ValueError(f"The path '{config['paths'][key]}' for '{key}' must be an existing directory.")
        if key == "model_save" and not os.path.isdir(os.path.dirname(config["paths"][key])):
            raise ValueError(f"The directory for '{key}' must exist: {os.path.dirname(config['paths'][key])}")
    logging.info("'paths' section validated. All required paths exist and are correct.")

    # Validate 'data_splits'
    data_splits = config["data_splits"]
    if not isinstance(data_splits, dict):
        raise ValueError("'data_splits' must be a dictionary.")
    if not all(split in ["train", "val", "test"] for split in data_splits):
        raise ValueError("'data_splits' can only contain 'train', 'val', and 'test'.")
    if len(set(data_splits.values())) != len(data_splits):
        raise ValueError("Split names in 'data_splits' must be unique.")
    if data_splits["train"] != config["paths"]["train_data"].split("/")[-1]:
        raise ValueError("'train' split must match the folder name in 'paths.train_data'.")
    if data_splits["val"] != config["paths"]["val_data"].split("/")[-1]:
        raise ValueError("'val' split must match the folder name in 'paths.val_data'.")
    if "test" in data_splits and data_splits["test"] != config["paths"]["test_data"].split("/")[-1]:
        raise ValueError("'test' split must match the folder name in 'paths.test_data', if defined.")
    logging.info("'data_splits' section validated. All splits are unique and correctly mapped.")

    # Validate 'yolo_splits'
    yolo_splits = config["yolo_splits"]
    if not isinstance(yolo_splits, dict) or set(yolo_splits.keys()) != {"images", "labels"}:
        raise ValueError("'yolo_splits' must contain only 'images' and 'labels' keys.")
    for key, value in yolo_splits.items():
        if not isinstance(value, str) or not value.isalnum():
            raise ValueError(f"Invalid value for 'yolo_splits.{key}': Must be a valid alphanumeric string.")
    logging.info("'yolo_splits' section validated. All folder names are valid.")

    # Validate 'model'
    model = config["model"]
    if not isinstance(model.get("num_classes"), int) or model["num_classes"] <= 0:
        raise ValueError("'model.num_classes' must be a positive integer.")
    if not isinstance(model.get("input_shape"), list) or len(model["input_shape"]) != 3:
        raise ValueError("'model.input_shape' must be a list of three integers [height, width, channels].")
    if not all(isinstance(dim, int) and dim > 0 for dim in model["input_shape"]):
        raise ValueError("'model.input_shape' must contain positive integers.")
    if not isinstance(model.get("class_names"), list) or len(model["class_names"]) != model["num_classes"]:
        raise ValueError("'model.class_names' must be a list of strings matching 'model.num_classes'.")
    if not all(isinstance(name, str) and name.isprintable() for name in model["class_names"]):
        raise ValueError("'model.class_names' must be a list of valid, printable strings.")
    if not isinstance(model.get("unclassified"), str) or not model["unclassified"].isalnum():
        raise ValueError("'model.unclassified' must be a valid alphanumeric string.")
    logging.info("'model' section validated. All model parameters are correct.")

    # Validate 'training'
    training = config["training"]
    for key in ["train_batch_size", "val_batch_size", "test_batch_size", "num_epochs"]:
        if not isinstance(training.get(key), int) or training[key] <= 0:
            raise ValueError(f"'training.{key}' must be a positive integer.")
    if not isinstance(training.get("learning_rate"), float) or not (0 < training["learning_rate"] <= 1):
        raise ValueError("'training.learning_rate' must be a positive float between 0 and 1.")
    logging.info("'training' section validated. Training parameters are within valid ranges.")

    # Validate 'augmentation'
    augmentation = config["augmentation"]
    if not (isinstance(augmentation.get("rotation_range"), (int, float)) and 0 <= augmentation["rotation_range"] <= 360):
        raise ValueError("'augmentation.rotation_range' must be a number between 0 and 360.")
    for key in ["width_shift_range", "height_shift_range", "shear_range", "zoom_range"]:
        if not (isinstance(augmentation.get(key), (int, float)) and 0 <= augmentation[key] <= 1):
            raise ValueError(f"'augmentation.{key}' must be a number between 0 and 1.")
    if not isinstance(augmentation.get("horizontal_flip"), bool):
        raise ValueError("'augmentation.horizontal_flip' must be a boolean.")
    if not isinstance(augmentation.get("fill_mode"), str) or augmentation["fill_mode"] not in ["nearest", "reflect", "wrap", "constant"]:
        raise ValueError("'augmentation.fill_mode' must be one of 'nearest', 'reflect', 'wrap', or 'constant'.")
    logging.info("'augmentation' section validated. Augmentation parameters are valid.")

    # Validate 'extensions'
    extensions = config["extensions"]
    valid_extensions = {".jpg", ".png"}
    if not isinstance(extensions.get("image_extensions"), list) or not set(extensions["image_extensions"]).issubset(valid_extensions):
        raise ValueError("'extensions.image_extensions' must be a list containing only '.jpg' and/or '.png'.")
    logging.info("'extensions' section validated. Image extensions are supported.")

    logging.info("All configuration sections validated successfully.")


def get_config_section(config: Dict[str, Any], section: str) -> Any:
    """
    Retrieve a specific section from the configuration, searching recursively if necessary.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        section (str): Section to retrieve. Can be a nested key using dot notation (e.g., "paths.data").

    Returns:
        Any: The requested section or value.

    Raises:
        KeyError: If the section does not exist in the configuration.
    """
    keys = section.split(".")  # Support dot notation for nested keys
    current = config

    for key in keys:
        if key in current:
            current = current[key]
        else:
            raise KeyError(f"Missing configuration section: '{section}' (stopped at '{key}')")

    return current


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    logging.info(f"Random seed set to {seed}")


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary to a JSON file. Usage: Storing Model Metadata, predictions...

    Args:
        data (Dict[str, Any]): Data to save.
        file_path (str): Path to the JSON file.

    Raises:
        ValueError: If saving the file fails.
    """
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
        logging.info(f"Data saved to JSON file: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to save JSON file at {file_path}: {e}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Loaded data.

    Raises:
        ValueError: If loading the file fails.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logging.info(f"Data loaded from JSON file: {file_path}")
        return data
    except Exception as e:
        raise ValueError(f"Failed to load JSON file at {file_path}: {e}")


if __name__ == "__main__":
    try:
        # Load the configuration
        config = load_config()
        # Validate the configuration
        validate_config(config)
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        raise SystemExit(e)