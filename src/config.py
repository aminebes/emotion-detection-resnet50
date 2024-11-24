# src/config.py

import yaml  # YAML parsing library
import os  # Library for interacting with the file system
import logging  # Logging library for structured logging
from typing import Any, Dict, List, Optional, Union  # Type annotations for better readability and validation

# Configure logging with a consistent format and INFO level by default
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Create a logger instance


class ConfigManager:
    """
    A utility class to manage configuration loading, validation, and retrieval.

    This class handles reading a YAML configuration file, validating its contents,
    and providing access to specific sections or values of the configuration.
    """

    def __init__(self, file_path: str = "config.yaml"):
        """
        Initialize the ConfigManager with a configuration file.

        Args:
            file_path (str): Path to the configuration file (default: "config.yaml").
        """
        self.file_path: str = file_path  # Store the path to the configuration file
        self.config: Dict[str, Any] = {}  # Placeholder for the loaded configuration data

        # Load the configuration file during initialization
        self.load_config()

    def load_config(self) -> None:
        """
        Load the configuration from a YAML file.

        This method reads the YAML file specified in the file_path, parses its
        contents into a dictionary, and stores it in self.config.

        Raises:
            SystemExit: If the file is not found, unreadable, or fails to parse.
        """
        if not os.path.isfile(self.file_path):
            # Check if the configuration file exists. Exit the program if not found.
            logger.error(f"Configuration file '{self.file_path}' does not exist. Exiting.")
            raise SystemExit(f"File not found: {self.file_path}")

        try:
            # Safely load YAML content into a Python dictionary
            with open(self.file_path, "r") as file:
                self.config = yaml.safe_load(file) or {}  # Default to an empty dict if file is empty
            logger.info(f"Configuration loaded successfully from '{self.file_path}'.")
        except yaml.YAMLError as e:
            # Handle YAML parsing errors
            logger.error(f"Error parsing YAML in '{self.file_path}': {e}. Exiting.")
            raise SystemExit(e)
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.error(f"Unexpected error loading configuration: {e}. Exiting.")
            raise SystemExit(e)

    def validate_config(self, required_keys: Optional[List[str]] = None) -> None:
        """
        Validate the configuration for required keys, structure, and value ranges.

        Args:
            required_keys (Optional[List[str]]): List of top-level keys that must be present.
                                                 Defaults to a common set of keys.

        Raises:
            ValueError: If required keys are missing or validation fails.
        """
        # Define default required keys if not provided
        required_keys = required_keys or [
            "paths", "data_splits", "yolo_splits", "model", "training", "augmentation", "extensions"
        ]

        # Check for missing required top-level keys in the configuration
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            logger.error(f"Configuration validation failed. Missing keys: {missing_keys}")
            raise ValueError(f"Missing keys: {missing_keys}")

        # Perform validation for each section of the configuration
        self._validate_paths()
        self._validate_data_splits()
        self._validate_yolo_splits()
        self._validate_model()
        self._validate_training()
        self._validate_augmentation()
        self._validate_extensions()

        logger.info("All configuration sections validated successfully.")

    def _validate_paths(self) -> None:
        """
        Validate the 'paths' section of the configuration.

        Raises:
            ValueError: If any required path is missing, invalid, or not accessible.
        """
        paths = self.config.get("paths", {})  # Retrieve the 'paths' section as a dictionary
        required_keys = [
            "data", "train_data", "val_data", "test_data", "video", "performance", "results", "model_save"
        ]  # List of required path keys

        for key in required_keys:
            value = paths.get(key)  # Get the value for each required key
            if not value:
                # Raise an error if a required key is missing
                raise ValueError(f"Missing required path key: '{key}'.")
            resolved_path = os.path.expandvars(value)  # Expand environment variables in the path (e.g., $HOME)
            if key != "model_save" and not os.path.isdir(resolved_path):
                # Validate that all paths except 'model_save' point to existing directories
                raise ValueError(f"The path '{resolved_path}' for '{key}' must be an existing directory.")
            if key == "model_save" and not os.path.isdir(os.path.dirname(resolved_path)):
                # Ensure the parent directory for 'model_save' exists
                raise ValueError(f"The directory for '{key}' must exist: {os.path.dirname(resolved_path)}.")

        logger.info("'paths' section validated. All required paths exist and are correct.")

    def _validate_data_splits(self) -> None:
        """
        Validate the 'data_splits' section of the configuration.

        This ensures that the splits (train, val, test) match their respective
        folder names in the 'paths' section.

        Raises:
            ValueError: If the structure or values in 'data_splits' are invalid.
        """
        data_splits = self.config.get("data_splits", {})  # Retrieve the 'data_splits' section
        paths = self.config.get("paths", {})  # Retrieve the 'paths' section

        if not isinstance(data_splits, dict):
            raise ValueError("'data_splits' must be a dictionary.")

        required_splits = {"train", "val", "test"}  # Expected keys in 'data_splits'
        if not required_splits.issubset(data_splits.keys()):
            raise ValueError(f"'data_splits' must contain these keys: {required_splits}")

        # Ensure each split value matches the folder name in the corresponding path
        if data_splits["train"] != os.path.basename(paths["train_data"]):
            raise ValueError(f"'train' split '{data_splits['train']}' does not match folder name in 'paths.train_data'.")
        if data_splits["val"] != os.path.basename(paths["val_data"]):
            raise ValueError(f"'val' split '{data_splits['val']}' does not match folder name in 'paths.val_data'.")
        if data_splits["test"] != os.path.basename(paths["test_data"]):
            raise ValueError(f"'test' split '{data_splits['test']}' does not match folder name in 'paths.test_data'.")

        logger.info("'data_splits' section validated and matches folder names in 'paths'.")

    def _validate_yolo_splits(self) -> None:
        """
        Validate the 'yolo_splits' section of the configuration.

        Ensures the presence of the required keys ('images', 'labels') and that
        their values are valid alphanumeric strings.

        Raises:
            ValueError: If the structure or values in 'yolo_splits' are invalid.
        """
        yolo_splits = self.config.get("yolo_splits", {})  # Retrieve the 'yolo_splits' section
        if not isinstance(yolo_splits, dict):
            raise ValueError("'yolo_splits' must be a dictionary.")

        required_keys = {"images", "labels"}  # Expected keys in 'yolo_splits'
        if not required_keys.issubset(yolo_splits.keys()):
            raise ValueError(f"'yolo_splits' must contain these keys: {required_keys}")

        # Ensure values are alphanumeric strings
        for key, value in yolo_splits.items():
            if not isinstance(value, str) or not value.isalnum():
                raise ValueError(f"Invalid value for 'yolo_splits.{key}': Must be a valid alphanumeric string.")

        logger.info("'yolo_splits' section validated successfully.")

    def _validate_model(self) -> None:
        """
        Validate the 'model' section of the configuration.

        Ensures the presence and validity of parameters like 'num_classes' and
        'input_shape'.

        Raises:
            ValueError: If required model parameters are invalid.
        """
        model = self.config.get("model", {})  # Retrieve the 'model' section
        if not isinstance(model, dict):
            raise ValueError("'model' section must be a dictionary.")

        num_classes = model.get("num_classes")  # Validate 'num_classes'
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("'model.num_classes' must be a positive integer.")

        input_shape = model.get("input_shape")  # Validate 'input_shape'
        if not (isinstance(input_shape, list) and len(input_shape) == 3 and all(isinstance(dim, int) for dim in input_shape)):
            raise ValueError("'model.input_shape' must be a list of three positive integers.")

        logger.info("'model' section validated.")

    def _validate_training(self) -> None:
        """
        Validate the 'training' section of the configuration.

        This method ensures that the training parameters (e.g., batch sizes, number of epochs,
        and learning rate) are within valid ranges and of the correct types.

        Raises:
            ValueError: If any training parameter is invalid or out of range.
        """
        training = self.config.get("training", {})  # Retrieve the 'training' section

        # Validate that batch sizes and number of epochs are positive integers
        for key in ["train_batch_size", "val_batch_size", "num_epochs"]:
            if not isinstance(training.get(key), int) or training[key] <= 0:
                raise ValueError(f"'training.{key}' must be a positive integer.")

        # Validate that learning rate is a float between 0 and 1
        learning_rate = training.get("learning_rate")
        if not isinstance(learning_rate, float) or not (0 < learning_rate <= 1):
            raise ValueError("'training.learning_rate' must be a float between 0 and 1.")

        logger.info("'training' section validated.")

    def _validate_augmentation(self) -> None:
        """
        Validate the 'augmentation' section of the configuration.

        This method ensures that the data augmentation parameters, such as rotation range,
        width/height shifts, shear, and zoom ranges, are within valid ranges and of correct types.

        Raises:
            ValueError: If any augmentation parameter is invalid or out of range.
        """
        augmentation = self.config.get("augmentation", {})  # Retrieve the 'augmentation' section

        # Validate that horizontal_flip is a boolean
        if not isinstance(augmentation.get("horizontal_flip"), bool):
            raise ValueError("'augmentation.horizontal_flip' must be a boolean.")

        # Add validation for other augmentation parameters if needed
        # Example: Validate ranges for rotation, shear, and zoom
        for key in ["rotation_range", "width_shift_range", "height_shift_range", "shear_range", "zoom_range"]:
            value = augmentation.get(key)
            if key in augmentation and not (isinstance(value, (int, float)) and value >= 0):
                raise ValueError(f"'augmentation.{key}' must be a non-negative number.")

        logger.info("'augmentation' section validated.")

    def _validate_extensions(self) -> None:
        """
        Validate the 'extensions' section of the configuration.

        This method ensures that the 'image_extensions' parameter is a list containing
        valid file extensions (e.g., '.jpg', '.png').

        Raises:
            ValueError: If 'image_extensions' is not a list or contains invalid entries.
        """
        extensions = self.config.get("extensions", {})  # Retrieve the 'extensions' section

        # Validate that 'image_extensions' is a list
        image_extensions = extensions.get("image_extensions")
        if not isinstance(image_extensions, list):
            raise ValueError("'extensions.image_extensions' must be a list.")

        # Ensure all extensions in the list are strings and start with '.'
        valid_extensions = {".jpg", ".png"}
        if not set(image_extensions).issubset(valid_extensions):
            raise ValueError("'extensions.image_extensions' must only contain valid extensions: .jpg, .png.")

        logger.info("'extensions' section validated.")

    def get_config_section(self, section: str) -> Union[Any, None]:
        """
        Retrieve a specific section or nested key from the configuration.

        This method supports dot notation for nested keys (e.g., "paths.data"), allowing
        users to easily access specific parts of the configuration.

        Args:
            section (str): The section or key to retrieve. Use dot notation for nested keys.

        Returns:
            Union[Any, None]: The requested section or value if found.

        Raises:
            KeyError: If the requested section or key does not exist in the configuration.
        """
        keys = section.split(".")  # Split the section into individual keys for navigation
        current = self.config  # Start at the top-level configuration dictionary

        for key in keys:
            current = current.get(key, None)  # Navigate to the next level using the current key
            if current is None:
                # Raise an error if the key does not exist at any level
                raise KeyError(f"Missing configuration section: '{section}' (stopped at '{key}')")

        return current  # Return the value or section found


if __name__ == "__main__":
    # Instantiate the ConfigManager with the default or specified configuration file
    config_manager = ConfigManager()

    try:
        # Validate the configuration to ensure all sections are correct
        config_manager.validate_config()
    except Exception as e:
        # Log any validation errors that occur
        logger.error(f"Validation error: {e}")
    
    # Get class names
    class_names = config_manager.get_config_section("model.class_names")
    logging.info(f"Class names: {class_names}")
