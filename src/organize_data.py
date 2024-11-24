# src/organize_data.py

import os
import shutil
import logging
from config import ConfigManager
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Instantiate the ConfigManager with the default or specified configuration file
config = ConfigManager()
try:
    # Validate the configuration to ensure all sections are correct
    config.validate_config()
except Exception as e:
    # Log any validation errors that occur
    logging.error(f"Validation error: {e}")

# Extract configurations
BASE_PATH: str = config.get_config_section("paths.data")  # Base path to the dataset
IMAGES: str = config.get_config_section("yolo_splits.images")  # Name of the images folder
LABELS: str = config.get_config_section("yolo_splits.labels")  # Name of the labels folder
SPLITS: list[str] = list(config.get_config_section("data_splits").values())  # Dataset splits (train, val, test)
IMAGE_EXTENSIONS: list[str] = config.get_config_section("extensions.image_extensions")  # Supported image extensions
CLASS_NAMES: list[str] = config.get_config_section("model.class_names")  # Emotion classes
UNCLASSIFIED: str = config.get_config_section("model.unclassified")  # Unclassified class


def create_class_folders() -> None:
    """
    Creates class-specific folders for each split if they do not exist.

    Iterates over each dataset split (e.g., train, val, test) and each emotion class
    to create directories for organizing the dataset.
    """
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            folder_path = os.path.join(BASE_PATH, split, class_name)  # Construct folder path
            try:
                os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
            except Exception as e:
                logging.error(f"Error creating folder '{folder_path}': {e}")
    logging.info("Class-specific folders created for all splits.")


def find_corresponding_image(label_file: str, images_path: str) -> Optional[str]:
    """
    Finds the corresponding image file for a given label file.

    Args:
        label_file (str): The label file name (e.g., 'image00000.txt').
        images_path (str): Path to the images folder.

    Returns:
        Optional[str]: Full path to the image file if found, or None if not found.
    """
    base_name = os.path.splitext(label_file)[0]  # Remove the .txt extension to get the base name
    for ext in IMAGE_EXTENSIONS:  # Check each supported image extension
        image_path = os.path.join(images_path, f"{base_name}{ext}")
        if os.path.exists(image_path):  # Return the path if the image exists
            return image_path
    return None  # Return None if no matching image is found


def organize_images_by_class(split: str) -> None:
    """
    Organizes images into class-specific folders based on YOLO label files for a given split.

    Args:
        split (str): The dataset split ('train', 'val', or 'test') to organize.
    """
    images_path = os.path.join(BASE_PATH, split, IMAGES)  # Path to the images folder
    labels_path = os.path.join(BASE_PATH, split, LABELS)  # Path to the labels folder

    # Ensure source folders exist
    if not os.path.exists(images_path):
        logging.warning(f"Missing '{IMAGES}' folder for '{split}' split. Skipping.")
        return
    if not os.path.exists(labels_path):
        logging.warning(f"Missing '{LABELS}' folder for '{split}' split. Skipping.")
        return

    for label_file in os.listdir(labels_path):  # Iterate over all label files
        label_file_path = os.path.join(labels_path, label_file)  # Full path to the label file

        # Read the class ID from the label file
        try:
            with open(label_file_path, "r") as file:
                first_line = file.readline().strip()  # Read the first line of the label file
                if not first_line:  # Skip empty label files
                    logging.warning(f"Empty label file '{label_file}'. Skipping.")
                    continue
                class_id = int(first_line.split()[0])  # Extract the class ID
        except (ValueError, IndexError) as e:
            logging.warning(f"Invalid format in label file '{label_file}': {e}. Skipping.")
            continue

        # Validate class ID
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            logging.warning(f"Invalid class ID '{class_id}' in file '{label_file}'. Skipping.")
            continue

        class_name = CLASS_NAMES[class_id]  # Map class ID to class name
        image_path = find_corresponding_image(label_file, images_path)  # Find the corresponding image

        if image_path:  # If the image exists, move it to the class folder
            dst_image_path = os.path.join(BASE_PATH, split, class_name, os.path.basename(image_path))
            try:
                shutil.move(image_path, dst_image_path)  # Move the image
                os.remove(label_file_path)  # Remove the label file after moving the image
            except Exception as e:
                logging.error(f"Error moving image '{image_path}' to '{dst_image_path}': {e}")
        else:
            logging.warning(f"No corresponding image found for label '{label_file}'. Skipping.")

    logging.info(f"Images for the '{split}' split have been organized.")


def cleanup_empty_folders_and_labels(split: str) -> None:
    """
    Cleans up the images and labels folders for the given split.
    - Removes the images folder if empty.
    - Removes the redundant labels folder.
    - If the images folder isn't empty, create an "Unclassified" folder and move the remaining images and labels there.

    Args:
        split (str): The dataset split ('train', 'val', or 'test') to clean up.
    """
    images_path = os.path.join(BASE_PATH, split, IMAGES)  # Path to the images folder
    labels_path = os.path.join(BASE_PATH, split, LABELS)  # Path to the labels folder
    unclassified_path = os.path.join(BASE_PATH, split, UNCLASSIFIED)  # Path to the unclassified folder

    try:
        # Check if the images folder is empty
        if os.path.exists(images_path) and not os.listdir(images_path):
            os.rmdir(images_path)  # Remove the images folder if empty
            logging.info(f"Removed empty images folder: {images_path}")
            shutil.rmtree(labels_path)  # Remove the redundant labels folder
            logging.info(f"Removed redundant labels folder: {labels_path}")
        else:
            # Move remaining files to the "Unclassified" folder
            os.makedirs(unclassified_path, exist_ok=True)
            if os.path.exists(images_path):
                shutil.move(images_path, unclassified_path)
                logging.info(f"Moved remaining images to '{unclassified_path}/{IMAGES}'.")
            if os.path.exists(labels_path):
                shutil.move(labels_path, unclassified_path)
                logging.info(f"Moved remaining labels to '{unclassified_path}/{LABELS}'.")
    except Exception as e:
        logging.error(f"Error during cleanup for split '{split}': {e}")


def organize_dataset() -> None:
    """
    Organize images for all dataset splits.

    This function:
    - Creates class-specific folders.
    - Organizes images based on YOLO label files.
    - Cleans up empty folders and unclassified files.

    Raises:
        SystemExit: If an error occurs during dataset organization.
    """
    try:
        logging.info("Starting dataset organization...")
        create_class_folders()  # Create folders for all classes
        for split in SPLITS:
            logging.info(f"Processing split: {split}")
            organize_images_by_class(split)  # Organize images for the current split
            cleanup_empty_folders_and_labels(split)  # Clean up any leftover files
        logging.info("Dataset organization complete.")
    except Exception as e:
        logging.error(f"Error during dataset organization: {e}")
        raise SystemExit(e)


if __name__ == "__main__":
    organize_dataset()