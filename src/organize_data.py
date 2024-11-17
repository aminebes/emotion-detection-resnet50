# src/organize_data.py

import os
import shutil

# Define constants
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))  # Base path to the dataset
SPLITS = ["train", "val", "test"]  # Dataset splits
CLASS_NAMES = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]  # Emotion classes
IMAGE_EXTENSIONS = [".jpg", ".png"]  # Supported image extensions


def create_class_folders():
    """
    Creates class-specific folders for each split if they do not exist.
    """
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            folder_path = os.path.join(BASE_PATH, split, class_name)
            try:
                os.makedirs(folder_path, exist_ok=True)
            except Exception as e:
                print(f"Error creating folder '{folder_path}': {e}")
    print("Class-specific folders created for all splits.")


def find_corresponding_image(label_file, images_path):
    """
    Finds the corresponding image file for a given label file.

    Args:
        label_file (str): The label file name (e.g., 'image00000.txt').
        images_path (str): Path to the images folder.

    Returns:
        str: Full path to the image file if found, or None if not found.
    """
    base_name = os.path.splitext(label_file)[0]  # Remove the .txt extension
    for ext in IMAGE_EXTENSIONS:
        image_path = os.path.join(images_path, f"{base_name}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None


def organize_images_by_class(split):
    """
    Organizes images into class-specific folders based on YOLO label files for a given split.

    Args:
        split (str): The dataset split ('train', 'val', or 'test') to organize.
    """
    images_path = os.path.join(BASE_PATH, split, "images")
    labels_path = os.path.join(BASE_PATH, split, "labels")

    # Ensure source folders exist
    if not os.path.exists(images_path):
        print(f"Error: Missing 'images' folder for '{split}' split. Skipping.")
        return
    if not os.path.exists(labels_path):
        print(f"Error: Missing 'labels' folder for '{split}' split. Skipping.")
        return

    for label_file in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, label_file)

        # Read the class ID from the label file
        with open(label_file_path, "r") as file:
            first_line = file.readline().strip()
            if not first_line:
                print(f"Warning: Empty label file '{label_file}'. Skipping.")
                continue
            try:
                class_id = int(first_line.split()[0])
            except ValueError:
                print(f"Warning: Invalid format in label file '{label_file}'. Skipping.")
                continue

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            print(f"Warning: Invalid class ID '{class_id}' in file '{label_file}'. Skipping.")
            continue

        class_name = CLASS_NAMES[class_id]
        image_path = find_corresponding_image(label_file, images_path)

        if image_path:
            dst_image_path = os.path.join(BASE_PATH, split, class_name, os.path.basename(image_path))
            try:
                shutil.move(image_path, dst_image_path)
                os.remove(label_file_path)  # Remove the label file after moving the image
            except Exception as e:
                print(f"Error removing label file and moving image '{image_path}' to '{dst_image_path}': {e}")
        else:
            print(f"Warning: No corresponding image found for label '{label_file}'. Skipping.")

    print(f"Images for the '{split}' split have been organized.")


def cleanup_empty_folders_and_labels(split):
    """
    Cleans up the images and labels folders for the given split.
    - Removes the images folder if empty.
    - Removes the redundant labels folder.
    - If the images folder isn't empty, create an "Unclassified" folder and move the remaining images and labels there.

    Args:
        split (str): The dataset split ('train', 'val', or 'test') to clean up.
    """
    images_path = os.path.join(BASE_PATH, split, "images")
    labels_path = os.path.join(BASE_PATH, split, "labels")
    unclassified_path = os.path.join(BASE_PATH, split, "Unclassified")

    try:
        # Check if the images folder is empty
        if os.path.exists(images_path) and not os.listdir(images_path):
            os.rmdir(images_path) # Removes the images folder if empty.
            print(f"Removed empty images folder: {images_path}")
            shutil.rmtree(labels_path) # Removes the redundant labels folder.
            print(f"Removed redundant labels folder: {labels_path}")

        else:
            os.makedirs(unclassified_path, exist_ok=True)
            if os.path.exists(images_path):
                shutil.move(images_path, unclassified_path)
                print(f"Moved remaining images to '{unclassified_path}/images'.")
            if os.path.exists(labels_path):
                shutil.move(labels_path, unclassified_path)
                print(f"Moved remaining labels to '{unclassified_path}/labels'.")

    except Exception as e:
        print(f"Error during cleanup for split '{split}': {e}")


def organize_dataset():
    """
    Organize images for all dataset splits.
    """
    try:
        print("Starting dataset organization...")
        create_class_folders()
        for split in SPLITS:
            print(f"Processing split: {split}")
            organize_images_by_class(split)
            cleanup_empty_folders_and_labels(split)
        print("Dataset organization complete.")
    except Exception as e:
        print(f"Error during dataset organization: {e}")
        raise


if __name__ == "__main__":
    organize_dataset()
