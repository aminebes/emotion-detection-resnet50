import os
import shutil

# Define paths and class names
base_path = "../data"  # Base path to the dataset
splits = ["train", "val", "test"]  # Dataset splits
class_names = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]  # Emotion classes

# Create class-specific folders if they don't exist
for split in splits:
    for class_name in class_names:
        # Create directories for each class within each dataset split
        os.makedirs(os.path.join(base_path, split, class_name), exist_ok=True)

def organize_images(split):
    """
    Organizes images into class-specific folders based on YOLO label files.
    
    Args:
        split (str): The dataset split ('train', 'val', or 'test') to organize.
    """
    # Paths to the images and labels directories for the given split
    images_path = os.path.join(base_path, split, "images")
    labels_path = os.path.join(base_path, split, "labels")
    
    # Loop through each label file in the labels directory
    for label_file in os.listdir(labels_path):
        # Read the class ID from the label file (first line, first item)
        with open(os.path.join(labels_path, label_file), "r") as file:
            class_id = int(file.readline().split()[0])

        # Determine the class name using the class ID
        class_name = class_names[class_id]

        # Find the corresponding image file (assumes .jpg format)
        image_file = label_file.replace(".txt", ".jpg")
        
        # Paths for the source image and the destination in the class-specific folder
        src_image_path = os.path.join(images_path, image_file)
        dst_image_path = os.path.join(base_path, split, class_name, image_file)
        
        # Copy the image to the appropriate class folder
        shutil.copy(src_image_path, dst_image_path)

def main():
    """
    Organize images into class-specific folders for each split.
    """
    for split in splits:
        print(f"Organizing images for the '{split}' split...")
        organize_images(split)
    print("Dataset organization complete.")

# Run the main function
if __name__ == "__main__":
    main()
