# Emotion Detection using ResNet50

This project is a deep learning-based emotion detection model built using TensorFlow and ResNet50. It leverages the AffectNet dataset to classify human facial expressions into categories such as _happy_, _sad_, _angry_, and more.

---

## About the AffectNet Dataset

AffectNet is a large facial expression dataset with around **0.4 million images** manually labeled for the presence of **eight facial expressions**:

- Neutral
- Happy
- Angry
- Sad
- Fear
- Surprise
- Disgust
- Contempt

The dataset also provides information on the intensity of **valence** and **arousal** for each image. To accommodate common memory constraints, the resolution of all images was reduced to **96x96 pixels**, ensuring uniform size.

---

### Emotion Classes Mapping

In AffectNet, the eight emotion classes are mapped as follows:

| Class ID | Emotion  |
| -------- | -------- |
| 0        | Anger    |
| 1        | Contempt |
| 2        | Disgust  |
| 3        | Fear     |
| 4        | Happy    |
| 5        | Neutral  |
| 6        | Sad      |
| 7        | Surprise |

---

### Data Split

The dataset is split as follows:

- **70%** for training
- **20%** for validation
- **10%** for testing

This ensures a balanced distribution of data for model training and evaluation.

---

## Project Structure

```
emotion_detection_resnet50/
├── data/                          # Folder to store raw and processed data
│   ├── train/                     # Training images
│   ├── val/                       # Validation images
│   └── test/                      # Test images
├── models/                        # Folder to store saved models
│   └── emotion_detection_model.h5
├── src/                           # Source code folder
│   ├── __init__.py                # Marks src as a package
│   ├── data_preprocessing.py      # Data preprocessing and augmentation functions
│   ├── model.py                   # Model definition and ResNet50 setup
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Model evaluation script
│   ├── predict.py                 # Script for making predictions with the trained model
│   └── organize_data.py           # Script to organize the data into class folders
├── notebooks/                     # Jupyter notebooks for exploration and testing
│   └── data_analysis.ipynb
├── config.yaml                    # Configurations for training (paths, hyperparameters, etc.)
├── LICENSE                        # License file for the project
├── requirements.txt               # Dependencies for the project
├── .gitignore                     # Git ignore file for ignoring unnecessary files
└── README.md                      # Project description and instructions
```

---

## Setup

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd emotion_detection_resnet50
```

---

### 2. Create and Activate a Virtual Environment

Create a virtual environment to manage project dependencies:

```bash
python -m venv venv
. venv/Scripts/activate  # Windows
source venv/bin/activate  # macOS/Linux
```

---

### 3. Install Dependencies

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 4. Download the Dataset

Download the AffectNet dataset in YOLO format from [Kaggle](https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format?select=YOLO_format). Extract the `YOLO_format/` folder into the root directory of the repository, rename it to `data/`, and remove `data.yaml`.

The extracted folder structure should look like this:

```
data/
├── train/
│   ├── images/
│   ├── labels/
├── val/
│   ├── images/
│   ├── labels/
├── test/
    ├── images/
    ├── labels/
```

---

### 5. Organize the Data

Run the `organize_data.py` script to organize the images into class-specific folders:

```bash
python src/organize_data.py
```

After running the script, the `data/` folder will be reorganized into the following structure:

```
data/
├── train/
│   ├── anger/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── val/
│   ├── anger/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── anger/
    ├── contempt/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

---

### 6. Configure Settings

Edit `config.yaml` to specify hyperparameters and paths:

```yaml
data_path: "./data"
train_batch_size: 32
val_batch_size: 32
test_batch_size: 32
learning_rate: 0.0001
num_epochs: 20
model_save_path: "./models/emotion_detection_model.h5"
```

---

## Usage

### Training the Model

To train the ResNet50-based emotion detection model, run:

```bash
python src/train.py
```

---

### Evaluating the Model

Evaluate the trained model on the test set by running:

```bash
python src/evaluate.py
```

---

### Making Predictions

Use the `predict.py` script to make predictions on new images:

```bash
python src/predict.py --image_path path/to/image.jpg
```

---

## Files Description

- **data_preprocessing.py**: Functions for loading, augmenting, and preprocessing data.
- **model.py**: Contains the ResNet50 model architecture, with transfer learning and additional dense layers for emotion classification.
- **train.py**: Trains the model using the specified dataset and saves the best model.
- **evaluate.py**: Evaluates the model on the test set and outputs metrics.
- **predict.py**: Loads the trained model and makes predictions on input images.
- **organize_data.py**: Organizes images into class-specific folders based on YOLO-style label files.

---

## Dependencies

Install the following dependencies using `pip install -r requirements.txt`:

- TensorFlow
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn
- PyYAML
- tqdm

---

## Results

Include details of your training results, such as accuracy, loss, and any sample predictions, to document the model's performance.

---

## Troubleshooting

### Common Issues

- **TensorFlow Installation**: Ensure compatibility between your Python version and TensorFlow.
- **Virtual Environment**: If dependencies aren’t being recognized, ensure your virtual environment is activated.
