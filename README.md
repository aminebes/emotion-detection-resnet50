# Emotion Detection using ResNet50

This project is a deep learning-based emotion detection model built using TensorFlow and ResNet50. It leverages the AffectNet dataset to classify human facial expressions into categories such as _happy_, _sad_, _angry_, and more.

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

To accommodate common memory constraints, the resolution of all images was reduced to **96x96 pixels**, ensuring uniform size.

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

### Data Split

The dataset is split as follows:

- **70%** for training
- **20%** for validation
- **10%** for testing

This ensures a balanced distribution of data for model training and evaluation.

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
│   ├── organize_data.py           # Script to organize the data into class folders
│   └── inference.py               # Script for running inference on video or camera
├── notebooks/                     # Jupyter notebooks for exploration and testing
│   └── data_analysis.ipynb
├── config.yaml                    # Configurations for training (paths, hyperparameters, etc.)
├── LICENSE                        # License file for the project
├── requirements.txt               # Dependencies for the project
├── .gitignore                     # Git ignore file for ignoring unnecessary files
└── README.md                      # Project description and instructions
```

## Setup

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd emotion_detection_resnet50
```

### 2. Create and Activate a Virtual Environment

Create a virtual environment to manage project dependencies:

```bash
python -m venv venv
. venv/Scripts/activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Download and Prepare the Dataset

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

### 5. Configure Settings

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

## Usage

### Training the Model

To train the ResNet50-based emotion detection model, run:

```bash
python src/train.py
```

### Evaluating the Model

Evaluate the trained model on the test set by running:

```bash
python src/evaluate.py
```

### Making Predictions

Use the `predict.py` script to make predictions on new images:

```bash
python src/predict.py --image_path path/to/image.jpg
```

## Files Description

- **data_preprocessing.py**: Functions for loading, augmenting, and preprocessing data.
- **model.py**: Contains the ResNet50 model architecture, with transfer learning and additional dense layers for emotion classification.
- **train.py**: Trains the model using the specified dataset and saves the best model.
- **evaluate.py**: Evaluates the model on the test set and outputs metrics.
- **predict.py**: Loads the trained model and makes predictions on input images.
- **organize_data.py**: Organizes images into class-specific folders based on YOLO-style label files.
- **inference.py**: Runs inference on video or camera.

## Inference

You can test model performance on a webcam or on a video file:

### Using a Webcam

Run the following command to test inference on your webcam:

```bash
python src/inference.py -c
```

### Using a Video File

Place your video file (`test.mp4`) in the `videos/` directory and run:

```bash
python src/inference.py -v
```

This will generate an output video (`test_out.avi`) with predictions and inference speed logged for each frame.

## Deployment

The model can be deployed on a webpage using the following steps.

### Webpage Deployment

You can use a lightweight JavaScript-based front-end framework to host the model for real-time predictions. Export the trained TensorFlow model to TensorFlow.js format using the `tensorflowjs_converter` tool.

Install TensorFlow.js:

```bash
pip install tensorflowjs
```

Convert the model:

```bash
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./models/emotion_detection_model ./web_model
```

Host the `web_model` directory on a webpage to enable real-time browser-based inference.

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
- tensorflowjs

## Performance and Results

Include details of your training results, such as accuracy, loss, and any sample predictions, to document the model's performance.

### **Performance and Results Directories**

To organize outputs and results generated during model training, evaluation, and inference, the project includes two dedicated directories:

#### **Performance Directory**

The `performance/` directory contains evaluation-related metrics and visualizations generated during the training and evaluation phases. Use this folder to analyze the model's effectiveness and tune its performance.

Example Contents:

- **`evaluation_metrics.txt`**: A text summary of key metrics, including accuracy, precision, recall, and F1-score.
- **`loss_accuracy_plot.png`**: A plot showing the model's training and validation loss/accuracy over epochs.
- **`confusion_matrix.png`**: A graphical representation of the confusion matrix, detailing the model's performance for each emotion class.
- **`detailed_report.csv`**: A CSV file providing per-class metrics, such as precision, recall, and F1-scores for all eight emotion categories.

#### **Results Directory**

The `results/` directory contains outputs generated during inference, such as predictions, processed videos, and logs. Use this folder to evaluate how the model performs on real-world data.

Example Contents:

- **`test_out.avi`**: A processed video file with emotion predictions overlayed on each frame.
- **`predictions.json`**: A JSON file containing predictions and confidence scores for a dataset or video.
- **`labeled_images/`**: A folder containing individual frames from a video or dataset with predicted emotions overlayed on each image.
  - Example: `frame_001.png`, `frame_002.png`, etc.
- **`inference_logs.txt`**: A log file documenting inference time and predictions for each frame.

### **How These Directories Are Used**

1. **Performance Directory**:

   - Stores training and evaluation results.
   - Analyze the data in this folder to:
     - Track training progress using loss/accuracy plots.
     - Understand model weaknesses using confusion matrices.
     - Evaluate per-class performance with detailed metrics.

2. **Results Directory**:
   - Stores outputs from inference runs.
   - Use these outputs to:
     - Verify the model’s predictions visually in labeled videos or images.
     - Share JSON predictions with downstream applications.
     - Examine logs to assess inference speed and accuracy.

### **How Outputs Are Generated**

1. **Performance Metrics**:

   - During training and evaluation, scripts like `train.py` and `evaluate.py` save visualizations and metrics directly to the `performance/` directory.

2. **Inference Results**:
   - The `inference.py` script saves processed outputs (e.g., labeled videos, JSON files, or frames) to the `results/` directory.

### **Folder Structure Example**

```
emotion_detection_resnet50/
├── performance/                # Evaluation-related metrics and visualizations
│   ├── evaluation_metrics.txt
│   ├── loss_accuracy_plot.png
│   ├── confusion_matrix.png
│   └── detailed_report.csv
├── results/                    # Outputs generated by the model
│   ├── test_out.avi
│   ├── predictions.json
│   ├── labeled_images/
│   │   ├── frame_001.png
│   │   └── frame_002.png
│   └── inference_logs.txt
```

## Troubleshooting

### Common Issues

- **TensorFlow Installation**: Ensure compatibility between your Python version and TensorFlow.
- **Virtual Environment**: If dependencies aren’t being recognized, ensure your virtual environment is activated.

## Authors

- **Mohamed Amine Bessrour**  
  Email: mbess081@uottawa.com  
  GitHub: [aminebes](https://github.com/aminebes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
