# src/inference.py

import cv2
import numpy as np
import tensorflow as tf
import argparse

# Define emotion classes
CLASS_NAMES = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def preprocess_frame(frame, target_size=(96, 96)):
    """
    Preprocess a frame for the model.
    Args:
        frame: The frame to preprocess (numpy array).
        target_size: The target size for the model input.
    Returns:
        Preprocessed frame (numpy array).
    """
    frame = cv2.resize(frame, target_size)
    frame = frame.astype("float32") / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

def predict_emotion(frame, model):
    """
    Predict emotion from a frame using the loaded model.
    Args:
        frame: The input frame (numpy array).
        model: The pre-trained model.
    Returns:
        Predicted emotion (str) and confidence (float).
    """
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    emotion = CLASS_NAMES[class_idx]
    return emotion, confidence

def run_inference_on_video(video_path, model):
    """
    Perform inference on a video file.
    Args:
        video_path: Path to the video file.
        model: The pre-trained model.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Output video file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("test_out.avi", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        emotion, confidence = predict_emotion(frame, model)

        # Display the emotion on the frame
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        # Display frame with predictions
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def run_inference_on_webcam(model):
    """
    Perform inference on webcam feed.
    Args:
        model: The pre-trained model.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        emotion, confidence = predict_emotion(frame, model)

        # Display the emotion on the frame
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame with predictions
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Detection Inference")
    parser.add_argument("-v", "--video", type=str, help="Path to video file for inference.")
    parser.add_argument("-c", "--camera", action="store_true", help="Use webcam for inference.")
    args = parser.parse_args()

    # Load pre-trained model
    model_path = "./models/emotion_detection_model.h5"
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Perform inference
    if args.camera:
        print("Starting inference on webcam...")
        run_inference_on_webcam(model)
    elif args.video:
        print(f"Starting inference on video: {args.video}")
        run_inference_on_video(args.video, model)
    else:
        print("Please specify either --video or --camera.")
