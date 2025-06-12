"""
Real-Time Sign Language Recognition

This script captures video from your webcam, detects a single hand using MediaPipe,
extracts a region of interest (ROI) around the hand, preprocesses it to match the
training input format (224×224 RGB), and feeds it into a fine-tuned ResNet152V2
model to classify sign-language gestures (digits 0–9 + a–z). It smooths predictions
over a short window for stability and overlays the predicted class and confidence
on the live video feed.

Requirements:
  - TensorFlow & Keras (with ResNet152V2)
  - OpenCV
  - MediaPipe
  - numpy

Usage:
  1. Place your pretrained model file `best_sign_lang_finetuned.h5` in the working dir. (Current version from method 1 can be used here, achieves 80% accuracy: https://mega.nz/file/obZygDzL#9L_vIn4CarApTnz0GeBqrQoqeYEmWIjLo77WD9CvZrE)
  2. Run the script. Press ‘q’ to quit, ‘r’ to reset smoothing.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.regularizers import l2

# === MediaPipe Hand-Detection Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # Continuous video stream
    max_num_hands=1,               # Only one hand at a time
    min_detection_confidence=0.7,  # Detection threshold
    min_tracking_confidence=0.5    # Tracking threshold
)
mp_draw = mp.solutions.drawing_utils

# === PARAMETERS & CLASS DEFINITIONS ===
MODEL_FILE = 'best_sign_lang_finetuned.h5'
IMG_SIZE = (224, 224)  # Model’s expected input dimensions
# Classes: digits 0–9, then letters a–z
class_names = list(map(str, range(10))) + list('abcdefghijklmnopqrstuvwxyz')
NUM_CLASSES = len(class_names)

# === VERIFY MODEL EXISTS ===
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {os.path.abspath(MODEL_FILE)}")

# === MODEL CREATION FUNCTION ===
def create_model(num_classes, input_shape):
    """
    Reconstruct the fine-tuned ResNet152V2 architecture:
      - Pretrained base (no top)
      - BatchNorm + Dense + Dropout layers matching training script
    """
    base = ResNet152V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    model = models.Sequential([
        base,
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# === LOAD OR RECONSTRUCT MODEL ===
print("Loading model...")
try:
    # Try loading full model
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    print(f"✓ Loaded complete model from {MODEL_FILE}")
except Exception as e:
    print(f"Complete load failed: {e}\nRecreating architecture and loading weights…")
    model = create_model(NUM_CLASSES, IMG_SIZE + (3,))
    model.load_weights(MODEL_FILE)
    print(f"✓ Recreated model and loaded weights from {MODEL_FILE}")

model.trainable = False  # Inference mode

# === IMAGE PREPROCESSING ===
def preprocess_image(roi):
    """
    - Resize to 224×224
    - Normalize pixel values to [0,1]
    - Add batch dimension
    """
    resized = cv2.resize(roi, IMG_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

# === PREDICTION SMOOTHING ===
class PredictionSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions = []

    def update(self, prediction):
        self.predictions.append(prediction)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        return np.mean(self.predictions, axis=0)

smoother = PredictionSmoother(window_size=3)

# === WEBCAM SETUP ===
print("Opening camera…")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam. Check device & permissions.")

# Improve capture quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("✓ Camera opened. Press 'q' to quit, 'r' to reset smoothing.")

# === MAIN LOOP ===
frame_count = 0
prediction_text = ""
confidence_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    h, w, _ = frame.shape

    # Detect hands
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Overlay instructions
    cv2.putText(frame, "Show your hand sign", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if results.multi_hand_landmarks:
        # Process first hand
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Compute bounding box
            pts = np.array([[int(lm.x*w), int(lm.y*h)] for lm in landmarks.landmark])
            x_min, y_min = np.min(pts, axis=0) - 40
            x_max, y_max = np.max(pts, axis=0) + 40
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size and roi.shape[0]>50 and roi.shape[1]>50:
                processed = preprocess_image(roi)
                if frame_count % 3 == 0:
                    preds = model.predict(processed, verbose=0)[0]
                    sm = smoother.update(preds)
                    top_i = np.argmax(sm)
                    conf = sm[top_i]
                    if conf > confidence_threshold:
                        prediction_text = f"{class_names[top_i].upper()} ({conf*100:.1f}%)"
                    else:
                        prediction_text = "Low confidence"

                # Draw prediction
                if prediction_text:
                    tw, th = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.rectangle(frame, (x_min, y_min-40), (x_min+tw+10, y_min-5), (0,255,0), -1)
                    cv2.putText(frame, prediction_text, (x_min+5, y_min-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

                    # Show small ROI for debugging
                    small = cv2.resize(roi, (100,100))
                    frame[10:110, w-110:w-10] = small
                    cv2.rectangle(frame, (w-110,10), (w-10,110), (255,255,255),2)
    else:
        if frame_count % 10 == 0:
            prediction_text = ""

    frame_count += 1
    if frame_count % 30 == 0:
        cv2.putText(frame, f"Frame: {frame_count}", (w-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow('Sign Language Recognition', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        smoother = PredictionSmoother(window_size=3)
        prediction_text = ""
        print("Smoothing reset")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")
