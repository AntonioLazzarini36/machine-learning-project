# sign-recognition-ml

This repository implements a sign language recognition system using machine learning. We explore two different methods for classifying hand gestures (digits 0–9 and letters a–z) and compare their performance.

---

## 🔍 Method 1: Convolutional Neural Network (CNN)

In **Method 1**, we use a custom-built Convolutional Neural Network (CNN) for sign language classification. This method is runnable in Google Colab.

1.  **Data Preprocessing**: Images are resized to 128x128 pixels and normalized to a [0,1] range. Data augmentation techniques like random flips, brightness/contrast adjustments, and rotations are applied to the training set to improve model generalization.
2.  **Model Architecture**:
    * The CNN consists of four convolutional blocks with increasing filter sizes (32, 64, 128, 256), each followed by Batch Normalization and MaxPooling.
    * Dropout is used after the second and fourth blocks to prevent overfitting.
    * The convolutional base is connected to a dense layer of 512 units with L2 regularization, followed by a final softmax output layer for 36 classes (0-9, a-z).
3.  **Training & Evaluation**: The model is trained using the Adam optimizer and callbacks like `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping`. On notebook evaluation, the model achieves **~90% accuracy** on the held-out test set.
4.  **Outputs**:
    * The final trained model is quantized to reduce its size and saved in the `models/` folder.
    * Training curves and evaluation metrics are saved in the notebook.

---

## 🔧 Method 2: Random Forest

In **Method 2**, we use a Random Forest classifier for hand gesture classification.

1.  **Data Preprocessing**: The image data from the `SignLangDataset` is used for training the model.
2.  **Model Architecture**: A Random Forest model is trained on the image data.
3.  **Training & Evaluation**: On notebook evaluation, the model achieves **~70% accuracy**.
4.  **Outputs**:
    * The trained Random Forest model is saved using `joblib` and located in the `models/` folder.

---

## 🎥 Real-Time Sign Recognition Demo

We provide a script to run live inference using your webcam. This demo uses the quantized CNN model from **Method 1**. It:

-   Captures frames from the camera.
-   Uses MediaPipe to detect and track a single hand.
-   Extracts a bounding box around the hand and preprocesses the ROI.
-   Feeds the ROI into the model, smooths predictions over a window for stability, and overlays the result on the video.

Press `q` to quit the demo or `r` to reset prediction smoothing.