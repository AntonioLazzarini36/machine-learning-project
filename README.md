# sign-recognition-ml

This repository implements a sign language recognition system using machine learning. We explore two different methods for classifying hand gestures (digits 0–9 and letters a–z) and compare their performance.

---

## 🔍 Method 1: Transfer Learning with CNN

In **Method 1**, we leverage a pretrained ResNet152V2 model and fine‑tune it for sign language classification:

1. **Data Preprocessing**: Images are resized to 224×224 pixels and normalized to [0,1].
2. **Model Architecture**:
   - Base: ResNet152V2 (imagenet weights, without top layers)
   - Custom head: BatchNormalization → Dense(1024) → Dropout → Dense(512) → Dropout → Dense(256) → Dropout → Dense(36, softmax)
   - L2 regularization and dropout are used to reduce overfitting.
3. **Training & Evaluation**: The model achieves ~80% accuracy on the held-out test set.
4. **Outputs**:
   - Saved model file: `best_sign_lang_finetuned.h5`
   - Training curves and evaluation metrics saved in the notebook.

---

## 🔧 Method 2: Alternative Approach (TBD)

Details for the second classification strategy will be added soon.

---

## 🎥 Real-Time Sign Recognition Demo

We provide a script to run live inference using your webcam. It:

- Captures frames from the camera.
- Uses MediaPipe to detect and track a single hand.
- Extracts a bounding box around the hand and preprocesses the ROI.
- Feeds the ROI into the fine-tuned model, smooths predictions over a window for stability, and overlays the result on the video.

**Usage**:

1. Place or download the pretrained model (`best_sign_lang_finetuned.h5`).
2. Run the demo notebook or script:
   ```bash
   python realtime_sign_recognition.py
   ```

---

## 📥 Download Links

* **Pretrained Model (80% accuracy)** on MEGA:

  ```
  https://mega.nz/file/obZygDzL#9L_vIn4CarApTnz0GeBqrQoqeYEmWIjLo77WD9CvZrE
  ```

* **Dataset Archive** (`sign_lang_train.zip`) on MEGA:

  ```
  https://mega.nz/file/FLITlD7C#GbU0J0Yc3v9GLYKDm4os_a2-Ib9YTnBf5vwLSi4iC08
  ```


---
