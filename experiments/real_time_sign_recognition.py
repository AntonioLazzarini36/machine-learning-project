'''
Test program that uses a pretrained TFLite model for real-time sign language recognition from webcam input
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import sys
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'model1.tflite')

IMG_SIZE = (128, 128)
INTERMEDIATE_RESOLUTION = (128, 128)

class_names = list(map(str, range(10))) + list('abcdefghijklmnopqrstuvwxyz')

def load_tflite_model(model_path):
    """Loads a TFLite model, allocates tensors, and returns the interpreter."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{os.path.abspath(model_path)}'")
        print(f"Please make sure '{MODEL_FILE}' is in the same directory as the script.")
        sys.exit(1)

    print(f"Loading TFLite model from '{model_path}'...")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("✓ TFLite model loaded successfully.")
        return interpreter
    except Exception as e:
        print(f"Failed to load TFLite model: {e}")
        sys.exit(1)

def preprocess_for_model(roi, input_details):
    """
    Preprocesses the Region of Interest (ROI) to match the model's input requirements.
    """
    # 1. Downscale to an intermediate size. Using INTER_AREA is good for shrinking.
    low_res_roi = cv2.resize(roi, INTERMEDIATE_RESOLUTION, interpolation=cv2.INTER_AREA)

    # 2. Convert to grayscale.
    gray_roi = cv2.cvtColor(low_res_roi, cv2.COLOR_BGR2GRAY)

    # 3. Upscale back to the model's expected input size. INTER_LINEAR gives a softer look.
    upscaled_roi = cv2.resize(gray_roi, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    # 4. Reshape for the model: (128, 128) -> (1, 128, 128, 1)
    #    The TFLite model expects a batch dimension.
    img_reshaped = np.reshape(upscaled_roi, (1, IMG_SIZE[0], IMG_SIZE[1], 1))

    # 5. Normalize and convert to the correct data type for the model.
    #    TFLite models are often quantized or expect a specific float type.
    input_type = input_details[0]['dtype']
    img_processed = img_reshaped.astype(input_type)
    if np.issubdtype(input_type, np.floating):
        img_processed = img_processed / 255.0

    # Create a displayable version of the image that's being fed to the model.
    displayable_image = cv2.cvtColor(upscaled_roi, cv2.COLOR_GRAY2BGR)

    return img_processed, displayable_image

def tflite_predict(interpreter, model_input, input_details, output_details):
    """Performs inference using the TFLite interpreter."""
    interpreter.set_tensor(input_details[0]['index'], model_input)
    
    # Run the inference.
    interpreter.invoke()
    
    # Get the results.
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

def main():
    """Main function to run the camera and real-time prediction."""
    interpreter = load_tflite_model(MODEL_FILE)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n--- Model Details ---")
    print(f"Input Shape: {input_details[0]['shape']}")
    print(f"Input Type: {input_details[0]['dtype']}")
    print(f"Output Shape: {output_details[0]['shape']}")
    print(f"Output Type: {output_details[0]['dtype']}")
    print("---------------------\n")


    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        sys.exit(1)
    print("✓ Camera opened. Press 'q' to quit.")

    prediction_text = ""
    confidence_threshold = 0.5 

    processed_view = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Segmentation and Hand Detection ---
            seg_results = segmentation.process(rgb_frame)
            condition = np.stack((seg_results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            segmented_frame = np.where(condition, frame, bg_image)
            hand_results = hands.process(rgb_frame)

            if hand_results.multi_hand_landmarks:
                for landmarks in hand_results.multi_hand_landmarks:
                    # Draw landmarks on the original frame
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Calculate bounding box for the hand
                    pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark])
                    x_min, y_min = np.min(pts, axis=0)
                    x_max, y_max = np.max(pts, axis=0)
                    
                    # Add padding to the bounding box
                    padding = 50
                    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                    # Extract the Region of Interest (ROI) from the segmented frame
                    roi = segmented_frame[y_min:y_max, x_min:x_max]

                    if roi.size > 0:
                        # Preprocess the ROI for the TFLite model
                        model_input, display_img = preprocess_for_model(roi, input_details)
                        processed_view = display_img
                        
                        # Get predictions from the TFLite model
                        preds = tflite_predict(interpreter, model_input, input_details, output_details)
                        
                        top_index = np.argmax(preds)
                        confidence = preds[top_index]

                        if confidence > confidence_threshold:
                            prediction_text = f"{class_names[top_index].upper()} ({confidence*100:.1f}%)"
                        else:
                            prediction_text = "..."

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, prediction_text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                 prediction_text = ""

            cv2.imshow('Live Sign Recognition', frame)
            cv2.imshow('Processed View for Model', processed_view)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == '__main__':
    main()
