import tensorflow as tf
import numpy as np
import cv2
import time

# --- CONFIGURATION ---
KERAS_MODEL_PATH = 'robot_classifier(test).keras'  # Your saved file
LABELS = ["Book", "Newspaper", "Old school bag", "Zip-top can"] 
IMG_SIZE = (224, 224)
# You should adjust this threshold until the random guesses are filtered out
CONFIDENCE_THRESHOLD = 0.90 

# --- A. LOAD MODEL ---
try:
    print("Loading Keras Model...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# --- B. OBJECT-TO-ACTION MAPPING FUNCTION (PROJECT LOGIC) ---
def determine_robot_action(detected_object_label, confidence):
    """Maps the detected object to a robot action based on the confidence threshold."""
    
    if confidence < CONFIDENCE_THRESHOLD:
        return "SAFE: Stop (Confidence too low)"

    # Action Logic (Reference: Robotic Personal Project 1.jpg)
    if detected_object_label == "Zip-top can":
        return "ACTION: Moving Forward"
    elif detected_object_label == "Book":
        return "ACTION: Turning Left"
    elif detected_object_label == "Newspaper":
        return "ACTION: Turning Right"
    elif detected_object_label == "Old school bag":
        return "ACTION: Stop / Reverse"
    else:
        return "SAFE: Stop (Unknown Object)"


# --- C. CAMERA LOOP ---
def run_pc_test():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            input_data = cv2.resize(frame, IMG_SIZE)
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

            predictions = model.predict(input_data, verbose=0)[0]
            
            prediction_index = np.argmax(predictions)
            confidence = predictions[prediction_index]
            detected_object = LABELS[prediction_index]
            
            # --- NEW: Get Action String ---
            action_string = determine_robot_action(detected_object, confidence)

            # 5. Display Result and Action on Frame
            pred_text = f"Pred: {detected_object} ({confidence*100:.2f}%)"
            action_text = f"Action: {action_string}"
            
            # Use color to indicate action status
            color = (0, 255, 0) if action_string.startswith("ACTION") else (0, 165, 255)
            
            cv2.putText(frame, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, action_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            print(f"Highest Prediction: {pred_text} | {action_text}")

            cv2.imshow('PC Model Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        print(f"An error occurred during the loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run_pc_test()