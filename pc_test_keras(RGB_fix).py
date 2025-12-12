import tensorflow as tf
import numpy as np
import cv2
import time

# --- CONFIGURATION ---
KERAS_MODEL_PATH = 'robot_classifier(test).keras'
LABELS = ["Book", "Newspaper", "Old school bag", "Zip-top can"]
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.90

# --- A. LOAD MODEL ---
try:
    print("Loading Keras Model...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# --- B. OBJECT TO ACTION ---
def determine_robot_action(label, conf):
    if conf < CONFIDENCE_THRESHOLD:
        return "SAFE: Stop (Confidence too low)"

    if label == "Zip-top can":
        return "ACTION: Moving Forward"
    elif label == "Book":
        return "ACTION: Turning Left"
    elif label == "Newspaper":
        return "ACTION: Turning Right"
    elif label == "Old school bag":
        return "ACTION: Stop / Reverse"

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
            if not ret:
                break

            # 🔥 FIX: Convert BGR → RGB before prediction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize and prepare model input
            input_data = cv2.resize(rgb, IMG_SIZE).astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)

            # Run model
            predictions = model.predict(input_data, verbose=0)[0]

            idx = np.argmax(predictions)
            confidence = predictions[idx]
            detected_object = LABELS[idx]

            action = determine_robot_action(detected_object, confidence)

            # Draw text
            pred_text = f"Pred: {detected_object} ({confidence*100:.2f}%)"
            action_text = f"Action: {action}"

            color = (0, 255, 0) if action.startswith("ACTION") else (0, 165, 255)

            cv2.putText(frame, pred_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, action_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            print(f"{pred_text} | {action_text}")

            cv2.imshow("PC Model Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run_pc_test()
