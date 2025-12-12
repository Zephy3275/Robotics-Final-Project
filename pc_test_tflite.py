import tensorflow as tf
import numpy as np
import cv2
import time

# --- CONFIGURATION ---
TFLITE_MODEL_PATH = 'robot_model_f16(test).tflite'   # Your TFLite model file
LABELS = ["Book", "Newspaper", "Old school bag", "Zip-top can"]
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.90 


# --- A. LOAD TFLITE MODEL ---
try:
    print("Loading TFLite Model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite model loaded successfully.")

except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit()


# --- B. OBJECT-TO-ACTION MAPPING FUNCTION ---
def determine_robot_action(detected_object_label, confidence):

    if confidence < CONFIDENCE_THRESHOLD:
        return "SAFE: Stop (Confidence too low)"

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
            if not ret:
                break

            # Preprocess image
            input_data = cv2.resize(frame, IMG_SIZE).astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)

            # Set input to interpreter
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get predictions
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

            prediction_index = np.argmax(predictions)
            confidence = predictions[prediction_index]
            detected_object = LABELS[prediction_index]

            action_string = determine_robot_action(detected_object, confidence)

            pred_text = f"Pred: {detected_object} ({confidence*100:.2f}%)"
            action_text = f"Action: {action_string}"

            color = (0, 255, 0) if action_string.startswith("ACTION") else (0, 165, 255)

            cv2.putText(frame, pred_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, action_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            print(f"Highest Prediction: {pred_text} | {action_text}")

            cv2.imshow('PC Model Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred during the loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run_pc_test()
