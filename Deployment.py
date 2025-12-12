# =========================================================================
# === STEP 1: LIBRARIES & CONFIGURATION (FINAL) ===
# =========================================================================
from auppbot import AUPPBot 
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time
import threading
from flask import Flask, Response, render_template_string

# --- CONFIGURATION ---
ROBOT_PORT = "/dev/ttyUSB0" 
ROBOT_BAUD = 115200 
ROBOT = AUPPBot(port=ROBOT_PORT, baud=ROBOT_BAUD, auto_safe=True)

# Speed for the AUPPBot system is 0-99
SPEED_0_99 = 20

TFLITE_MODEL_PATH = 'robot_model_f16.tflite'
LABELS = ["Book", "Newspaper", "Old school bag", "Zip-top can"]
IMG_SIZE = (224, 224) 
CONFIDENCE_THRESHOLD = 0.9

# Action Duration Settings
FORWARD_DURATION = 0.6  # seconds
TURN_DURATION = 0.6     # seconds
REVERSE_DURATION = 0.5  # seconds
COOLDOWN_TIME = 1.0     # seconds between actions

# --- GLOBAL VARIABLES for Web Streaming ---
output_frame = None
lock = threading.Lock()

# =========================================================================
# === STEP 2: ACTION MANAGER CLASS ===
# =========================================================================

class ActionManager:
    """Manages timed robot actions with cooldown periods"""
    
    def __init__(self, robot):
        self.robot = robot
        self.current_action = None
        self.last_executed_action = None
        self.is_executing = False
        self.in_cooldown = False
        self.action_start_time = 0
        self.cooldown_start_time = 0
        self.lock = threading.Lock()
        
        # Start action management thread
        self.running = True
        self.action_thread = threading.Thread(target=self._action_manager_loop, daemon=True)
        self.action_thread.start()
    
    def request_action(self, action_label):
        """
        Request a new action to be executed
        Args:
            action_label: String label from LABELS list
        """
        with self.lock:
            # Don't queue new actions if executing or in cooldown
            if self.is_executing or self.in_cooldown:
                return
            
            # Only trigger if action changed
            if action_label != self.current_action:
                self.current_action = action_label
                self.is_executing = True
                self.action_start_time = time.time()
                print(f"▶ ACTION: {action_label}")
                self.last_executed_action = action_label
    
    def _action_manager_loop(self):
        """Background thread to manage timed actions"""
        action_started = False
        
        while self.running:
            with self.lock:
                current_time = time.time()
                
                # Handle cooldown period
                if self.in_cooldown:
                    if current_time - self.cooldown_start_time >= COOLDOWN_TIME:
                        self.in_cooldown = False
                        self.current_action = None
                        action_started = False
                        print("✓ Cooldown complete - Ready for next action")
                
                # Handle executing action
                elif self.is_executing:
                    action = self.current_action
                    elapsed = current_time - self.action_start_time
                    
                    # Send motor command ONCE at the start of action
                    if not action_started:
                        print(f"🚗 Starting motor command for: {action}")
                        if action == "Zip-top can":
                            duration = FORWARD_DURATION
                            self._execute_forward()
                        elif action == "Book":
                            duration = TURN_DURATION
                            self._execute_turn_left()
                        elif action == "Newspaper":
                            duration = TURN_DURATION
                            self._execute_turn_right()
                        elif action == "Old school bag":
                            duration = REVERSE_DURATION
                            self._execute_reverse()
                        else:
                            duration = 0
                            self._execute_stop()
                        action_started = True
                    
                    # Determine duration (use stored value)
                    if action == "Zip-top can":
                        duration = FORWARD_DURATION
                    elif action in ["Book", "Newspaper"]:
                        duration = TURN_DURATION
                    elif action == "Old school bag":
                        duration = REVERSE_DURATION
                    else:
                        duration = 0
                    
                    # Check if action duration completed
                    if elapsed >= duration:
                        self._execute_stop()
                        self.is_executing = False
                        self.in_cooldown = True
                        self.cooldown_start_time = time.time()
                        action_started = False
                        print(f"⏸ Action complete - Entering {COOLDOWN_TIME}s cooldown")
            
            time.sleep(0.05)  # 20Hz update rate
    
    def get_status(self):
        """Get current execution status"""
        with self.lock:
            return {
                'is_executing': self.is_executing,
                'current_action': self.current_action,
                'last_executed_action': self.last_executed_action,
                'in_cooldown': self.in_cooldown
            }
    
    def stop(self):
        """Stop the action manager thread"""
        self.running = False
        self._execute_stop()
    
    # -------- Motor Control Methods --------
    
    def _execute_forward(self):
        """Move forward"""
        self.robot.motor1.forward(SPEED_0_99)
        self.robot.motor2.forward(SPEED_0_99)
        self.robot.motor3.forward(SPEED_0_99)
        self.robot.motor4.forward(SPEED_0_99)
    
    def _execute_turn_left(self):
        """Turn left"""
        self.robot.motor1.speed(-SPEED_0_99) 
        self.robot.motor2.speed(-SPEED_0_99)
        self.robot.motor3.speed(SPEED_0_99)
        self.robot.motor4.speed(SPEED_0_99)
    
    def _execute_turn_right(self):
        """Turn right"""
        self.robot.motor1.speed(SPEED_0_99)
        self.robot.motor2.speed(SPEED_0_99)
        self.robot.motor3.speed(-SPEED_0_99)
        self.robot.motor4.speed(-SPEED_0_99)
    
    def _execute_reverse(self):
        """Move backward"""
        self.robot.motor1.backward(SPEED_0_99)
        self.robot.motor2.backward(SPEED_0_99)
        self.robot.motor3.backward(SPEED_0_99)
        self.robot.motor4.backward(SPEED_0_99)
    
    def _execute_stop(self):
        """Stop all motors"""
        self.robot.stop_all()

# =========================================================================
# === STEP 3: TFLITE INFERENCE AND FRAME CAPTURE WORKER ===
# =========================================================================

def inference_worker(action_manager):
    """Runs TFLite inference continuously and requests actions from ActionManager"""
    global output_frame

    # Load the TFLite model and setup
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Initialize Camera with better settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")
    
    print("✓ Camera opened successfully at 640x480")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from camera.")
                break
            
            # Rotate frame 180 degrees (fix upside-down camera)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # --- Inference and Action Request ---
            # 1. Preprocessing for TFLite
            input_data = cv2.resize(frame, IMG_SIZE)
            input_data_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            input_data_tflite = np.expand_dims(input_data_rgb, axis=0).astype(np.float32)

            # 2. Run Inference
            interpreter.set_tensor(input_details[0]['index'], input_data_tflite)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # 3. Post-processing
            prediction_index = np.argmax(output_data)
            confidence = output_data[0][prediction_index]
            detected_object = LABELS[prediction_index]
            
            # Debug: Print all class probabilities
            print(f"Predictions: Book={output_data[0][0]:.3f}, Newspaper={output_data[0][1]:.3f}, "
                  f"Bag={output_data[0][2]:.3f}, Can={output_data[0][3]:.3f}")
            print(f"Detected: {detected_object} (confidence: {confidence:.3f})")

            # 4. Request action through ActionManager (only if confidence is high enough)
            if confidence >= CONFIDENCE_THRESHOLD:
                action_manager.request_action(detected_object)
            else:
                print(f">>> Low confidence ({confidence:.2f}) - no action")

            # --- Web Stream Preparation ---
            # Get current status from ActionManager
            status = action_manager.get_status()
            
            # Add prediction and status text to frame
            text = f"{detected_object}: {confidence:.2f}"
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Add status overlay
            if status['is_executing']:
                status_text = f"EXECUTING: {status['current_action']}"
                status_color = (0, 255, 255)
            elif status['in_cooldown']:
                status_text = "COOLDOWN"
                status_color = (0, 165, 255)
            else:
                status_text = "READY"
                status_color = (0, 255, 0)
            
            cv2.putText(frame, status_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
            
            # Update the global output frame
            with lock:
                output_frame = frame.copy()

    except Exception as e:
        print(f"Inference worker failed: {e}")
    except KeyboardInterrupt:
        print("Exiting loop.")
    finally:
        action_manager.stop()
        cap.release()

# =========================================================================
# === STEP 4: FLASK WEB SERVER STREAMING ===
# =========================================================================
app = Flask(__name__)

def generate_frames():
    """Video streaming generator function (M-JPEG)."""
    global output_frame
    while True:
        time.sleep(0.05) 
        
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

        if not flag:
            continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    """Returns the streaming response."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    """Renders the HTML page to view the feed."""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robot Vision Stream</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center;
                background: #1a1a1a;
                color: #fff;
                padding: 20px;
            }
            h1 { color: #4CAF50; }
            .info {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 5px;
                margin: 20px auto;
                max-width: 640px;
            }
            img {
                border: 3px solid #4CAF50;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>🤖 Robot Vision & Inference Feed</h1>
        <div class="info">
            <p><strong>Status:</strong> Action Manager Running</p>
            <p>The robot executes timed actions with cooldown periods for smooth movement</p>
        </div>
        <img src="{{ url_for('video_feed') }}" width="640" height="480" alt="Live Video Feed">
    </body>
    </html>
    """)

# =========================================================================
# === STEP 5: MAIN EXECUTION ===
# =========================================================================

if __name__ == '__main__':
    # Test robot connection first
    print("="*60)
    print("TESTING ROBOT CONNECTION...")
    print("="*60)
    try:
        print(f"Robot connected on {ROBOT_PORT} at {ROBOT_BAUD} baud")
        print("\nTesting individual motors for 1 second each...")
        
        print("Testing Motor 1...")
        ROBOT.motor1.forward(30)
        time.sleep(1)
        ROBOT.motor1.stop()
        
        print("Testing Motor 2...")
        ROBOT.motor2.forward(30)
        time.sleep(1)
        ROBOT.motor2.stop()
        
        print("Testing Motor 3...")
        ROBOT.motor3.forward(30)
        time.sleep(1)
        ROBOT.motor3.stop()
        
        print("Testing Motor 4...")
        ROBOT.motor4.forward(30)
        time.sleep(1)
        ROBOT.motor4.stop()
        
        print("\n✓ Motor test complete!")
        print("Did all 4 motors spin? (Check physically)")
        input("Press Enter to continue to main program...")
        
    except Exception as e:
        print(f"✗ Motor test FAILED: {e}")
        print("Cannot proceed - fix hardware issue first")
        exit(1)
    
    print("\n" + "="*60)
    print("STARTING MAIN PROGRAM")
    print("="*60)
    
    # 1. Initialize ActionManager
    print("Initializing Action Manager...")
    action_manager = ActionManager(ROBOT)
    
    # 2. Start the TFLite Inference and Robot Control in a separate thread
    print("Starting TFLite Inference Worker...")
    t = threading.Thread(target=inference_worker, args=(action_manager,))
    t.daemon = True
    t.start()
    
    # 3. Start the Flask web server
    print("\nStarting Web Server on http://<Your-Pi-IP>:8000")
    try:
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping server and robot control.")
    finally:
        action_manager.stop()