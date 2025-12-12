# Smart Object Recognition Robot - MobileNetV2 Transfer Learning (TFLite Deployment)
A vision-based robotics project that uses MobileNetV2 and TensorFlow Lite to classify four custom objects — Book, Newspaper, Zip-top Can, Old School Bag — and translate those detections into real-world motor actions on a Raspberry Pi 4.

The model was trained using transfer learning, optimized with FP16 quantization, and deployed with a multi-threaded Python control system that handles real-time camera inference, action mapping, and safe motor control. When the robot detects an object with > 0.9 confidence, it performs a corresponding movement (turn, move forward, reverse) through an embedded state machine.

A custom dataset (~2.4k images) was collected and labeled, augmented through Roboflow, and fine-tuned for efficient edge deployment. The robot also includes a Flask-based live monitoring system, TFLite inference benchmarking, and an action manager that stabilizes robot movement.

This project demonstrates the full pipeline of dataset creation → training → optimization → Raspberry Pi deployment → real-world robotics testing.
