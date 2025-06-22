"""
Configuration file for VGG16-based Driver Distraction Detection System
Uses pre-trained model from: https://github.com/Abhinav1004/Distracted-Driver-Detection
"""

# IP Camera Configuration
IP_CAMERA_URL = "http://192.168.86.239:8080/video"

# Model Configuration
MODEL_PATH = "vgg16_driver_distraction.h5"  # Path to the pre-trained VGG16 model
INPUT_SHAPE = (224, 224, 3)  # Expected input shape (height, width, channels)
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence score to display prediction

# 10-Class Classification Labels
CLASS_LABELS = {
    0: "Safe driving",
    1: "Texting - right",
    2: "Talking on the phone - right",
    3: "Texting - left",
    4: "Talking on the phone - left",
    5: "Operating the radio",
    6: "Drinking",
    7: "Reaching behind",
    8: "Hair and makeup",
    9: "Talking to passenger"
}

# Temporal Smoothing Configuration
TEMPORAL_WINDOW_SIZE = 10  # Number of frames for temporal smoothing
FPS_UPDATE_INTERVAL = 1.0  # How often to update FPS display (seconds)

# Performance Configuration
MAX_FRAME_SKIP = 2  # Maximum frames to skip if processing is slow

# Display Configuration
WINDOW_NAME = "VGG16 Driver Distraction Detection - IP Camera"
SHOW_CONFIDENCE = True  # Show confidence scores on screen
SHOW_FPS = True  # Show FPS on screen
SHOW_TOP_PREDICTIONS = True  # Show top 3 predictions

# Color Configuration for Overlay
COLORS = {
    'safe_driving': (0, 255, 0),      # Green
    'distracted': (0, 0, 255),        # Red
    'low_confidence': (128, 128, 128), # Gray
    'text': (255, 255, 255),          # White
    'fps': (0, 255, 255),             # Cyan
    'background': (0, 0, 0)           # Black
}

# Logging Configuration
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FILE = "vgg16_detection_events.log"  # Log file name
ENABLE_LOGGING = True  # Enable event logging to file

# Model Inference Configuration
BATCH_SIZE_INFERENCE = 1  # Batch size for inference (1 for real-time)
VERBOSE_PREDICTION = False  # Show prediction progress (set to False for real-time)

# Error Handling Configuration
MAX_RETRY_ATTEMPTS = 3  # Maximum attempts to reconnect to camera
RETRY_DELAY = 2.0  # Delay between retry attempts (seconds)
FRAME_TIMEOUT = 5.0  # Timeout for frame reading (seconds)

# Data Collection Configuration (Optional)
SAVE_DETECTIONS = False  # Save detection results to file
DETECTION_LOG_FILE = "vgg16_detection_log.csv"  # File to save detection results
SAVE_FRAMES = False  # Save frames with detections
FRAME_SAVE_DIR = "detected_frames_vgg16"  # Directory to save frames 