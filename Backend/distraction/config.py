"""
Configuration file for Driver Distraction Detection System
"""

# IP Camera Configuration
IP_CAMERA_URL = "http://10.56.19.74:8080/video"

# Model Configuration
MODEL_PATH = "model.h5"  # Path to the pretrained Keras model
INPUT_SHAPE = (224, 224, 3)  # Expected input shape (height, width, channels)
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score to display prediction

# Binary Classification Labels
# The system now focuses on distracted vs safe driving
BINARY_LABELS = {
    0: "Safe Driving",
    1: "Distracted"
}

# Original 10-class labels (for reference if needed)
ORIGINAL_CLASS_LABELS = {
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

# Binary Classification Mapping
# Maps original 10 classes to binary (0 = safe, 1 = distracted)
CLASS_TO_BINARY_MAPPING = {
    0: 0,  # Safe driving -> Safe
    1: 1,  # Texting - right -> Distracted
    2: 1,  # Talking on the phone - right -> Distracted
    3: 1,  # Texting - left -> Distracted
    4: 1,  # Talking on the phone - left -> Distracted
    5: 1,  # Operating the radio -> Distracted
    6: 1,  # Drinking -> Distracted
    7: 1,  # Reaching behind -> Distracted
    8: 1,  # Hair and makeup -> Distracted
    9: 1   # Talking to passenger -> Distracted
}

# Performance Configuration
FPS_UPDATE_INTERVAL = 1.0  # How often to update FPS display (seconds)
MAX_FRAME_SKIP = 2  # Maximum frames to skip if processing is slow

# Display Configuration
WINDOW_NAME = "Driver Distraction Detection - IP Camera"
SHOW_CONFIDENCE = True  # Show confidence scores on screen
SHOW_FPS = True  # Show FPS on screen
SHOW_BINARY_RESULT = True  # Show binary result (Safe/Distracted)

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
LOG_FILE = "distraction_events.log"  # Log file name
ENABLE_LOGGING = True  # Enable event logging to file

# Model Inference Configuration
BATCH_SIZE = 1  # Batch size for inference (1 for real-time)
VERBOSE_PREDICTION = False  # Show prediction progress (set to False for real-time)

# Error Handling Configuration
MAX_RETRY_ATTEMPTS = 3  # Maximum attempts to reconnect to camera
RETRY_DELAY = 2.0  # Delay between retry attempts (seconds)
FRAME_TIMEOUT = 5.0  # Timeout for frame reading (seconds)

# Audio Alert Configuration (Optional)
ENABLE_AUDIO_ALERT = False  # Enable audio alerts for distractions
AUDIO_ALERT_COOLDOWN = 3.0  # Minimum seconds between audio alerts

# Data Collection Configuration (Optional)
SAVE_DETECTIONS = False  # Save detection results to file
DETECTION_LOG_FILE = "detection_log.csv"  # File to save detection results
SAVE_FRAMES = False  # Save frames with detections
FRAME_SAVE_DIR = "detected_frames"  # Directory to save frames 