"""
Configuration file for Driver Drowsiness Detection System
"""

# IP Camera Configuration
IP_CAMERA_URL = "http://10.56.19.74:8080/video"

# Detection Parameters
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold (0.2-0.3 recommended)
CONSECUTIVE_FRAMES = 50  # Number of consecutive frames with closed eyes to trigger alert

# Alert Configuration
ENABLE_AUDIO_ALERT = True  # Enable audio beep alerts
ENABLE_LOGGING = True  # Enable event logging to file

# Performance Configuration
MAX_EAR_HISTORY = 100  # Maximum number of EAR values to store for graphing
FPS_UPDATE_INTERVAL = 1.0  # How often to update FPS display (seconds)

# Display Configuration
SHOW_EAR_GRAPH = True  # Show real-time EAR graph
GRAPH_WIDTH = 200  # Width of the EAR graph in pixels
GRAPH_HEIGHT = 100  # Height of the EAR graph in pixels

# Audio Alert Configuration (Windows only)
AUDIO_FREQUENCY = 1000  # Frequency in Hz for audio alert
AUDIO_DURATION = 500  # Duration in milliseconds for audio alert

# Logging Configuration
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FILE = "drowsiness_events.log"  # Log file name

# MediaPipe Configuration
FACE_MESH_MAX_FACES = 1  # Maximum number of faces to detect
FACE_MESH_REFINE_LANDMARKS = True  # Enable refined landmarks
FACE_MESH_DETECTION_CONFIDENCE = 0.5  # Minimum detection confidence
FACE_MESH_TRACKING_CONFIDENCE = 0.5  # Minimum tracking confidence 