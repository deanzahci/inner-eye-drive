"""
Configuration file for EEG Data Acquisition and Dizziness Detection
"""

# OpenBCI Configuration
PORT = "/dev/ttyUSB0"  # Default OpenBCI port (adjust for your system)
# For macOS, might be something like "/dev/tty.usbserial-XXXXXXXX"
# For Windows, might be something like "COM3"

# EEG Data Parameters
CHANNEL_COUNT = 8  # Number of EEG channels to use
SAMPLE_RATE = 250  # Hz - OpenBCI Cyton sampling rate
WINDOW_SIZE = 256  # Number of samples per analysis window (for FFT)
OVERLAP = 0.5  # Overlap ratio between consecutive windows (0.0 to 1.0)

# Processing Parameters
BUFFER_MAX_SIZE = 1000  # Maximum number of samples to keep in buffer
QUEUE_MAX_SIZE = 50  # Maximum number of epochs in processing queue

# Model Parameters
FFT_START_INDEX = 1  # Start index for FFT (skip DC component)
FFT_END_INDEX = 33   # End index for FFT (first 32 frequency bins)

# Dizziness Classification Levels
DIZZINESS_LEVELS = {
    0: "None",
    1: "Low", 
    2: "Moderate",
    3: "High"
}

# Alert Thresholds
HIGH_DIZZINESS_THRESHOLD = 0.7  # Threshold for high dizziness probability
MODERATE_DIZZINESS_THRESHOLD = 0.5  # Threshold for moderate dizziness probability
