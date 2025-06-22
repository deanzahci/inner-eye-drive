# Driver Distraction Detection System

A real-time driver distraction detection system using a pretrained Keras CNN model. This application connects to an IP camera feed and classifies driver activities into **binary categories**: Safe Driving vs Distracted.

## Features

- **Real-time Detection**: Processes live video feed from IP camera
- **Binary Classification**: Simple Safe Driving vs Distracted output
- **High Performance**: Optimized for real-time inference
- **Configurable**: Easy to adjust settings via config file
- **Visual Overlay**: Displays predictions and confidence scores on video
- **FPS Monitoring**: Shows real-time performance metrics
- **Statistics Tracking**: Monitors safe vs distracted frame percentages

## Binary Classification

The system provides a simple binary output:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Safe Driving | Normal driving behavior |
| 1 | Distracted | Any distracting activity detected |

### Original 10 Classes (Mapped to Binary)

The system uses a 10-class model internally but maps the results to binary:

| Original Class | Activity | Binary Result |
|----------------|----------|---------------|
| c0 | Safe driving | Safe Driving |
| c1 | Texting - right | Distracted |
| c2 | Talking on the phone - right | Distracted |
| c3 | Texting - left | Distracted |
| c4 | Talking on the phone - left | Distracted |
| c5 | Operating the radio | Distracted |
| c6 | Drinking | Distracted |
| c7 | Reaching behind | Distracted |
| c8 | Hair and makeup | Distracted |
| c9 | Talking to passenger | Distracted |

## Requirements

### Hardware
- Computer with webcam or IP camera setup
- Android phone with IP Webcam app (for mobile camera feed)
- Stable network connection

### Software
- Python 3.8+
- TensorFlow 2.8+
- OpenCV 4.5+
- NumPy 1.21+

## Installation

1. **Clone or navigate to the distraction folder:**
   ```bash
   cd Backend/distraction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a pretrained model:**
   - Place a `model.h5` file in the current directory
   - The model should be trained on the Kaggle State Farm Distracted Driver dataset
   - Expected input shape: (224, 224, 3)
   - Expected output: 10 classes (c0-c9) - will be mapped to binary

## Setup IP Camera

1. **Install IP Webcam on your Android phone:**
   - Download from Google Play Store
   - Launch the app and start the server

2. **Get your phone's IP address:**
   - The app will display the IP address and port
   - Default URL format: `http://[PHONE_IP]:8080/video`

3. **Update configuration:**
   - Edit `config.py` and update `IP_CAMERA_URL`
   - Default: `http://10.56.19.74:8080/video`

## Usage

### Quick Start
```bash
python run_distraction_detection.py
```

### Direct Usage
```bash
python distraction_detector.py
```

### Test System
```bash
python test_system.py
```

### Configuration
Edit `config.py` to customize:
- IP camera URL
- Model path
- Confidence threshold
- Display options
- Performance settings

## Controls

- **q**: Quit the application
- **r**: Reset statistics
- **Ctrl+C**: Force quit (terminal)

## Configuration Options

### Model Settings
```python
MODEL_PATH = "model.h5"              # Path to pretrained model
INPUT_SHAPE = (224, 224, 3)          # Model input dimensions
CONFIDENCE_THRESHOLD = 0.5           # Minimum confidence to display
```

### Camera Settings
```python
IP_CAMERA_URL = "http://10.56.19.74:8080/video"  # Camera feed URL
```

### Display Settings
```python
SHOW_CONFIDENCE = True        # Show confidence scores
SHOW_FPS = True              # Show FPS counter
SHOW_BINARY_RESULT = True    # Show binary result (Safe/Distracted)
```

## Model Architecture

The system expects a Keras model with:
- **Input**: RGB images of shape (224, 224, 3)
- **Output**: 10-class softmax probabilities (mapped to binary)
- **Preprocessing**: Normalized pixel values [0, 1]

### Binary Mapping
The system automatically maps the 10-class output to binary:
- **Class 0 (Safe driving)** → Binary 0 (Safe Driving)
- **Classes 1-9 (All distractions)** → Binary 1 (Distracted)

### Recommended Model Sources
- [Kaggle Distracted Driver Detection](https://github.com/toshi-k/kaggle-distracted-driver-detection)
- [State Farm Distracted Driver Dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

## Performance Tips

1. **Optimize Model Size**: Use lightweight architectures for real-time performance
2. **Adjust Confidence Threshold**: Lower values show more predictions, higher values show only confident ones
3. **Network Quality**: Ensure stable network connection for IP camera
4. **Hardware**: Use GPU acceleration if available for better performance

## Troubleshooting

### Common Issues

**Model not found:**
```
❌ Model file not found: model.h5
```
- Ensure `model.h5` exists in the current directory
- Check file permissions

**Camera connection failed:**
```
❌ Error: Could not connect to IP camera
```
- Verify IP Webcam is running on your phone
- Check network connectivity
- Update IP address in `config.py`

**Low FPS:**
- Reduce input image size in `config.py`
- Use a lighter model architecture
- Enable GPU acceleration if available

**High memory usage:**
- Close other applications
- Reduce batch size in `config.py`
- Use model quantization

## File Structure

```
Backend/distraction/
├── distraction_detector.py      # Main detection application
├── run_distraction_detection.py # Runner script
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── test_system.py               # System testing
└── model.h5                     # Pretrained model (not included)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Inner Eye Drive system. Please refer to the main project license.

## Acknowledgments

- Based on the Kaggle State Farm Distracted Driver Detection competition
- Uses TensorFlow and Keras for deep learning
- OpenCV for computer vision processing 