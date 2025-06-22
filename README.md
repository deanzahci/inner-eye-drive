# Driver Distraction Detection via Keras

A complete deep-learning system to detect and classify driver distractions in real-time using Keras and TensorFlow 2.x, inspired by the [Toshi-K Kaggle Distracted Driver Detection](https://github.com/toshi-k/kaggle-distracted-driver-detection) repository.

## 🎯 Features

- **Real-time Detection**: Live video processing from IP cameras or local video files
- **10-Class Classification**: Detects all major driver distraction types
- **Transfer Learning**: Uses EfficientNetB0/MobileNetV2 with pre-trained weights
- **Data Augmentation**: Real-time augmentation during training
- **Model Interpretability**: Grad-CAM visualization support
- **Performance Monitoring**: FPS counter, confidence scores, and prediction smoothing
- **Comprehensive Logging**: Training logs, inference logs, and evaluation metrics
- **Unified Interface**: Single command-line interface for all operations

## 🧠 Detection Classes

The system classifies 10 types of driver behavior:

| Class | Description | Risk Level |
|-------|-------------|------------|
| c0 | Safe Driving | Low |
| c1 | Texting (Right) | **High** |
| c2 | Phone (Right) | **High** |
| c3 | Texting (Left) | **High** |
| c4 | Phone (Left) | **High** |
| c5 | Adjusting Radio | Medium |
| c6 | Drinking | Medium |
| c7 | Reaching Behind | Medium |
| c8 | Hair and Makeup | Medium |
| c9 | Talking to Passenger | Low |

## 📁 Project Structure

```
Backend/
├── run_distraction_detection.py  # Unified interface for all operations
├── model.py                      # Keras model definition and compilation
├── data.py                       # Data loading, preprocessing, and augmentation
├── train.py                      # Training script with CLI flags and callbacks
├── inference.py                  # Real-time inference from video streams
├── utils.py                      # Utility functions (confusion matrix, plots, Grad-CAM)
├── test_system.py                # System testing and validation
├── imgs/                         # Training data directory
│   ├── c0_safe_driving/
│   ├── c1_texting_right/
│   ├── c2_phone_right/
│   └── ... (other classes)
└── outputs/                      # Training outputs (created automatically)
    ├── models/                   # Trained models
    ├── plots/                    # Training plots and confusion matrices
    └── logs/                     # Training logs and TensorBoard files
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd inner-eye-drive

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your training data in the following structure:

```
Backend/imgs/
├── c0_safe_driving/
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── c1_texting_right/
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
└── ... (other classes)
```

### 3. System Testing

```bash
cd Backend

# Test all system components
python run_distraction_detection.py test
```

### 4. Training

```bash
# Basic training with default settings
python run_distraction_detection.py train --data-dir imgs

# Advanced training with custom parameters
python run_distraction_detection.py train \
    --data-dir imgs \
    --backbone efficientnetb0 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --use-augmentation \
    --fine-tune
```

### 5. Real-time Inference

```bash
# Live detection from Samsung phone IP camera
python run_distraction_detection.py inference \
    --model-path outputs/models/distraction_model.h5 \
    --source http://10.56.19.74:8080/video

# Detection from local video file
python run_distraction_detection.py inference \
    --model-path outputs/models/distraction_model.h5 \
    --source video.mp4 \
    --output output_video.mp4
```

## 🎮 Unified Interface

The system provides a single command-line interface for all operations:

```bash
# Show help
python run_distraction_detection.py help

# Train model
python run_distraction_detection.py train [options]

# Run inference
python run_distraction_detection.py inference [options]

# Test system
python run_distraction_detection.py test
```

### Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | imgs | Training data directory |
| `--backbone` | efficientnetb0 | Model architecture |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 32 | Training batch size |
| `--learning-rate` | 1e-4 | Initial learning rate |
| `--use-augmentation` | False | Enable data augmentation |
| `--fine-tune` | False | Enable fine-tuning |
| `--output-dir` | outputs | Output directory |

### Inference Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | Required | Path to trained model |
| `--source` | http://10.56.19.74:8080/video | Video source |
| `--output` | None | Output video path |
| `--confidence-threshold` | 0.5 | Minimum confidence |
| `--img-size` | 224 | Input image size |
| `--no-logs` | False | Disable prediction logging |

## 📊 Model Architecture

### Available Backbones

- **EfficientNetB0** (default): Best accuracy, moderate speed
- **MobileNetV2**: Faster inference, good for real-time
- **Custom CNN**: Lightweight, CPU-friendly

### Training Pipeline

1. **Data Loading**: Stratified train/val split (80/20)
2. **Preprocessing**: Resize to 224×224, normalize
3. **Augmentation**: Rotation, zoom, brightness, contrast, flip
4. **Transfer Learning**: Pre-trained backbone + custom head
5. **Fine-tuning**: Unfreeze top layers for domain adaptation

## 🎥 Real-time Inference

### Supported Input Sources

1. **IP Camera**: `http://10.56.19.74:8080/video`
2. **Local Video**: `video.mp4`, `video.avi`
3. **Webcam**: `0` (default camera)
4. **Image Folder**: Process all images in a directory

### Inference Features

- **Prediction Smoothing**: 5-frame moving average for stable predictions
- **Confidence Thresholding**: Filter low-confidence predictions
- **Top-3 Predictions**: Display multiple class probabilities
- **FPS Monitoring**: Real-time performance tracking
- **High-Risk Alerts**: Special warnings for texting/phone use
- **Screenshot Capture**: Press 's' to save current frame
- **Video Recording**: Save processed video with overlays

## 📈 Evaluation and Analysis

### Training Metrics

- **Accuracy**: Overall classification accuracy
- **Top-3 Accuracy**: Accuracy within top 3 predictions
- **Loss**: Training and validation loss curves
- **Learning Rate**: Adaptive learning rate schedule

### Evaluation Tools

```bash
# Generate confusion matrix
python -c "
from utils import save_confusion_matrix
import numpy as np
# Load your predictions and true labels
save_confusion_matrix(y_true, y_pred, class_names, 'confusion_matrix.png')
"

# Create Grad-CAM visualizations
python -c "
from utils import save_gradcam_visualization
save_gradcam_visualization(model, 'test_image.jpg', class_names)
"
```

### Performance Metrics

- **Overall Accuracy**: ≥92% target on validation set
- **Macro F1-Score**: Balanced performance across classes
- **Per-Class Metrics**: Precision, recall, F1 for each class
- **Inference Speed**: Real-time processing at 15+ FPS

## 🔧 Advanced Features

### Model Interpretability

- **Grad-CAM**: Visualize model attention areas
- **Class Probabilities**: Detailed probability distributions
- **Prediction Confidence**: Confidence-based filtering

### Ensemble Methods

```python
# Load multiple models for ensemble prediction
models = [
    keras.models.load_model('model1.h5'),
    keras.models.load_model('model2.h5'),
    keras.models.load_model('model3.h5')
]

# Average predictions
ensemble_pred = np.mean([model.predict(x) for model in models], axis=0)
```

### TFLite Export

```python
# Convert to TFLite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure TensorFlow ≥2.11.0 is installed
2. **Memory Issues**: Reduce batch size or image size
3. **Slow Inference**: Use MobileNetV2 backbone or TFLite
4. **Poor Accuracy**: Check data quality and class balance

### Performance Optimization

- **GPU Acceleration**: Enable CUDA for faster training
- **Mixed Precision**: Use `tf.keras.mixed_precision` for memory efficiency
- **Data Pipeline**: Use `tf.data` for optimized data loading

## 📝 Usage Examples

### Training Examples

```bash
# Quick training with default settings
python run_distraction_detection.py train --data-dir imgs

# Full training with augmentation and fine-tuning
python run_distraction_detection.py train \
    --data-dir imgs \
    --backbone efficientnetb0 \
    --epochs 100 \
    --use-augmentation \
    --fine-tune

# Lightweight model for CPU deployment
python run_distraction_detection.py train \
    --data-dir imgs \
    --backbone mobilenetv2 \
    --img-size 160 \
    --batch-size 16
```

### Inference Examples

```bash
# Live detection from IP camera
python run_distraction_detection.py inference \
    --model-path models/distraction_model.h5 \
    --source http://10.56.19.74:8080/video

# Process video file with recording
python run_distraction_detection.py inference \
    --model-path models/distraction_model.h5 \
    --source input_video.mp4 \
    --output processed_video.mp4 \
    --confidence-threshold 0.7

# High-sensitivity detection
python run_distraction_detection.py inference \
    --model-path models/distraction_model.h5 \
    --source 0 \
    --confidence-threshold 0.3
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by [Toshi-K's Kaggle Distracted Driver Detection](https://github.com/toshi-k/kaggle-distracted-driver-detection)
- Based on the State Farm Distracted Driver Detection dataset
- Uses TensorFlow/Keras for deep learning implementation

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the example commands above

---

**Note**: This system is designed for educational and research purposes. Always ensure proper safety measures when testing in real driving scenarios.