# VGG16 Driver Distraction Detection

Real-time distracted driver classification using a pre-trained VGG16 model from the [Distracted Driver Detection](https://github.com/Abhinav1004/Distracted-Driver-Detection) GitHub repository.

## Overview

This application performs **real-time 10-class driver distraction classification** using a pre-trained VGG16 model. It connects to an Android phone's IP camera feed and provides live predictions with temporal smoothing.

### Class Categories

- **c0**: Safe driving
- **c1**: Texting - right
- **c2**: Talking on the phone - right
- **c3**: Texting - left
- **c4**: Talking on the phone - left
- **c5**: Operating the radio
- **c6**: Drinking
- **c7**: Reaching behind
- **c8**: Hair and makeup
- **c9**: Talking to passenger

## Files

- `config_vgg16.py` - Configuration settings
- `run_vgg16_detection.py` - Real-time detection script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained Model

Download the pre-trained VGG16 model from the [GitHub repository](https://github.com/Abhinav1004/Distracted-Driver-Detection) and place it in the current directory as `vgg16_driver_distraction.h5`.

### 3. IP Camera Setup

Install IP Webcam on your Android device and configure it to stream at:
```
http://10.56.19.74:8080/video
```

## Usage

### Run Real-time Detection

```bash
python run_vgg16_detection.py
```

**Optional Arguments:**
```bash
python run_vgg16_detection.py \
    --model vgg16_driver_distraction.h5 \
    --camera http://10.56.19.74:8080/video \
    --confidence 0.5 \
    --window 10
```

### Controls
- Press `q` to quit
- Press `r` to reset statistics

## Features

- **10-class classification** using pre-trained VGG16 model
- **Real-time IP camera processing**
- **Temporal smoothing** (majority voting over 10 frames)
- **Comprehensive UI** with confidence scores, FPS, and top predictions
- **Color-coded status** (Green: Safe, Red: Distracted)

## Model Source

This implementation uses the pre-trained VGG16 model from:
https://github.com/Abhinav1004/Distracted-Driver-Detection

The model was trained on the Kaggle State Farm Distracted Driver Detection dataset and achieves high accuracy on the 10-class classification task. 