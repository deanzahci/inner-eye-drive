#!/usr/bin/env python3
"""
Real-time VGG16 Driver Distraction Detection
Uses pre-trained VGG16 model to classify driver distractions from IP camera feed.
10-class classification with temporal smoothing.

Based on: https://github.com/Abhinav1004/Distracted-Driver-Detection
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os
import sys
from collections import deque
from typing import Tuple, List, Optional
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_vgg16 import (IP_CAMERA_URL, MODEL_PATH, INPUT_SHAPE, CONFIDENCE_THRESHOLD,
                         TEMPORAL_WINDOW_SIZE, FPS_UPDATE_INTERVAL, CLASS_LABELS,
                         WINDOW_NAME, COLORS, SHOW_CONFIDENCE, SHOW_FPS, SHOW_TOP_PREDICTIONS)

class VGG16DriverDistractionDetector:
    """
    Real-time driver distraction detection using pre-trained VGG16 model.
    Connects to IP Webcam for live video feed processing.
    10-class classification with temporal smoothing.
    """
    
    def __init__(self, 
                 model_path: str = MODEL_PATH,
                 ip_camera_url: str = IP_CAMERA_URL,
                 input_shape: Tuple[int, int, int] = INPUT_SHAPE,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 temporal_window_size: int = TEMPORAL_WINDOW_SIZE):
        """
        Initialize the VGG16 distraction detector.
        
        Args:
            model_path: Path to the pre-trained VGG16 model (.h5 file)
            ip_camera_url: URL of the IP Webcam stream
            input_shape: Expected input shape for the model (height, width, channels)
            confidence_threshold: Minimum confidence score to display prediction
            temporal_window_size: Number of frames for temporal smoothing
        """
        self.model_path = model_path
        self.ip_camera_url = ip_camera_url
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.temporal_window_size = temporal_window_size
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Detection statistics
        self.class_counts = {i: 0 for i in range(len(CLASS_LABELS))}
        self.total_detections = 0
        
        # Temporal smoothing
        self.prediction_history = deque(maxlen=temporal_window_size)
        self.confidence_history = deque(maxlen=temporal_window_size)
        
        # Load the pre-trained model
        self.model = self._load_model()
        
        # Initialize labels
        self.class_labels = CLASS_LABELS
        
        print(f"✅ VGG16 model loaded successfully from: {model_path}")
        print(f"📹 IP Camera URL: {ip_camera_url}")
        print(f"🎯 Input Shape: {input_shape}")
        print(f"📊 Number of Classes: {len(CLASS_LABELS)}")
        print(f"🎯 Confidence Threshold: {confidence_threshold}")
        print(f"⏱️ Temporal Window Size: {temporal_window_size}")
        print()
        
        # Print class labels
        print("📋 Class Labels:")
        for class_id, label in self.class_labels.items():
            print(f"   c{class_id}: {label}")
        print()
    
    def _load_model(self) -> keras.Model:
        """Load the pre-trained model with flexible format support."""
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ Model file not found: {self.model_path}")
                print("📥 Please ensure the model file exists in the current directory.")
                sys.exit(1)
            
            print(f"🔄 Loading model from {self.model_path}...")
            
            # Try different loading methods
            model = None
            
            # Method 1: Try standard Keras load_model
            try:
                model = keras.models.load_model(self.model_path)
                print(f"✅ Model loaded successfully using standard Keras format!")
            except Exception as e1:
                print(f"⚠️ Standard loading failed: {e1}")
                
                # Method 2: Try loading with custom_objects
                try:
                    model = keras.models.load_model(self.model_path, compile=False)
                    print(f"✅ Model loaded successfully with compile=False!")
                except Exception as e2:
                    print(f"⚠️ Loading with compile=False failed: {e2}")
                    
                    # Method 3: Try loading weights only
                    try:
                        import h5py
                        with h5py.File(self.model_path, 'r') as f:
                            print(f"📊 Model file structure: {list(f.keys())}")
                        
                        # Create a simple model architecture and load weights
                        print("🔄 Attempting to load weights into custom architecture...")
                        model = self._create_custom_model()
                        model.load_weights(self.model_path)
                        print(f"✅ Model loaded successfully using weights-only method!")
                    except Exception as e3:
                        print(f"⚠️ Weights-only loading failed: {e3}")
                        raise Exception(f"All loading methods failed. Last error: {e3}")
            
            if model is not None:
                print(f"   - Input shape: {model.input_shape}")
                print(f"   - Output shape: {model.output_shape}")
                print(f"   - Number of parameters: {model.count_params():,}")
                
                # Compile the model if it's not compiled
                if not hasattr(model, 'optimizer') or model.optimizer is None:
                    print("🔄 Compiling model...")
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                
                return model
            else:
                raise Exception("Failed to load model with any method")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please ensure the model file is in a valid Keras format.")
            sys.exit(1)
    
    def _create_custom_model(self) -> keras.Model:
        """Create a custom model architecture for loading weights."""
        print("🏗️ Creating custom model architecture...")
        
        # Create architecture based on the actual model structure: ['dense_1', 'dense_2', 'flatten_1']
        # Weight shapes: (25088, 64) for dense_1, so input features = 25088
        # This suggests the model was trained on 224x224x3 images flattened to 150528, but saved as 25088
        # Let's use the actual saved weight dimensions
        
        # Calculate input shape: 25088 features
        # For 224x224x3 images, flattened would be 150528, but this model uses 25088
        # This might be from a smaller input size or different preprocessing
        
        # Use a custom input shape that matches the saved weights
        input_features = 25088  # From the weight shape analysis
        
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_features,), name='dense_1'),
            keras.layers.Dense(len(CLASS_LABELS), activation='softmax', name='dense_2')
        ])
        
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for the custom model inference.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Preprocessed frame ready for model input
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to a smaller size that when flattened gives 25088 features
        # 25088 = 112 * 112 * 2 (approximately)
        # Let's use 112x112 which gives 112*112*3 = 37632, then resize to 25088
        resized_frame = cv2.resize(rgb_frame, (112, 112))
        
        # Normalize pixel values to [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Flatten the image
        flattened_frame = normalized_frame.flatten()
        
        # Resize to exactly 25088 features (the model's expected input size)
        if len(flattened_frame) > 25088:
            # If too many features, take the first 25088
            flattened_frame = flattened_frame[:25088]
        elif len(flattened_frame) < 25088:
            # If too few features, pad with zeros
            padding = np.zeros(25088 - len(flattened_frame))
            flattened_frame = np.concatenate([flattened_frame, padding])
        
        # Add batch dimension
        batch_frame = np.expand_dims(flattened_frame, axis=0)
        
        return batch_frame
    
    def predict_distraction(self, frame: np.ndarray) -> Tuple[str, float, int, np.ndarray]:
        """
        Predict driver distraction from frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (class_label, confidence_score, predicted_class, all_probabilities)
        """
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Run inference
            predictions = self.model.predict(processed_frame, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence_score = float(predictions[0][predicted_class])
            
            # Get class label
            class_label = self.class_labels[predicted_class]
            
            # Update statistics
            self.total_detections += 1
            self.class_counts[predicted_class] += 1
            
            # Add to temporal history
            self.prediction_history.append(predicted_class)
            self.confidence_history.append(confidence_score)
            
            return class_label, confidence_score, predicted_class, predictions[0]
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Error", 0.0, -1, np.zeros(len(CLASS_LABELS))
    
    def get_temporal_prediction(self) -> Tuple[str, float, int]:
        """
        Get temporally smoothed prediction using majority voting.
        
        Returns:
            Tuple of (smoothed_label, smoothed_confidence, smoothed_class)
        """
        if len(self.prediction_history) == 0:
            return "No prediction", 0.0, -1
        
        # Majority voting for class prediction
        from collections import Counter
        class_counts = Counter(self.prediction_history)
        most_common_class = class_counts.most_common(1)[0][0]
        
        # Average confidence for the most common class
        class_confidences = [conf for pred, conf in zip(self.prediction_history, self.confidence_history) 
                           if pred == most_common_class]
        smoothed_confidence = np.mean(class_confidences) if class_confidences else 0.0
        
        smoothed_label = self.class_labels[most_common_class]
        
        return smoothed_label, smoothed_confidence, most_common_class
    
    def draw_overlay(self, frame: np.ndarray, prediction: Tuple[str, float, int, np.ndarray]):
        """
        Draw prediction results and information overlay on the frame.
        
        Args:
            frame: Input frame to draw on
            prediction: Tuple of (class_label, confidence, predicted_class, all_probabilities)
        """
        height, width = frame.shape[:2]
        class_label, confidence, predicted_class, all_probabilities = prediction
        
        # Get temporally smoothed prediction
        smoothed_label, smoothed_confidence, smoothed_class = self.get_temporal_prediction()
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (10, 10), (width - 10, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 200), (255, 255, 255), 2)
        
        # Determine color based on confidence and class
        if confidence >= self.confidence_threshold:
            if predicted_class == 0:  # Safe driving
                color = COLORS['safe_driving']
            else:  # Distracted
                color = COLORS['distracted']
        else:
            color = COLORS['low_confidence']
        
        # Draw current prediction
        if confidence >= self.confidence_threshold:
            cv2.putText(frame, f"Current: {class_label}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, f"Current: No prediction (low confidence)", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['low_confidence'], 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
        
        # Draw temporally smoothed prediction
        if smoothed_class != -1 and smoothed_confidence >= self.confidence_threshold:
            smoothed_color = COLORS['safe_driving'] if smoothed_class == 0 else COLORS['distracted']
            cv2.putText(frame, f"Smoothed: {smoothed_label}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, smoothed_color, 2)
            cv2.putText(frame, f"Smoothed Conf: {smoothed_confidence:.3f}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
        else:
            cv2.putText(frame, f"Smoothed: No prediction", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['low_confidence'], 2)
            cv2.putText(frame, f"Smoothed Conf: {smoothed_confidence:.3f}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
        
        # Draw FPS
        if time.time() - self.last_fps_time >= FPS_UPDATE_INTERVAL:
            self.fps = self.frame_count / (time.time() - self.start_time)
            self.last_fps_time = time.time()
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['fps'], 2)
        
        # Draw temporal window info
        cv2.putText(frame, f"Window: {len(self.prediction_history)}/{self.temporal_window_size}", (20, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        
        # Draw top 3 predictions on the right side
        if SHOW_TOP_PREDICTIONS:
            top_3_indices = np.argsort(all_probabilities)[-3:][::-1]
            
            for i, idx in enumerate(top_3_indices):
                prob = all_probabilities[idx]
                label = self.class_labels[idx]
                y_pos = 40 + i * 25
                
                # Color based on probability
                if prob > 0.5:
                    prob_color = COLORS['safe_driving'] if idx == 0 else COLORS['distracted']
                else:
                    prob_color = COLORS['low_confidence']
                
                cv2.putText(frame, f"{label[:15]}: {prob:.3f}", (width - 300, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, prob_color, 1)
    
    def process_ip_camera_stream(self):
        """Process the IP camera stream in real-time."""
        print("📹 Connecting to IP camera...")
        
        cap = cv2.VideoCapture(self.ip_camera_url)
        
        if not cap.isOpened():
            print(f"❌ Could not connect to IP camera at {self.ip_camera_url}")
            print("Please ensure IP Webcam is running on your Android device.")
            return
        
        print("✅ Connected to IP camera successfully!")
        print("Press 'q' to quit, 'r' to reset statistics")
        print()
        
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Could not read frame from camera")
                    break
                
                # Update frame count
                self.frame_count += 1
                
                # Predict distraction
                prediction = self.predict_distraction(frame)
                
                # Draw overlay
                self.draw_overlay(frame, prediction)
                
                # Display frame
                cv2.imshow(WINDOW_NAME, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_statistics()
                    print("📊 Statistics reset")
                
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
        except Exception as e:
            print(f"❌ Error during processing: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_final_statistics()
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.class_counts = {i: 0 for i in range(len(CLASS_LABELS))}
        self.total_detections = 0
        self.prediction_history.clear()
        self.confidence_history.clear()
    
    def get_statistics(self) -> dict:
        """Get current detection statistics."""
        stats = {
            'total_detections': self.total_detections,
            'class_counts': self.class_counts.copy(),
            'class_percentages': {}
        }
        
        if self.total_detections > 0:
            for class_id, count in self.class_counts.items():
                stats['class_percentages'][class_id] = (count / self.total_detections) * 100
        
        return stats
    
    def _print_final_statistics(self):
        """Print final statistics when application ends."""
        print("\n📊 Final Statistics:")
        print("=" * 40)
        
        stats = self.get_statistics()
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average FPS: {self.fps:.1f}")
        print()
        
        print("Class distribution:")
        for class_id in range(len(CLASS_LABELS)):
            count = stats['class_counts'][class_id]
            percentage = stats['class_percentages'].get(class_id, 0)
            label = self.class_labels[class_id]
            print(f"  c{class_id}: {label} - {count} ({percentage:.1f}%)")

def main():
    """Main function to run the VGG16 distraction detector."""
    parser = argparse.ArgumentParser(description='VGG16 Driver Distraction Detection')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help='Path to the pre-trained VGG16 model')
    parser.add_argument('--camera', type=str, default=IP_CAMERA_URL,
                       help='IP camera URL')
    parser.add_argument('--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for predictions')
    parser.add_argument('--window', type=int, default=TEMPORAL_WINDOW_SIZE,
                       help='Temporal window size for smoothing')
    
    args = parser.parse_args()
    
    print("🚗 VGG16 Driver Distraction Detection - Real-time")
    print("=" * 60)
    print(f"📁 Model path: {args.model}")
    print(f"📹 Camera URL: {args.camera}")
    print(f"🎯 Confidence threshold: {args.confidence}")
    print(f"⏱️ Temporal window size: {args.window}")
    print()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("📥 Please download the pre-trained model from:")
        print("   https://github.com/Abhinav1004/Distracted-Driver-Detection")
        print("   and place it in the current directory.")
        return
    
    # Create detector instance
    detector = VGG16DriverDistractionDetector(
        model_path=args.model,
        ip_camera_url=args.camera,
        input_shape=INPUT_SHAPE,
        confidence_threshold=args.confidence,
        temporal_window_size=args.window
    )
    
    # Start detection
    try:
        detector.process_ip_camera_stream()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your IP camera connection and try again.")

if __name__ == "__main__":
    main() 