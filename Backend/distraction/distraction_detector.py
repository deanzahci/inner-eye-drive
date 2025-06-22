#!/usr/bin/env python3
"""
Real-time Driver Distraction Detection
Uses a pretrained Keras CNN model to classify driver distractions from IP camera feed.
Binary classification: Safe Driving vs Distracted
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os
import sys
from typing import Tuple, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (IP_CAMERA_URL, MODEL_PATH, INPUT_SHAPE, BINARY_LABELS, 
                   CONFIDENCE_THRESHOLD, CLASS_TO_BINARY_MAPPING, ORIGINAL_CLASS_LABELS)

class DriverDistractionDetector:
    """
    Real-time driver distraction detection using pretrained Keras CNN model.
    Connects to IP Webcam for live video feed processing.
    Binary classification: Safe Driving vs Distracted
    """
    
    def __init__(self, 
                 model_path: str = MODEL_PATH,
                 ip_camera_url: str = IP_CAMERA_URL,
                 input_shape: Tuple[int, int, int] = INPUT_SHAPE,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        """
        Initialize the distraction detector.
        
        Args:
            model_path: Path to the pretrained Keras model (.h5 file)
            ip_camera_url: URL of the IP Webcam stream
            input_shape: Expected input shape for the model (height, width, channels)
            confidence_threshold: Minimum confidence score to display prediction
        """
        self.model_path = model_path
        self.ip_camera_url = ip_camera_url
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Detection statistics
        self.safe_frames = 0
        self.distracted_frames = 0
        self.total_detections = 0
        
        # Load the pretrained model
        self.model = self._load_model()
        
        # Initialize labels
        self.binary_labels = BINARY_LABELS
        self.class_to_binary_mapping = CLASS_TO_BINARY_MAPPING
        self.original_class_labels = ORIGINAL_CLASS_LABELS
        
        print(f"✅ Model loaded successfully from: {model_path}")
        print(f"📹 IP Camera URL: {ip_camera_url}")
        print(f"🎯 Input Shape: {input_shape}")
        print(f"📊 Confidence Threshold: {confidence_threshold}")
        print(f"🎯 Binary Classification: Safe Driving vs Distracted")
        print()
    
    def _load_model(self) -> keras.Model:
        """Load the pretrained Keras model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"🔄 Loading model from {self.model_path}...")
            model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully!")
            print(f"   - Input shape: {model.input_shape}")
            print(f"   - Output shape: {model.output_shape}")
            print(f"   - Number of parameters: {model.count_params():,}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please ensure the model.h5 file exists in the current directory.")
            sys.exit(1)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model inference.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Preprocessed frame ready for model input
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input shape
        resized_frame = cv2.resize(rgb_frame, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize pixel values to [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch_frame = np.expand_dims(normalized_frame, axis=0)
        
        return batch_frame
    
    def predict_distraction(self, frame: np.ndarray) -> Tuple[str, float, int, str]:
        """
        Predict driver distraction from frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (binary_label, confidence_score, binary_class, original_class_label)
        """
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Run inference
            predictions = self.model.predict(processed_frame, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence_score = float(predictions[0][predicted_class])
            
            # Map to binary classification
            binary_class = self.class_to_binary_mapping[predicted_class]
            binary_label = self.binary_labels[binary_class]
            original_class_label = self.original_class_labels[predicted_class]
            
            # Update statistics
            self.total_detections += 1
            if binary_class == 0:
                self.safe_frames += 1
            else:
                self.distracted_frames += 1
            
            return binary_label, confidence_score, binary_class, original_class_label
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Error", 0.0, -1, "Unknown"
    
    def draw_overlay(self, frame: np.ndarray, prediction: Tuple[str, float, int, str]):
        """
        Draw prediction results and information overlay on the frame.
        
        Args:
            frame: Input frame to draw on
            prediction: Tuple of (binary_label, confidence, binary_class, original_class_label)
        """
        height, width = frame.shape[:2]
        binary_label, confidence, binary_class, original_class_label = prediction
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (10, 10), (width - 10, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 140), (255, 255, 255), 2)
        
        # Determine color based on confidence and binary class
        if confidence >= self.confidence_threshold:
            if binary_class == 0:  # Safe driving
                color = (0, 255, 0)  # Green
                status = "SAFE"
            else:  # Distracted
                color = (0, 0, 255)  # Red
                status = "DISTRACTED"
        else:
            color = (128, 128, 128)  # Gray for low confidence
            status = "UNCERTAIN"
        
        # Draw main status
        cv2.putText(frame, f"Status: {status}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        # Draw binary label
        cv2.putText(frame, f"Binary: {binary_label}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw confidence score
        cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw original class (if different from binary)
        if binary_class != 0:  # Only show for distracted cases
            cv2.putText(frame, f"Activity: {original_class_label}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw statistics
        safe_percent = (self.safe_frames / max(self.total_detections, 1)) * 100
        distracted_percent = (self.distracted_frames / max(self.total_detections, 1)) * 100
        cv2.putText(frame, f"Safe: {safe_percent:.1f}%", (width - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Distracted: {distracted_percent:.1f}%", (width - 150, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit", (width - 150, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_ip_camera_stream(self):
        """Process video stream from IP Webcam."""
        print(f"🚗 Connecting to IP Camera: {self.ip_camera_url}")
        print("📋 Instructions:")
        print("   - Look at the camera")
        print("   - Try different activities to test distraction detection")
        print("   - Press 'q' to quit")
        print("   - Press 'r' to reset statistics")
        print()
        
        cap = cv2.VideoCapture(self.ip_camera_url)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not connect to IP camera at {self.ip_camera_url}")
            print("   Please check:")
            print("   - IP Webcam is running on your phone")
            print("   - IP address is correct")
            print("   - Both devices are on the same network")
            return
        
        print("✅ Successfully connected to IP camera")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from IP camera")
                    break
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.start_time)
                    self.last_fps_time = current_time
                
                # Predict distraction
                prediction = self.predict_distraction(frame)
                
                # Draw overlay
                self.draw_overlay(frame, prediction)
                
                # Display the frame
                cv2.imshow('Driver Distraction Detection - IP Camera', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('r'):
                    self.reset_statistics()
                    print("🔄 Statistics reset")
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_statistics()
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.last_fps_time = time.time()
        self.safe_frames = 0
        self.distracted_frames = 0
        self.total_detections = 0
        print("✅ Statistics reset successfully")
    
    def get_statistics(self) -> dict:
        """Get current detection statistics."""
        safe_percent = (self.safe_frames / max(self.total_detections, 1)) * 100
        distracted_percent = (self.distracted_frames / max(self.total_detections, 1)) * 100
        
        return {
            'total_frames_processed': self.frame_count,
            'current_fps': self.fps,
            'model_path': self.model_path,
            'input_shape': self.input_shape,
            'confidence_threshold': self.confidence_threshold,
            'total_detections': self.total_detections,
            'safe_frames': self.safe_frames,
            'distracted_frames': self.distracted_frames,
            'safe_percentage': safe_percent,
            'distracted_percentage': distracted_percent
        }
    
    def _print_final_statistics(self):
        """Print final statistics when processing ends."""
        stats = self.get_statistics()
        print("\n📊 Final Statistics:")
        print(f"   Total Frames Processed: {stats['total_frames_processed']}")
        print(f"   Average FPS: {stats['current_fps']:.1f}")
        print(f"   Total Detections: {stats['total_detections']}")
        print(f"   Safe Frames: {stats['safe_frames']} ({stats['safe_percentage']:.1f}%)")
        print(f"   Distracted Frames: {stats['distracted_frames']} ({stats['distracted_percentage']:.1f}%)")
        print(f"   Model Path: {stats['model_path']}")
        print(f"   Input Shape: {stats['input_shape']}")
        print(f"   Confidence Threshold: {stats['confidence_threshold']}")


def main():
    """Main function to run the distraction detector."""
    print("🚗 Driver Distraction Detection System")
    print("=" * 50)
    
    # Create detector instance
    detector = DriverDistractionDetector()
    
    # Start detection
    try:
        detector.process_ip_camera_stream()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your IP camera connection and model file.")

if __name__ == "__main__":
    main() 