#!/usr/bin/env python3
"""
Driver Distraction Detection Inference Script
Real-time classification using video stream
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import json

from model import build_model
from data import DistractionDataLoader

class DistractionDetector:
    """
    Real-time driver distraction detection using trained Keras model.
    """
    
    def __init__(self, model_path, img_size=(224, 224), confidence_threshold=0.5):
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Class names
        self.class_names = [
            'Safe Driving',
            'Texting (Right)',
            'Phone (Right)', 
            'Texting (Left)',
            'Phone (Left)',
            'Adjusting Radio',
            'Drinking',
            'Reaching Behind',
            'Hair & Makeup',
            'Talking to Passenger'
        ]
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=5)
        
        # Performance metrics
        self.fps_counter = deque(maxlen=30)
        self.start_time = time.time()
        
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for inference.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed frame ready for model input
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame_resized = cv2.resize(frame_rgb, self.img_size)
        
        # Normalize
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def predict(self, frame):
        """
        Make prediction on a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Make prediction
        predictions = self.model.predict(processed_frame, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return predicted_class, confidence, predictions[0]
    
    def smooth_predictions(self, current_prediction, current_confidence):
        """
        Smooth predictions using moving average.
        
        Args:
            current_prediction: Current frame prediction
            current_confidence: Current frame confidence
            
        Returns:
            Tuple of (smoothed_class, smoothed_confidence)
        """
        self.prediction_history.append((current_prediction, current_confidence))
        
        if len(self.prediction_history) < 3:
            return current_prediction, current_confidence
        
        # Calculate weighted average based on confidence
        total_weight = 0
        weighted_sum = 0
        
        for pred, conf in self.prediction_history:
            weight = conf ** 2  # Square confidence for more weight on high-confidence predictions
            weighted_sum += pred * weight
            total_weight += weight
        
        smoothed_class = int(weighted_sum / total_weight)
        
        # Get average confidence for the smoothed class
        smoothed_confidence = np.mean([
            conf for pred, conf in self.prediction_history 
            if pred == smoothed_class
        ])
        
        return smoothed_class, smoothed_confidence
    
    def draw_predictions(self, frame, predicted_class, confidence, all_probabilities):
        """
        Draw predictions and information on the frame.
        
        Args:
            frame: Input frame
            predicted_class: Predicted class index
            confidence: Prediction confidence
            all_probabilities: All class probabilities
            
        Returns:
            Frame with overlayed information
        """
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Draw background rectangle for text
        cv2.rectangle(output_frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(output_frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        # Draw main prediction
        class_name = self.class_names[predicted_class]
        color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 0, 255)
        
        cv2.putText(output_frame, f"Prediction: {class_name}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(output_frame, f"Confidence: {confidence:.3f}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw top 3 predictions
        top_indices = np.argsort(all_probabilities)[-3:][::-1]
        y_offset = 100
        
        cv2.putText(output_frame, "Top 3 Predictions:", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, idx in enumerate(top_indices):
            prob = all_probabilities[idx]
            name = self.class_names[idx]
            y_pos = y_offset + 25 + (i * 20)
            
            # Color code based on probability
            if prob > 0.7:
                color = (0, 255, 0)  # Green
            elif prob > 0.4:
                color = (0, 255, 255)  # Yellow
            else:
                color = (255, 255, 255)  # White
            
            cv2.putText(output_frame, f"{name}: {prob:.3f}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw FPS
        fps = len(self.fps_counter) / (time.time() - self.start_time) if len(self.fps_counter) > 0 else 0
        cv2.putText(output_frame, f"FPS: {fps:.1f}", 
                   (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw warning if high-risk distraction detected
        high_risk_classes = [1, 2, 3, 4]  # Texting and phone use
        if predicted_class in high_risk_classes and confidence > 0.7:
            cv2.putText(output_frame, "WARNING: HIGH RISK DISTRACTION!", 
                       (width//2 - 200, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 255), 3)
        
        return output_frame
    
    def process_video_stream(self, source, output_path=None, save_logs=True):
        """
        Process video stream in real-time.
        
        Args:
            source: Video source (file path, camera index, or IP camera URL)
            output_path: Path to save output video (optional)
            save_logs: Whether to save prediction logs
        """
        # Open video capture
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Setup logging
        log_data = []
        
        print("Starting real-time distraction detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            frame_count += 1
            frame_start_time = time.time()
            
            # Make prediction
            predicted_class, confidence, all_probabilities = self.predict(frame)
            
            # Smooth predictions
            smoothed_class, smoothed_confidence = self.smooth_predictions(predicted_class, confidence)
            
            # Update FPS counter
            self.fps_counter.append(time.time())
            
            # Draw predictions on frame
            output_frame = self.draw_predictions(frame, smoothed_class, smoothed_confidence, all_probabilities)
            
            # Log prediction
            if save_logs:
                log_entry = {
                    'frame': frame_count,
                    'timestamp': datetime.now().isoformat(),
                    'predicted_class': smoothed_class,
                    'class_name': self.class_names[smoothed_class],
                    'confidence': float(smoothed_confidence),
                    'all_probabilities': all_probabilities.tolist()
                }
                log_data.append(log_entry)
            
            # Display frame
            cv2.imshow('Driver Distraction Detection', output_frame)
            
            # Write frame if output path is provided
            if writer:
                writer.write(output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, output_frame)
                print(f"Screenshot saved: {screenshot_path}")
            
            # Calculate and display FPS
            frame_time = time.time() - frame_start_time
            if frame_count % 30 == 0:
                current_fps = 1.0 / frame_time
                print(f"Frame {frame_count}, FPS: {current_fps:.1f}, "
                      f"Prediction: {self.class_names[smoothed_class]} ({smoothed_confidence:.3f})")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Save logs
        if save_logs and log_data:
            log_path = f"inference_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"Inference log saved: {log_path}")
        
        print("Inference completed!")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time driver distraction detection'
    )
    
    parser.add_argument(
        '--model-path', 
        type=str, 
        required=True,
        help='Path to trained model (.h5 file)'
    )
    
    parser.add_argument(
        '--source', 
        type=str, 
        default='http://10.56.19.74:8080/video',
        help='Video source (file path, camera index, or IP camera URL)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output video path (optional)'
    )
    
    parser.add_argument(
        '--img-size', 
        type=int, 
        default=224,
        help='Input image size for model'
    )
    
    parser.add_argument(
        '--confidence-threshold', 
        type=float, 
        default=0.5,
        help='Confidence threshold for predictions'
    )
    
    parser.add_argument(
        '--no-logs', 
        action='store_true',
        help='Disable prediction logging'
    )
    
    return parser.parse_args()

def main():
    """Main inference function."""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Create detector
    detector = DistractionDetector(
        model_path=args.model_path,
        img_size=(args.img_size, args.img_size),
        confidence_threshold=args.confidence_threshold
    )
    
    # Process video stream
    detector.process_video_stream(
        source=args.source,
        output_path=args.output,
        save_logs=not args.no_logs
    )

if __name__ == "__main__":
    main() 