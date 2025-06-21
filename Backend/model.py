import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
from typing import Tuple, List, Optional
import logging
from datetime import datetime

class DriverDrowsinessDetector:
    """
    Real-time driver drowsiness detection using MediaPipe Face Mesh and Eye Aspect Ratio (EAR).
    Connects to IP Webcam for live video feed processing.
    """
    
    def __init__(self, 
                 ip_camera_url: str = "http://10.56.19.74:8080/video",
                 ear_threshold: float = 0.25,
                 consecutive_frames: int = 50,
                 enable_audio_alert: bool = True,
                 enable_logging: bool = True):
        """
        Initialize the drowsiness detector.
        
        Args:
            ip_camera_url: URL of the IP Webcam stream
            ear_threshold: Threshold for EAR below which eyes are considered closed
            consecutive_frames: Number of consecutive frames with closed eyes to trigger drowsiness alert
            enable_audio_alert: Whether to enable audio alerts
            enable_logging: Whether to enable event logging
        """
        self.ip_camera_url = ip_camera_url
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.enable_audio_alert = enable_audio_alert
        self.enable_logging = enable_logging
        
        # Detection state
        self.counter = 0  # Counter for consecutive frames with closed eyes
        self.total_blinks = 0
        self.drowsiness_alerts = 0
        self.is_drowsy = False
        self.ear_history = []  # Store recent EAR values for graphing
        self.max_history = 100  # Maximum number of EAR values to store
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Eye landmark indices for MediaPipe Face Mesh
        # Left eye landmarks (16 points)
        self.LEFT_EYE = [
            362, 382, 381, 380, 374, 373, 390, 249,  # Outer eye contour
            263, 466, 388, 387, 386, 385, 384, 398   # Inner eye contour
        ]
        # Right eye landmarks (16 points)
        self.RIGHT_EYE = [
            33, 7, 163, 144, 145, 153, 154, 155,     # Outer eye contour
            133, 173, 157, 158, 159, 160, 161, 246   # Inner eye contour
        ]
        
        # EAR calculation points (6 points for each eye)
        self.LEFT_EAR_POINTS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EAR_POINTS = [33, 160, 158, 133, 153, 145]
        
        # Setup logging
        if self.enable_logging:
            self._setup_logging()
        
        # Audio alert setup
        if self.enable_audio_alert:
            self._setup_audio_alert()
    
    def _setup_logging(self):
        """Setup logging for drowsiness events."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('drowsiness_events.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_audio_alert(self):
        """Setup audio alert system."""
        try:
            import winsound
            self.winsound = winsound
            self.audio_available = True
        except ImportError:
            self.audio_available = False
            print("Warning: Audio alerts not available on this platform")
    
    def _play_audio_alert(self):
        """Play audio alert for drowsiness detection."""
        if self.enable_audio_alert and self.audio_available:
            try:
                # Play a beep sound (frequency: 1000Hz, duration: 500ms)
                self.winsound.Beep(1000, 500)
            except Exception as e:
                print(f"Audio alert error: {e}")
    
    def _log_drowsiness_event(self, ear_value: float, alert_level: str):
        """Log drowsiness detection event."""
        if self.enable_logging:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.warning(
                f"DROWSINESS ALERT - EAR: {ear_value:.3f}, "
                f"Level: {alert_level}, "
                f"Consecutive Frames: {self.counter}"
            )
    
    def compute_ear(self, eye_points: List[int], landmarks) -> float:
        """
        Compute the Eye Aspect Ratio (EAR) for given eye landmarks.
        
        Args:
            eye_points: List of landmark indices for the eye
            landmarks: MediaPipe face landmarks
            
        Returns:
            EAR value (float)
        """
        # Get the coordinates of the eye landmarks
        eye_coords = []
        for point in eye_points:
            x = landmarks.landmark[point].x
            y = landmarks.landmark[point].y
            eye_coords.append((x, y))
        
        # Calculate vertical distances
        A = self._euclidean_distance(eye_coords[1], eye_coords[5])
        B = self._euclidean_distance(eye_coords[2], eye_coords[4])
        
        # Calculate horizontal distance
        C = self._euclidean_distance(eye_coords[0], eye_coords[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _euclidean_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_drowsiness(self, frame: np.ndarray) -> dict:
        """
        Detect drowsiness in a single frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary containing detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        drowsiness_result = {
            'drowsy': False,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'face_detected': False,
            'alert_level': 'normal',
            'consecutive_closed_frames': self.counter
        }
        
        if results.multi_face_landmarks:
            drowsiness_result['face_detected'] = True
            
            for face_landmarks in results.multi_face_landmarks:
                # Calculate EAR for both eyes
                left_ear = self.compute_ear(self.LEFT_EAR_POINTS, face_landmarks)
                right_ear = self.compute_ear(self.RIGHT_EAR_POINTS, face_landmarks)
                
                # Average EAR
                avg_ear = (left_ear + right_ear) / 2.0
                
                drowsiness_result.update({
                    'left_ear': left_ear,
                    'right_ear': right_ear,
                    'avg_ear': avg_ear
                })
                
                # Update EAR history for graphing
                self.ear_history.append(avg_ear)
                if len(self.ear_history) > self.max_history:
                    self.ear_history.pop(0)
                
                # Check if eyes are closed
                if avg_ear < self.ear_threshold:
                    self.counter += 1
                else:
                    # Eyes are open, but only reset counter if they stay open for a few frames
                    # This prevents resetting on brief eye openings during drowsiness
                    if self.counter >= self.consecutive_frames:
                        # If we were in drowsy state, require more frames of open eyes to reset
                        if self.counter >= self.consecutive_frames * 2:
                            # Critical drowsiness - require more frames to reset
                            if self.counter >= self.consecutive_frames * 3:
                                self.counter = 0  # Reset only after sustained alertness
                            else:
                                self.counter = max(self.counter - 1, self.consecutive_frames)  # Gradual reset
                        else:
                            # Warning drowsiness - reset after brief opening
                            self.counter = max(self.counter - 2, 0)
                    else:
                        # Normal state - reset normally
                        if self.counter > 0:
                            self.total_blinks += 1
                        self.counter = 0
                
                drowsiness_result['consecutive_closed_frames'] = self.counter
                
                # Determine drowsiness and alert level
                if self.counter >= self.consecutive_frames:
                    drowsiness_result['drowsy'] = True
                    self.is_drowsy = True
                    self.drowsiness_alerts += 1
                    
                    if self.counter >= self.consecutive_frames * 2:
                        drowsiness_result['alert_level'] = 'critical'
                        self._play_audio_alert()
                        self._log_drowsiness_event(avg_ear, 'critical')
                    else:
                        drowsiness_result['alert_level'] = 'warning'
                        self._log_drowsiness_event(avg_ear, 'warning')
                else:
                    self.is_drowsy = False
        
        return drowsiness_result
    
    def draw_overlay(self, frame: np.ndarray, result: dict):
        """Draw detection results and information overlay on the frame."""
        height, width = frame.shape[:2]
        
        # Draw face detection status
        if result['face_detected']:
            cv2.putText(frame, f"Face: Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Face: Not Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw EAR values
        cv2.putText(frame, f"Left EAR: {result['left_ear']:.3f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Right EAR: {result['right_ear']:.3f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Avg EAR: {result['avg_ear']:.3f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw drowsiness alert
        if result['drowsy']:
            color = (0, 0, 255) if result['alert_level'] == 'critical' else (0, 165, 255)
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Alert Level: {result['alert_level'].upper()}", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "Status: Alert", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw statistics
        cv2.putText(frame, f"Consecutive Closed: {result['consecutive_closed_frames']}", (10, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Blinks: {self.total_blinks}", (10, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Alerts: {self.drowsiness_alerts}", (10, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw EAR threshold line
        cv2.putText(frame, f"Threshold: {self.ear_threshold}", (width - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def draw_ear_graph(self, frame: np.ndarray):
        """Draw a real-time EAR graph on the frame."""
        if len(self.ear_history) < 2:
            return
        
        height, width = frame.shape[:2]
        graph_width = 200
        graph_height = 100
        graph_x = width - graph_width - 20
        graph_y = height - graph_height - 20
        
        # Create graph background
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                     (255, 255, 255), 1)
        
        # Draw threshold line
        threshold_y = graph_y + graph_height - int((self.ear_threshold / 0.4) * graph_height)
        cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_width, threshold_y), 
                (0, 255, 255), 1)
        
        # Draw EAR curve
        if len(self.ear_history) > 1:
            points = []
            for i, ear in enumerate(self.ear_history):
                x = graph_x + int((i / len(self.ear_history)) * graph_width)
                y = graph_y + graph_height - int((ear / 0.4) * graph_height)
                points.append((x, y))
            
            # Draw the curve
            for i in range(1, len(points)):
                color = (0, 255, 0) if self.ear_history[i] >= self.ear_threshold else (0, 0, 255)
                cv2.line(frame, points[i-1], points[i], color, 2)
        
        # Draw graph title
        cv2.putText(frame, "EAR Graph", (graph_x, graph_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_ip_camera_stream(self):
        """Process video stream from IP Webcam."""
        print(f"🚗 Connecting to IP Camera: {self.ip_camera_url}")
        print("📋 Instructions:")
        print("   - Look at the camera")
        print("   - Try closing your eyes for a few seconds")
        print("   - Press 'q' to quit")
        print("   - Press 'r' to reset statistics")
        print("   - Press 's' to show/hide EAR graph")
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
        
        show_graph = True
        last_fps_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from IP camera")
                    break
                
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.start_time)
                    last_fps_time = current_time
                
                # Detect drowsiness
                result = self.detect_drowsiness(frame)
                
                # Draw overlay
                self.draw_overlay(frame, result)
                
                # Draw EAR graph if enabled
                if show_graph:
                    self.draw_ear_graph(frame)
                
                # Display the frame
                cv2.imshow('Driver Drowsiness Detection - IP Camera', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('r'):
                    self.reset_statistics()
                    print("🔄 Statistics reset")
                elif key == ord('s'):
                    show_graph = not show_graph
                    print(f"📊 EAR Graph: {'ON' if show_graph else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_statistics()
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.counter = 0
        self.total_blinks = 0
        self.drowsiness_alerts = 0
        self.is_drowsy = False
        self.ear_history.clear()
        self.frame_count = 0
        self.start_time = time.time()
        print("✅ Statistics reset successfully")
    
    def get_statistics(self) -> dict:
        """Get current detection statistics."""
        return {
            'total_blinks': self.total_blinks,
            'drowsiness_alerts': self.drowsiness_alerts,
            'ear_threshold': self.ear_threshold,
            'consecutive_frames_threshold': self.consecutive_frames,
            'current_fps': self.fps,
            'total_frames_processed': self.frame_count,
            'is_drowsy': self.is_drowsy
        }
    
    def _print_final_statistics(self):
        """Print final statistics when processing ends."""
        stats = self.get_statistics()
        print("\n📊 Final Statistics:")
        print(f"   Total Blinks: {stats['total_blinks']}")
        print(f"   Drowsiness Alerts: {stats['drowsiness_alerts']}")
        print(f"   EAR Threshold: {stats['ear_threshold']}")
        print(f"   Consecutive Frames Threshold: {stats['consecutive_frames_threshold']}")
        print(f"   Total Frames Processed: {stats['total_frames_processed']}")
        print(f"   Average FPS: {stats['current_fps']:.1f}")
        print(f"   Final Drowsy State: {stats['is_drowsy']}")

# Example usage
if __name__ == "__main__":
    # Create detector instance with IP camera URL
    detector = DriverDrowsinessDetector(
        ip_camera_url="http://10.56.19.74:8080/video",
        ear_threshold=0.25,
        consecutive_frames=50,
        enable_audio_alert=True,
        enable_logging=True
    )
    
    # Start real-time detection
    detector.process_ip_camera_stream()
