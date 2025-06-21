import cv2
import mediapipe as mp
import numpy as np
import math
import time
import logging
from datetime import datetime
from typing import Tuple, List, Optional, Dict
import threading

class DriverDistractionDetector:
    """
    Real-time driver distraction detection using MediaPipe Face Mesh.
    Tracks head pose, gaze direction, and face visibility to detect distraction.
    """
    
    def __init__(self, 
                 ip_camera_url: str = "http://10.56.19.74:8080/video",
                 yaw_threshold: float = 25.0,
                 pitch_threshold: float = 30.0,
                 visibility_timeout: float = 3.0,
                 distraction_persistence: float = 2.0,
                 enable_audio_alert: bool = True,
                 enable_logging: bool = True):
        """
        Initialize the distraction detector.
        
        Args:
            ip_camera_url: URL of the IP Webcam stream
            yaw_threshold: Maximum allowed head yaw angle (left/right) in degrees
            pitch_threshold: Maximum allowed head pitch angle (up/down) in degrees
            visibility_timeout: Time in seconds before considering driver not visible
            distraction_persistence: Time in seconds before triggering persistent distraction alert
            enable_audio_alert: Whether to enable audio alerts
            enable_logging: Whether to enable event logging
        """
        self.ip_camera_url = ip_camera_url
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.visibility_timeout = visibility_timeout
        self.distraction_persistence = distraction_persistence
        self.enable_audio_alert = enable_audio_alert
        self.enable_logging = enable_logging
        
        # Detection state
        self.is_distracted = False
        self.distraction_start_time = None
        self.last_face_detection_time = time.time()
        self.distraction_duration = 0.0
        self.total_distractions = 0
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        
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
        
        # Head pose estimation points (nose, left ear, right ear, left eye, right eye)
        self.HEAD_POSE_POINTS = {
            'nose': 1,
            'left_ear': 234,
            'right_ear': 454,
            'left_eye': 33,
            'right_eye': 263,
            'left_cheek': 123,
            'right_cheek': 352
        }
        
        # Gaze tracking points (eye corners and iris)
        self.GAZE_POINTS = {
            'left_eye_corner': 33,
            'right_eye_corner': 263,
            'left_iris': 468,
            'right_iris': 473
        }
        
        # Setup logging
        if self.enable_logging:
            self._setup_logging()
        
        # Audio alert setup
        if self.enable_audio_alert:
            self._setup_audio_alert()
    
    def _setup_logging(self):
        """Setup logging for distraction events."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('distraction_events.log'),
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
        """Play audio alert for distraction detection."""
        if self.enable_audio_alert and self.audio_available:
            try:
                # Play a different beep sound for distraction (frequency: 800Hz, duration: 300ms)
                self.winsound.Beep(800, 300)
            except Exception as e:
                print(f"Audio alert error: {e}")
    
    def _log_distraction_event(self, distraction_type: str, duration: float):
        """Log distraction detection event."""
        if self.enable_logging:
            self.logger.warning(
                f"DISTRACTION ALERT - Type: {distraction_type}, "
                f"Duration: {duration:.2f}s, "
                f"Yaw: {self.current_yaw:.1f}°, "
                f"Pitch: {self.current_pitch:.1f}°"
            )
    
    def estimate_head_pose(self, landmarks) -> Dict[str, float]:
        """
        Estimate head pose (yaw, pitch, roll) from facial landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            Dictionary containing yaw, pitch, and roll angles in degrees
        """
        # Get 3D coordinates of key facial points
        nose = np.array([landmarks.landmark[self.HEAD_POSE_POINTS['nose']].x,
                        landmarks.landmark[self.HEAD_POSE_POINTS['nose']].y,
                        landmarks.landmark[self.HEAD_POSE_POINTS['nose']].z])
        
        left_ear = np.array([landmarks.landmark[self.HEAD_POSE_POINTS['left_ear']].x,
                            landmarks.landmark[self.HEAD_POSE_POINTS['left_ear']].y,
                            landmarks.landmark[self.HEAD_POSE_POINTS['left_ear']].z])
        
        right_ear = np.array([landmarks.landmark[self.HEAD_POSE_POINTS['right_ear']].x,
                             landmarks.landmark[self.HEAD_POSE_POINTS['right_ear']].y,
                             landmarks.landmark[self.HEAD_POSE_POINTS['right_ear']].z])
        
        left_eye = np.array([landmarks.landmark[self.HEAD_POSE_POINTS['left_eye']].x,
                            landmarks.landmark[self.HEAD_POSE_POINTS['left_eye']].y,
                            landmarks.landmark[self.HEAD_POSE_POINTS['left_eye']].z])
        
        right_eye = np.array([landmarks.landmark[self.HEAD_POSE_POINTS['right_eye']].x,
                             landmarks.landmark[self.HEAD_POSE_POINTS['right_eye']].y,
                             landmarks.landmark[self.HEAD_POSE_POINTS['right_eye']].z])
        
        # Calculate head pose angles
        # Yaw (left/right rotation)
        ear_vector = right_ear - left_ear
        yaw = math.degrees(math.atan2(ear_vector[0], abs(ear_vector[2])))
        
        # Pitch (up/down rotation)
        eye_vector = (left_eye + right_eye) / 2 - nose
        pitch = math.degrees(math.atan2(eye_vector[1], abs(eye_vector[2])))
        
        # Roll (head tilt)
        eye_line = right_eye - left_eye
        roll = math.degrees(math.atan2(eye_line[1], eye_line[0]))
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }
    
    def estimate_gaze_direction(self, landmarks) -> Dict[str, float]:
        """
        Estimate gaze direction from eye landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            Dictionary containing gaze angles
        """
        try:
            # Get iris positions (if available)
            left_iris = np.array([landmarks.landmark[self.GAZE_POINTS['left_iris']].x,
                                 landmarks.landmark[self.GAZE_POINTS['left_iris']].y])
            
            right_iris = np.array([landmarks.landmark[self.GAZE_POINTS['right_iris']].x,
                                  landmarks.landmark[self.GAZE_POINTS['right_iris']].y])
            
            # Get eye corner positions
            left_corner = np.array([landmarks.landmark[self.GAZE_POINTS['left_eye_corner']].x,
                                   landmarks.landmark[self.GAZE_POINTS['left_eye_corner']].y])
            
            right_corner = np.array([landmarks.landmark[self.GAZE_POINTS['right_eye_corner']].x,
                                    landmarks.landmark[self.GAZE_POINTS['right_eye_corner']].y])
            
            # Calculate gaze direction relative to eye centers
            left_gaze = left_iris - left_corner
            right_gaze = right_iris - right_corner
            
            # Average gaze direction
            avg_gaze = (left_gaze + right_gaze) / 2
            
            # Convert to angles
            gaze_x = math.degrees(math.atan2(avg_gaze[0], 0.1))  # Horizontal gaze
            gaze_y = math.degrees(math.atan2(avg_gaze[1], 0.1))  # Vertical gaze
            
            return {
                'gaze_x': gaze_x,
                'gaze_y': gaze_y,
                'left_gaze_x': math.degrees(math.atan2(left_gaze[0], 0.1)),
                'right_gaze_x': math.degrees(math.atan2(right_gaze[0], 0.1))
            }
            
        except (IndexError, AttributeError):
            # Fallback if iris landmarks are not available
            return {
                'gaze_x': 0.0,
                'gaze_y': 0.0,
                'left_gaze_x': 0.0,
                'right_gaze_x': 0.0
            }
    
    def detect_distraction(self, frame: np.ndarray) -> dict:
        """
        Detect driver distraction in a single frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary containing detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        distraction_result = {
            'distracted': False,
            'distraction_type': 'none',
            'face_detected': False,
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'gaze_x': 0.0,
            'gaze_y': 0.0,
            'distraction_duration': self.distraction_duration,
            'visibility_time': time.time() - self.last_face_detection_time
        }
        
        current_time = time.time()
        
        if results.multi_face_landmarks:
            # Face detected
            self.last_face_detection_time = current_time
            distraction_result['face_detected'] = True
            
            for face_landmarks in results.multi_face_landmarks:
                # Estimate head pose
                pose_angles = self.estimate_head_pose(face_landmarks)
                self.current_yaw = pose_angles['yaw']
                self.current_pitch = pose_angles['pitch']
                self.current_roll = pose_angles['roll']
                
                # Estimate gaze direction
                gaze_angles = self.estimate_gaze_direction(face_landmarks)
                
                distraction_result.update({
                    'yaw': self.current_yaw,
                    'pitch': self.current_pitch,
                    'roll': self.current_roll,
                    'gaze_x': gaze_angles['gaze_x'],
                    'gaze_y': gaze_angles['gaze_y']
                })
                
                # Check for distraction based on head pose
                is_yaw_distracted = abs(self.current_yaw) > self.yaw_threshold
                is_pitch_distracted = abs(self.current_pitch) > self.pitch_threshold
                
                # Check for distraction based on gaze (optional)
                is_gaze_distracted = abs(gaze_angles['gaze_x']) > 20.0  # 20 degrees gaze threshold
                
                if is_yaw_distracted or is_pitch_distracted or is_gaze_distracted:
                    if not self.is_distracted:
                        # Start of distraction
                        self.is_distracted = True
                        self.distraction_start_time = current_time
                        self.total_distractions += 1
                    
                    # Determine distraction type
                    if is_yaw_distracted:
                        distraction_result['distraction_type'] = 'head_turn'
                    elif is_pitch_distracted:
                        distraction_result['distraction_type'] = 'head_tilt'
                    elif is_gaze_distracted:
                        distraction_result['distraction_type'] = 'gaze_shift'
                    
                    distraction_result['distracted'] = True
                    self.distraction_duration = current_time - self.distraction_start_time
                    distraction_result['distraction_duration'] = self.distraction_duration
                    
                    # Check for persistent distraction
                    if self.distraction_duration >= self.distraction_persistence:
                        self._play_audio_alert()
                        self._log_distraction_event(
                            distraction_result['distraction_type'], 
                            self.distraction_duration
                        )
                else:
                    # No distraction detected
                    if self.is_distracted:
                        # End of distraction
                        self.is_distracted = False
                        self.distraction_start_time = None
                        self.distraction_duration = 0.0
                    
                    distraction_result['distracted'] = False
                    distraction_result['distraction_type'] = 'none'
        else:
            # No face detected
            time_since_face = current_time - self.last_face_detection_time
            
            if time_since_face > self.visibility_timeout:
                # Driver not visible for too long
                if not self.is_distracted:
                    # Start of distraction
                    self.is_distracted = True
                    self.distraction_start_time = current_time
                    self.total_distractions += 1
                
                distraction_result['distracted'] = True
                distraction_result['distraction_type'] = 'not_visible'
                self.distraction_duration = current_time - self.distraction_start_time
                distraction_result['distraction_duration'] = self.distraction_duration
                
                # Check for persistent distraction
                if self.distraction_duration >= self.distraction_persistence:
                    self._play_audio_alert()
                    self._log_distraction_event('not_visible', self.distraction_duration)
        
        return distraction_result
    
    def draw_feedback(self, frame: np.ndarray, result: dict):
        """Draw detection feedback and information overlay on the frame."""
        height, width = frame.shape[:2]
        
        # Draw face detection status
        if result['face_detected']:
            cv2.putText(frame, f"Face: Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Face: Not Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw head pose angles
        cv2.putText(frame, f"Yaw: {result['yaw']:.1f}°", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {result['pitch']:.1f}°", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Roll: {result['roll']:.1f}°", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw gaze direction
        cv2.putText(frame, f"Gaze X: {result['gaze_x']:.1f}°", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Gaze Y: {result['gaze_y']:.1f}°", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw distraction status
        if result['distracted']:
            color = (0, 0, 255) if result['distraction_duration'] >= self.distraction_persistence else (0, 165, 255)
            cv2.putText(frame, "DISTRACTED!", (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Type: {result['distraction_type'].replace('_', ' ').title()}", (10, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Duration: {result['distraction_duration']:.1f}s", (10, 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "FOCUSED", (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Draw statistics
        cv2.putText(frame, f"Total Distractions: {self.total_distractions}", (10, 330), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw thresholds
        cv2.putText(frame, f"Yaw Threshold: ±{self.yaw_threshold}°", (width - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Pitch Threshold: ±{self.pitch_threshold}°", (width - 200, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def draw_facial_landmarks(self, frame: np.ndarray, landmarks):
        """Draw facial landmarks and head pose indicators."""
        # Draw face mesh
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # Draw head pose axes
        if landmarks:
            # Get nose position for head pose visualization
            nose = landmarks.landmark[self.HEAD_POSE_POINTS['nose']]
            nose_x = int(nose.x * frame.shape[1])
            nose_y = int(nose.y * frame.shape[0])
            
            # Draw head pose direction indicator
            if abs(self.current_yaw) > self.yaw_threshold:
                # Draw arrow indicating head turn direction
                arrow_length = 50
                if self.current_yaw > 0:
                    # Turned right
                    cv2.arrowedLine(frame, (nose_x, nose_y), (nose_x + arrow_length, nose_y), 
                                   (0, 0, 255), 3, tipLength=0.3)
                else:
                    # Turned left
                    cv2.arrowedLine(frame, (nose_x, nose_y), (nose_x - arrow_length, nose_y), 
                                   (0, 0, 255), 3, tipLength=0.3)
    
    def process_ip_camera_stream(self):
        """Process video stream from IP Webcam for distraction detection."""
        print(f"🚗 Connecting to IP Camera: {self.ip_camera_url}")
        print("📋 Instructions:")
        print("   - Look at the camera")
        print("   - Try turning your head left/right")
        print("   - Try looking up/down")
        print("   - Press 'q' to quit")
        print("   - Press 'r' to reset statistics")
        print("   - Press 'l' to toggle landmark display")
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
        
        show_landmarks = True
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
                
                # Detect distraction
                result = self.detect_distraction(frame)
                
                # Draw feedback overlay
                self.draw_feedback(frame, result)
                
                # Draw facial landmarks if enabled
                if show_landmarks and result['face_detected']:
                    results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_face_landmarks:
                        self.draw_facial_landmarks(frame, results.multi_face_landmarks[0])
                
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
                elif key == ord('l'):
                    show_landmarks = not show_landmarks
                    print(f"🎯 Landmarks: {'ON' if show_landmarks else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_statistics()
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.is_distracted = False
        self.distraction_start_time = None
        self.distraction_duration = 0.0
        self.total_distractions = 0
        self.last_face_detection_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
        print("✅ Statistics reset successfully")
    
    def get_statistics(self) -> dict:
        """Get current detection statistics."""
        return {
            'total_distractions': self.total_distractions,
            'current_distraction_duration': self.distraction_duration,
            'is_distracted': self.is_distracted,
            'yaw_threshold': self.yaw_threshold,
            'pitch_threshold': self.pitch_threshold,
            'current_fps': self.fps,
            'total_frames_processed': self.frame_count
        }
    
    def _print_final_statistics(self):
        """Print final statistics when processing ends."""
        stats = self.get_statistics()
        print("\n📊 Final Statistics:")
        print(f"   Total Distractions: {stats['total_distractions']}")
        print(f"   Final Distraction Duration: {stats['current_distraction_duration']:.2f}s")
        print(f"   Final Distracted State: {stats['is_distracted']}")
        print(f"   Yaw Threshold: ±{stats['yaw_threshold']}°")
        print(f"   Pitch Threshold: ±{stats['pitch_threshold']}°")
        print(f"   Total Frames Processed: {stats['total_frames_processed']}")
        print(f"   Average FPS: {stats['current_fps']:.1f}")

# Example usage
if __name__ == "__main__":
    # Create detector instance
    detector = DriverDistractionDetector(
        ip_camera_url="http://10.56.19.74:8080/video",
        yaw_threshold=25.0,
        pitch_threshold=30.0,
        visibility_timeout=3.0,
        distraction_persistence=2.0,
        enable_audio_alert=True,
        enable_logging=True
    )
    
    # Start real-time detection
    detector.process_ip_camera_stream() 