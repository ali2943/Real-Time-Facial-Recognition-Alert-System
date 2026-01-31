"""
Liveness Detection Module
Detects spoofing attempts (photos, videos, masks) using multiple strategies
"""

import cv2
import numpy as np
from collections import deque
import time
import config


class LivenessDetector:
    """Anti-spoofing detection using motion, blink, and texture analysis"""
    
    def __init__(self):
        """Initialize liveness detector"""
        self.method = config.LIVENESS_METHOD
        self.frames_required = config.LIVENESS_FRAMES_REQUIRED
        self.require_blink = config.REQUIRE_BLINK
        self.blink_timeout = config.BLINK_TIMEOUT
        self.texture_threshold = config.TEXTURE_ANALYSIS_THRESHOLD
        
        # Frame buffers for temporal analysis
        self.frame_buffer = deque(maxlen=self.frames_required)
        self.face_positions = deque(maxlen=self.frames_required)
        self.landmarks_buffer = deque(maxlen=self.frames_required)
        
        # Blink detection
        self.blink_detected = False
        self.blink_check_start = None
        self.ear_threshold = 0.21  # Eye Aspect Ratio threshold for blink
        
        print(f"[INFO] Liveness Detector initialized (method: {self.method})")
    
    def calculate_ear(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR)
        
        Args:
            eye_points: List of eye landmark points
            
        Returns:
            EAR value (lower when eye is closed)
        """
        # For InsightFace, we only have one point per eye
        # This is a simplified version
        # In a full implementation, you'd need 6 points per eye
        return 0.3  # Default open eye value
    
    def detect_eye_blink(self, landmarks_sequence):
        """
        Detect eye blinks using Eye Aspect Ratio (EAR)
        
        Args:
            landmarks_sequence: Sequence of facial landmarks from multiple frames
            
        Returns:
            True if blink detected, False otherwise
        """
        if landmarks_sequence is None or len(landmarks_sequence) < 3:
            return False
        
        # Calculate EAR for each frame
        ear_values = []
        for landmarks in landmarks_sequence:
            if landmarks and 'left_eye' in landmarks and 'right_eye' in landmarks:
                # Simplified EAR calculation
                # In production, use proper 6-point eye landmarks
                left_ear = self.calculate_ear([landmarks['left_eye']])
                right_ear = self.calculate_ear([landmarks['right_eye']])
                avg_ear = (left_ear + right_ear) / 2
                ear_values.append(avg_ear)
        
        if len(ear_values) < 3:
            return False
        
        # Detect blink pattern: EAR drops then rises
        # Look for valley in EAR values
        for i in range(1, len(ear_values) - 1):
            if ear_values[i] < self.ear_threshold and \
               ear_values[i] < ear_values[i-1] and \
               ear_values[i] < ear_values[i+1]:
                return True
        
        return False
    
    def check_motion_liveness(self, face_sequence):
        """
        Analyze face movement across frames
        
        Args:
            face_sequence: List of face positions (bounding boxes) from multiple frames
            
        Returns:
            Tuple of (is_live: bool, confidence: float)
        """
        if len(face_sequence) < 3:
            # Not enough frames
            return True, 0.5
        
        # Calculate movement between consecutive frames
        movements = []
        for i in range(1, len(face_sequence)):
            prev_box = face_sequence[i-1]
            curr_box = face_sequence[i]
            
            # Calculate center displacement
            prev_center = np.array([prev_box[0] + prev_box[2]/2, prev_box[1] + prev_box[3]/2])
            curr_center = np.array([curr_box[0] + curr_box[2]/2, curr_box[1] + curr_box[3]/2])
            
            displacement = np.linalg.norm(curr_center - prev_center)
            movements.append(displacement)
        
        # Check for natural micro-movements
        avg_movement = np.mean(movements)
        movement_variance = np.var(movements)
        
        # Static images have very low movement and variance
        # Real faces have natural micro-movements (breathing, slight head movement)
        is_static = avg_movement < 1.0 and movement_variance < 0.5
        
        if is_static:
            # Likely a photo
            return False, 0.3
        else:
            # Natural movement detected
            confidence = min(1.0, (avg_movement + movement_variance) / 10.0)
            return True, confidence
    
    def analyze_texture(self, face_image):
        """
        Analyze texture for print/screen detection using LBP
        
        Args:
            face_image: Face image to analyze
            
        Returns:
            Tuple of (is_real: bool, score: float)
        """
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate gradient magnitude (edges)
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Calculate texture score
            # Real faces have more natural texture variation
            # Photos/screens have more uniform patterns or moire patterns
            edge_density = np.mean(gradient_magnitude > 30)
            texture_variance = np.var(gradient_magnitude)
            
            # Combine metrics
            texture_score = (edge_density * 0.5 + min(1.0, texture_variance / 1000) * 0.5)
            
            # High texture score indicates real face
            is_real = texture_score > self.texture_threshold
            
            return is_real, texture_score
        
        except Exception as e:
            print(f"[WARNING] Texture analysis failed: {e}")
            return True, 0.5
    
    def is_live(self, current_frame, face_box, landmarks):
        """
        Multi-strategy liveness check
        
        Args:
            current_frame: Current video frame
            face_box: Bounding box of detected face [x, y, w, h]
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (is_live: bool, confidence: float, reason: str)
        """
        # Add current data to buffers
        self.face_positions.append(face_box)
        self.landmarks_buffer.append(landmarks)
        
        # Extract face region
        x, y, w, h = face_box
        face_region = current_frame[y:y+h, x:x+w]
        self.frame_buffer.append(face_region)
        
        # Initialize blink timer if needed
        if self.require_blink and self.blink_check_start is None:
            self.blink_check_start = time.time()
        
        # Wait for enough frames
        if len(self.frame_buffer) < self.frames_required:
            return True, 0.5, "Collecting frames..."
        
        # Perform liveness checks based on method
        if self.method == 'motion':
            is_live, confidence = self.check_motion_liveness(list(self.face_positions))
            reason = "Motion analysis"
            
        elif self.method == 'blink':
            blink_detected = self.detect_eye_blink(list(self.landmarks_buffer))
            is_live = blink_detected
            confidence = 1.0 if blink_detected else 0.0
            reason = "Blink detected" if blink_detected else "No blink detected"
            
        elif self.method == 'texture':
            is_live, texture_score = self.analyze_texture(face_region)
            confidence = texture_score
            reason = f"Texture analysis (score: {texture_score:.2f})"
            
        elif self.method == 'combined':
            # Combine multiple strategies
            motion_live, motion_conf = self.check_motion_liveness(list(self.face_positions))
            texture_live, texture_score = self.analyze_texture(face_region)
            blink_detected = self.detect_eye_blink(list(self.landmarks_buffer))
            
            # Voting system
            votes = [motion_live, texture_live]
            if self.require_blink:
                votes.append(blink_detected)
            
            is_live = sum(votes) >= len(votes) / 2
            confidence = (motion_conf + texture_score + (1.0 if blink_detected else 0.0)) / 3
            reason = f"Combined (motion:{motion_conf:.2f}, texture:{texture_score:.2f}, blink:{blink_detected})"
            
        else:
            # Default: assume live
            is_live = True
            confidence = 1.0
            reason = "No liveness check"
        
        # Check blink timeout if required
        if self.require_blink and not self.blink_detected:
            elapsed = time.time() - self.blink_check_start
            if elapsed > self.blink_timeout:
                return False, 0.0, "Blink timeout exceeded"
            
            if self.detect_eye_blink(list(self.landmarks_buffer)):
                self.blink_detected = True
        
        return is_live, confidence, reason
    
    def reset(self):
        """Reset liveness detector state"""
        self.frame_buffer.clear()
        self.face_positions.clear()
        self.landmarks_buffer.clear()
        self.blink_detected = False
        self.blink_check_start = None
