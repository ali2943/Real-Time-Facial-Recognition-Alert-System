"""
Eye State Detection
Detects if eyes are open or closed using Eye Aspect Ratio (EAR)
"""

import cv2
import numpy as np
import config


class EyeStateDetector:
    """
    Detects eye state (open/closed) using Eye Aspect Ratio (EAR)
    
    EAR (Eye Aspect Ratio):
    - Open eyes: EAR > 0.2
    - Closed eyes: EAR < 0.2
    - Partially closed: EAR = 0.15-0.25
    """
    
    def __init__(self):
        """Initialize eye state detector"""
        self.EAR_THRESHOLD = 0.21  # Below this = closed
        self.EAR_CONSEC_FRAMES = 1  # Frames to consider closed
        print("[INFO] Eye State Detector initialized")
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_landmarks: 6 (x, y) coordinates for one eye
            
        Returns:
            EAR value (float)
        """
        # Compute euclidean distances between vertical eye landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Compute euclidean distance between horizontal eye landmarks
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Compute EAR
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def are_eyes_open(self, landmarks):
        """
        Check if both eyes are open
        
        Args:
            landmarks: Facial landmarks dictionary
            
        Returns:
            Tuple of (both_open, left_ear, right_ear, reason)
        """
        # Extract eye coordinates
        left_eye = self._get_left_eye_points(landmarks)
        right_eye = self._get_right_eye_points(landmarks)
        
        if left_eye is None or right_eye is None:
            return False, 0.0, 0.0, "Eye landmarks not detected"
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Check if eyes are open
        both_open = avg_ear > self.EAR_THRESHOLD
        
        if not both_open:
            if left_ear < self.EAR_THRESHOLD and right_ear < self.EAR_THRESHOLD:
                reason = f"Both eyes closed (EAR: {avg_ear:.3f})"
            elif left_ear < self.EAR_THRESHOLD:
                reason = f"Left eye closed (EAR: {left_ear:.3f})"
            else:
                reason = f"Right eye closed (EAR: {right_ear:.3f})"
        else:
            reason = f"Eyes open (EAR: {avg_ear:.3f})"
        
        return both_open, left_ear, right_ear, reason
    
    def detect_eye_occlusion(self, face_img, landmarks):
        """
        Detect if eyes are occluded (sunglasses, hand, etc.)
        
        Args:
            face_img: Face image
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (is_occluded, confidence, reason)
        """
        # Extract eye regions
        left_eye_region = self._extract_eye_region(face_img, landmarks, 'left')
        right_eye_region = self._extract_eye_region(face_img, landmarks, 'right')
        
        if left_eye_region is None or right_eye_region is None:
            return True, 0.9, "Cannot extract eye regions"
        
        # Check for sunglasses (dark, uniform regions)
        left_dark = self._is_region_too_dark(left_eye_region)
        right_dark = self._is_region_too_dark(right_eye_region)
        
        if left_dark and right_dark:
            return True, 0.95, "Eyes occluded - sunglasses detected"
        elif left_dark:
            return True, 0.8, "Left eye occluded"
        elif right_dark:
            return True, 0.8, "Right eye occluded"
        
        return False, 0.9, "Eyes visible"
    
    def _get_left_eye_points(self, landmarks):
        """Extract left eye landmarks"""
        if 'left_eye' in landmarks:
            return np.array(landmarks['left_eye'])
        return None
    
    def _get_right_eye_points(self, landmarks):
        """Extract right eye landmarks"""
        if 'right_eye' in landmarks:
            return np.array(landmarks['right_eye'])
        return None
    
    def _extract_eye_region(self, face_img, landmarks, eye='left'):
        """Extract eye region from face"""
        if eye == 'left' and 'left_eye' in landmarks:
            eye_point = landmarks['left_eye']
        elif eye == 'right' and 'right_eye' in landmarks:
            eye_point = landmarks['right_eye']
        else:
            return None
        
        # Extract region around eye
        h, w = face_img.shape[:2]
        x, y = int(eye_point[0] * w) if eye_point[0] < 1 else int(eye_point[0]), \
               int(eye_point[1] * h) if eye_point[1] < 1 else int(eye_point[1])
        
        margin = 20
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(w, x+margin), min(h, y+margin)
        
        return face_img[y1:y2, x1:x2]
    
    def _is_region_too_dark(self, region):
        """Check if region is too dark (sunglasses)"""
        if region.size == 0:
            return True
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        mean_brightness = np.mean(gray)
        
        # Very dark = likely sunglasses
        return mean_brightness < 50
