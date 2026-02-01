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
        """Initialize eye state detector with RELAXED threshold"""
        self.EAR_THRESHOLD = 0.18  # Lowered from 0.21 - more lenient
        self.EAR_CONSEC_FRAMES = 1  # Frames to consider closed
        self.SUNGLASSES_BRIGHTNESS_THRESHOLD = 40  # Lowered from 50 - more lenient
        print("[INFO] Eye State Detector initialized (relaxed threshold)")
    
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
        Check if both eyes are open with better tolerance
        
        Args:
            landmarks: Facial landmarks dictionary
            
        Returns:
            Tuple of (both_open, left_ear, right_ear, reason)
        """
        # Extract eye coordinates
        left_eye = self._get_left_eye_points(landmarks)
        right_eye = self._get_right_eye_points(landmarks)
        
        if left_eye is None or right_eye is None:
            # Can't detect eyes - skip check instead of failing
            if config.DEBUG_MODE:
                print("[DEBUG] Eye landmarks not available - skipping eye check")
            return True, 0.0, 0.0, "Eye landmarks unavailable - check skipped"
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # More lenient check
        both_open = avg_ear > self.EAR_THRESHOLD
        
        if config.DEBUG_MODE:
            print(f"[DEBUG] Eye EAR: Left={left_ear:.3f}, Right={right_ear:.3f}, Avg={avg_ear:.3f}, Threshold={self.EAR_THRESHOLD}")
        
        if not both_open:
            reason = f"Eyes appear closed (EAR: {avg_ear:.3f} < {self.EAR_THRESHOLD})"
        else:
            reason = f"Eyes open (EAR: {avg_ear:.3f})"
        
        return both_open, left_ear, right_ear, reason
    
    def detect_eye_occlusion(self, face_img, landmarks):
        """
        Detect if eyes are occluded (sunglasses, hand, etc.) with HIGHER confidence requirement
        
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
            # Can't extract - skip check
            return False, 0.9, "Eye regions unavailable - check skipped"
        
        # Check for sunglasses (dark, uniform regions) with LOWER threshold (more lenient)
        left_dark = self._is_region_too_dark(left_eye_region, threshold=self.SUNGLASSES_BRIGHTNESS_THRESHOLD)
        right_dark = self._is_region_too_dark(right_eye_region, threshold=self.SUNGLASSES_BRIGHTNESS_THRESHOLD)
        
        # Require BOTH eyes very dark (sunglasses)
        if left_dark and right_dark:
            return True, 0.9, "Sunglasses detected (both eyes very dark)"
        
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
            eye_points = landmarks['left_eye']
        elif eye == 'right' and 'right_eye' in landmarks:
            eye_points = landmarks['right_eye']
        else:
            return None
        
        # Calculate center of eye from all points
        if isinstance(eye_points, np.ndarray) and len(eye_points.shape) == 2:
            # eye_points is array of (x, y) coordinates
            center_x = np.mean(eye_points[:, 0])
            center_y = np.mean(eye_points[:, 1])
        else:
            # Fallback to first point if format is unexpected
            try:
                center_x = eye_points[0] if isinstance(eye_points[0], (int, float, np.number)) else eye_points[0][0]
                center_y = eye_points[1] if isinstance(eye_points[1], (int, float, np.number)) else eye_points[0][1]
            except:
                return None
        
        # Extract region around eye
        h, w = face_img.shape[:2]
        x = self._normalize_coordinate(center_x, w)
        y = self._normalize_coordinate(center_y, h)
        
        margin = 20
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(w, x+margin), min(h, y+margin)
        
        return face_img[y1:y2, x1:x2]
    
    def _normalize_coordinate(self, point, dimension):
        """Normalize coordinate to pixel value"""
        # Handle numpy arrays and scalars
        if isinstance(point, np.ndarray):
            point = float(point)
        return int(point * dimension) if point < 1 else int(point)
    
    def _is_region_too_dark(self, region, threshold=40):
        """
        Check if region is too dark (sunglasses) with configurable threshold
        
        Args:
            region: Eye region image
            threshold: Brightness threshold (default: 40)
            
        Returns:
            True if region is too dark
        """
        if region.size == 0:
            return False  # Can't determine - don't fail
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        mean_brightness = np.mean(gray)
        
        # Very dark = likely sunglasses
        return mean_brightness < threshold
