"""
Face Quality Assessment Module
Ensures high-quality face images for accurate recognition
"""

import cv2
import numpy as np
from config import config


class FaceQualityChecker:
    """Comprehensive face quality assessment for optimal recognition accuracy"""
    
    def __init__(self):
        """Initialize quality checker with thresholds from config"""
        self.blur_threshold = config.BLUR_THRESHOLD
        self.brightness_range = config.BRIGHTNESS_RANGE
        self.min_contrast = config.MIN_CONTRAST
        self.max_pose_angle = config.MAX_POSE_ANGLE
        self.min_resolution = config.MIN_FACE_RESOLUTION
        
        print("[INFO] Face Quality Checker initialized")
    
    def check_blur_laplacian(self, image):
        """
        Detect blur using Laplacian variance method
        
        Args:
            image: Face image (grayscale or color)
            
        Returns:
            Tuple of (is_sharp: bool, variance: float)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Higher variance = sharper image
        is_sharp = variance > self.blur_threshold
        
        return is_sharp, variance
    
    def check_brightness(self, image):
        """
        Check if image has adequate lighting
        
        Args:
            image: Face image
            
        Returns:
            Tuple of (is_adequate: bool, mean_brightness: float)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        
        # Check if within acceptable range
        is_adequate = self.brightness_range[0] <= mean_brightness <= self.brightness_range[1]
        
        return is_adequate, mean_brightness
    
    def check_contrast(self, image):
        """
        Ensure sufficient contrast in the image
        
        Args:
            image: Face image
            
        Returns:
            Tuple of (is_adequate: bool, std_dev: float)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate standard deviation (measure of contrast)
        std_dev = np.std(gray)
        
        # Higher std dev = better contrast
        is_adequate = std_dev > self.min_contrast
        
        return is_adequate, std_dev
    
    def check_resolution(self, image):
        """
        Check if face resolution is sufficient
        
        Args:
            image: Face image
            
        Returns:
            Tuple of (is_adequate: bool, min_dimension: int)
        """
        height, width = image.shape[:2]
        min_dimension = min(height, width)
        
        is_adequate = min_dimension >= self.min_resolution
        
        return is_adequate, min_dimension
    
    def check_pose_angle(self, landmarks):
        """
        Estimate head pose angle from facial landmarks
        
        Args:
            landmarks: Dictionary with facial landmark positions
                      Expected keys: 'left_eye', 'right_eye', 'nose'
            
        Returns:
            Tuple of (is_frontal: bool, angle: float)
        """
        if landmarks is None:
            # If no landmarks provided, assume frontal
            return True, 0.0
        
        try:
            # Get eye positions
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            
            # Calculate eye center and distance
            eye_center = (left_eye + right_eye) / 2
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # Calculate roll angle (head tilt)
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            roll_angle = np.abs(np.degrees(np.arctan2(dy, dx)))
            
            # Simple yaw estimation using nose position relative to eye center
            if 'nose' in landmarks:
                nose = np.array(landmarks['nose'])
                nose_offset = np.abs(nose[0] - eye_center[0])
                yaw_ratio = nose_offset / (eye_distance / 2)
                # Convert ratio to approximate angle (rough estimation)
                yaw_angle = yaw_ratio * 30  # Scaled approximation
            else:
                yaw_angle = 0
            
            # Overall pose angle (combine roll and yaw)
            overall_angle = max(roll_angle, yaw_angle)
            
            # Check if within acceptable range
            is_frontal = overall_angle <= self.max_pose_angle
            
            return is_frontal, overall_angle
        
        except Exception as e:
            print(f"[WARNING] Failed to estimate pose angle: {e}")
            return True, 0.0
    
    def check_eyes_visible(self, landmarks):
        """
        Check if both eyes are visible in the image
        
        Args:
            landmarks: Dictionary with facial landmark positions
            
        Returns:
            bool: True if both eyes are visible
        """
        if landmarks is None:
            return True
        
        # Check if eye landmarks are present
        has_left_eye = 'left_eye' in landmarks and landmarks['left_eye'] is not None
        has_right_eye = 'right_eye' in landmarks and landmarks['right_eye'] is not None
        
        return has_left_eye and has_right_eye
    
    def check_all(self, face_image, landmarks=None):
        """
        Comprehensive quality assessment
        
        Args:
            face_image: Face image to check
            landmarks: Optional facial landmarks
            
        Returns:
            Dictionary with all quality check results
        """
        checks = {}
        
        # Blur check
        is_sharp, blur_value = self.check_blur_laplacian(face_image)
        checks['blur'] = {
            'passed': is_sharp,
            'value': blur_value,
            'threshold': self.blur_threshold
        }
        
        # Brightness check
        is_bright_ok, brightness_value = self.check_brightness(face_image)
        checks['brightness'] = {
            'passed': is_bright_ok,
            'value': brightness_value,
            'range': self.brightness_range
        }
        
        # Contrast check
        is_contrast_ok, contrast_value = self.check_contrast(face_image)
        checks['contrast'] = {
            'passed': is_contrast_ok,
            'value': contrast_value,
            'threshold': self.min_contrast
        }
        
        # Resolution check
        is_resolution_ok, resolution_value = self.check_resolution(face_image)
        checks['resolution'] = {
            'passed': is_resolution_ok,
            'value': resolution_value,
            'threshold': self.min_resolution
        }
        
        # Pose check (if landmarks available)
        is_pose_ok, pose_value = self.check_pose_angle(landmarks)
        checks['pose'] = {
            'passed': is_pose_ok,
            'value': pose_value,
            'threshold': self.max_pose_angle
        }
        
        # Eyes visibility check
        eyes_visible = self.check_eyes_visible(landmarks)
        checks['eyes_visible'] = {
            'passed': eyes_visible,
            'value': eyes_visible
        }
        
        return checks
    
    def get_quality_score(self, face_image, landmarks=None):
        """
        Return overall quality score (0-100)
        
        Args:
            face_image: Face image to check
            landmarks: Optional facial landmarks
            
        Returns:
            Quality score from 0-100
        """
        checks = self.check_all(face_image, landmarks)
        
        # Calculate weighted score
        # Weight distribution rationale:
        # - blur (25%): Most critical for feature extraction accuracy
        # - brightness (20%): Essential for consistent lighting
        # - contrast (20%): Important for feature visibility
        # - resolution (15%): Minimum detail requirement
        # - pose (15%): Face orientation impacts matching
        # - eyes_visible (5%): Basic sanity check, binary yes/no
        weights = {
            'blur': 25,
            'brightness': 20,
            'contrast': 20,
            'resolution': 15,
            'pose': 15,
            'eyes_visible': 5
        }
        
        score = 0
        for check_name, weight in weights.items():
            if checks[check_name]['passed']:
                score += weight
            else:
                # Partial credit based on how close to threshold
                if check_name == 'blur':
                    # Blur: ratio of actual to threshold
                    ratio = min(1.0, checks[check_name]['value'] / checks[check_name]['threshold'])
                    score += weight * ratio
                elif check_name == 'brightness':
                    # Brightness: how far from acceptable range
                    value = checks[check_name]['value']
                    min_b, max_b = checks[check_name]['range']
                    if value < min_b:
                        ratio = value / min_b
                    else:
                        ratio = max_b / value
                    score += weight * max(0, ratio)
                elif check_name == 'contrast':
                    # Contrast: ratio to threshold
                    ratio = min(1.0, checks[check_name]['value'] / checks[check_name]['threshold'])
                    score += weight * ratio
                elif check_name == 'resolution':
                    # Resolution: ratio to minimum
                    ratio = min(1.0, checks[check_name]['value'] / checks[check_name]['threshold'])
                    score += weight * ratio
                elif check_name == 'pose':
                    # Pose: inverse ratio (lower angle is better)
                    ratio = max(0, 1 - (checks[check_name]['value'] / checks[check_name]['threshold']))
                    score += weight * ratio
        
        return round(score, 2)
    
    def check_symmetry(self, face_img):
        """
        Check face symmetry
        Real faces are slightly asymmetric
        Perfect symmetry = possible fake
        """
        h, w = face_img.shape[:2]
        mid = w // 2
        
        left = face_img[:, :mid]
        right = cv2.flip(face_img[:, mid:], 1)
        
        # Make same size
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        
        # Calculate difference
        diff = cv2.absdiff(left, right)
        asymmetry = np.mean(diff)
        
        # Real: 20-50, Fake: < 10
        if asymmetry < 10:
            return False, f"Too symmetric ({asymmetry:.1f})"
        return True, f"Natural asymmetry ({asymmetry:.1f})"

    def check_resolution(self, face_img):
        """Check if face has sufficient resolution"""
        h, w = face_img.shape[:2]
        min_dim = min(h, w)
        
        if min_dim < 80:
            return False, f"Resolution too low ({min_dim}px)"
        elif min_dim < 112:
            return True, f"Borderline resolution ({min_dim}px)"
        return True, f"Good resolution ({min_dim}px)"

    def check_noise(self, face_img):
        """Estimate image noise level"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Estimate noise using Laplacian
        noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if noise > 1000:
            return False, f"Too noisy ({noise:.0f})"
        return True, f"Acceptable noise ({noise:.0f})"
