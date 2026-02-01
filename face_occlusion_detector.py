"""
Face Occlusion Detection
Detects masks, covered faces, and partial occlusions
"""

import cv2
import numpy as np
import config


class FaceOcclusionDetector:
    """
    Detects face occlusions including masks, hands, and other coverings
    
    Uses multiple strategies:
    1. Mouth visibility detection (masks cover mouth)
    2. Nose visibility detection
    3. Texture analysis (fabric patterns)
    4. Color analysis (medical masks are usually blue/white)
    5. Landmark-based occlusion detection
    """
    
    def __init__(self):
        """Initialize occlusion detector"""
        self.mouth_region_threshold = 0.6
        self.nose_region_threshold = 0.6
        print("[INFO] Face Occlusion Detector initialized")
    
    def is_mouth_visible(self, face_img, landmarks=None):
        """
        Check if mouth region is visible (not covered by mask)
        
        Args:
            face_img: Face image
            landmarks: Facial landmarks (optional)
            
        Returns:
            Tuple of (is_visible, confidence)
        """
        if landmarks is None:
            # Use lower face region estimation
            h, w = face_img.shape[:2]
            mouth_region = face_img[int(h*0.65):h, int(w*0.25):int(w*0.75)]
        else:
            # Use landmarks if available
            # Extract mouth region using landmarks
            # Points 48-68 in dlib are mouth landmarks
            if 'mouth_left' in landmarks and 'mouth_right' in landmarks:
                mouth_region = self._extract_mouth_region(face_img, landmarks)
            else:
                h, w = face_img.shape[:2]
                mouth_region = face_img[int(h*0.65):h, int(w*0.25):int(w*0.75)]
        
        # Analyze mouth region
        # Masks typically have uniform color/texture
        variance = self._calculate_texture_variance(mouth_region)
        
        # Low variance = uniform region = likely covered
        is_visible = variance > self.mouth_region_threshold
        confidence = min(1.0, variance)
        
        return is_visible, confidence
    
    def is_nose_visible(self, face_img, landmarks=None):
        """
        Check if nose region is visible
        
        Args:
            face_img: Face image
            landmarks: Facial landmarks (optional)
            
        Returns:
            Tuple of (is_visible, confidence)
        """
        if landmarks is None:
            h, w = face_img.shape[:2]
            nose_region = face_img[int(h*0.4):int(h*0.65), int(w*0.35):int(w*0.65)]
        else:
            if 'nose' in landmarks:
                nose_region = self._extract_nose_region(face_img, landmarks)
            else:
                h, w = face_img.shape[:2]
                nose_region = face_img[int(h*0.4):int(h*0.65), int(w*0.35):int(w*0.65)]
        
        variance = self._calculate_texture_variance(nose_region)
        is_visible = variance > self.nose_region_threshold
        confidence = min(1.0, variance)
        
        return is_visible, confidence
    
    def detect_mask(self, face_img, landmarks=None):
        """
        Detect if face is wearing a mask
        
        Args:
            face_img: Face image
            landmarks: Facial landmarks (optional)
            
        Returns:
            Tuple of (has_mask, confidence, reason)
        """
        # Check 1: Mouth visibility
        mouth_visible, mouth_conf = self.is_mouth_visible(face_img, landmarks)
        
        # Check 2: Nose visibility
        nose_visible, nose_conf = self.is_nose_visible(face_img, landmarks)
        
        # Check 3: Color analysis (masks are often blue/white)
        mask_color_detected, color_conf = self._detect_mask_color(face_img)
        
        # Combine checks
        if not mouth_visible and not nose_visible:
            return True, 0.9, "Mouth and nose covered - mask detected"
        elif not mouth_visible:
            return True, 0.8, "Mouth covered - likely wearing mask"
        elif mask_color_detected:
            return True, color_conf, "Mask color pattern detected"
        else:
            return False, max(mouth_conf, nose_conf), "Face appears uncovered"
    
    def detect_occlusion(self, face_img, landmarks=None):
        """
        Detect any face occlusion (hand, scarf, etc.)
        
        Args:
            face_img: Face image
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (is_occluded, confidence, occluded_regions)
        """
        occluded_regions = []
        
        # Check different face regions
        mouth_visible, mouth_conf = self.is_mouth_visible(face_img, landmarks)
        nose_visible, nose_conf = self.is_nose_visible(face_img, landmarks)
        
        if not mouth_visible:
            occluded_regions.append("mouth")
        if not nose_visible:
            occluded_regions.append("nose")
        
        # Check for hand/object occlusion using edge detection
        edges = cv2.Canny(face_img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Unusual edge patterns suggest occlusion
        if edge_density > 0.3:  # High edge density
            occluded_regions.append("object_detected")
        
        is_occluded = len(occluded_regions) > 0
        confidence = 1.0 - min(mouth_conf, nose_conf) if is_occluded else max(mouth_conf, nose_conf)
        
        return is_occluded, confidence, occluded_regions
    
    def _calculate_texture_variance(self, region):
        """Calculate texture variance (higher = more texture)"""
        if region.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Calculate standard deviation (texture indicator)
        std = np.std(gray)
        
        # Normalize to 0-1 range
        normalized = min(1.0, std / 50.0)
        
        return normalized
    
    def _detect_mask_color(self, face_img):
        """Detect typical mask colors (blue, white, black)"""
        h, w = face_img.shape[:2]
        lower_face = face_img[int(h*0.5):, :]
        
        # Convert to HSV
        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        
        # Blue mask detection
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
        
        # White mask detection
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_ratio = np.sum(white_mask > 0) / white_mask.size
        
        # Black mask detection
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        black_ratio = np.sum(black_mask > 0) / black_mask.size
        
        # If significant portion is mask color
        max_ratio = max(blue_ratio, white_ratio, black_ratio)
        
        if max_ratio > 0.3:  # 30% of lower face is mask color
            return True, max_ratio
        
        return False, 0.0
    
    def _extract_mouth_region(self, face_img, landmarks):
        """Extract mouth region using landmarks"""
        # Placeholder - implement based on your landmark format
        h, w = face_img.shape[:2]
        return face_img[int(h*0.65):h, int(w*0.25):int(w*0.75)]
    
    def _extract_nose_region(self, face_img, landmarks):
        """Extract nose region using landmarks"""
        # Placeholder - implement based on your landmark format
        h, w = face_img.shape[:2]
        return face_img[int(h*0.4):int(h*0.65), int(w*0.35):int(w*0.65)]
