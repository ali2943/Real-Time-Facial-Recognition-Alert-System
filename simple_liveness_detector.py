"""
Simple Liveness Detection
Lightweight anti-spoofing checks
"""

import cv2
import numpy as np


class SimpleLivenessDetector:
    """
    Multi-method liveness detection
    
    Methods:
    1. Texture analysis (real skin has rich texture)
    2. Color distribution (real skin has natural colors)
    3. Edge characteristics (photos have different edge patterns)
    """
    
    def __init__(self):
        print("[INFO] Simple Liveness Detector initialized")
    
    def check_liveness(self, face_img):
        """
        Check if face is live
        
        Args:
            face_img: Face image (BGR)
            
        Returns:
            (is_live, confidence, details)
        """
        scores = {}
        
        # Check 1: Texture variance
        texture_score = self._check_texture(face_img)
        scores['texture'] = texture_score
        
        # Check 2: Color distribution
        color_score = self._check_color_distribution(face_img)
        scores['color'] = color_score
        
        # Check 3: Edge characteristics
        edge_score = self._check_edges(face_img)
        scores['edges'] = edge_score
        
        # Overall score (weighted combination)
        overall_score = (
            texture_score * 0.40 +
            color_score * 0.35 +
            edge_score * 0.25
        )
        
        # Decision
        is_live = overall_score > 0.50
        
        details = {
            'texture': texture_score,
            'color': color_score,
            'edges': edge_score,
            'overall': overall_score
        }
        
        return is_live, overall_score, details
    
    def _check_texture(self, face_img):
        """
        Check texture variance
        
        Real face: High micro-texture variance (pores, wrinkles)
        Photo: Low variance (smooth, printed)
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance using sliding window
        kernel_size = 5
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur((gray.astype(np.float32)) ** 2, (kernel_size, kernel_size))
        variance = sqr_mean - mean ** 2
        
        # Average variance across face
        avg_variance = np.mean(variance)
        
        # Normalize to score
        # Typical values: real face ~800-3000, photo ~100-600
        score = min(1.0, max(0.0, (avg_variance - 200) / 2000))
        
        return score
    
    def _check_color_distribution(self, face_img):
        """
        Check color distribution
        
        Real face: Natural skin tones, good saturation
        Photo/screen: Altered colors, low saturation
        """
        # Convert to HSV
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Check if hue is in skin tone range
        # Skin tone: hue 0-20 (red-orange) or 160-180 (wraps around)
        skin_mask = ((h >= 0) & (h <= 25)) | ((h >= 155) & (h <= 180))
        skin_ratio = np.sum(skin_mask) / skin_mask.size
        
        # Check saturation (photos often have lower/higher saturation)
        avg_saturation = np.mean(s)
        saturation_score = 1.0 - abs(avg_saturation - 128) / 128  # Optimal around 128
        
        # Combine
        score = (
            skin_ratio * 0.6 +           # Skin color present
            saturation_score * 0.4       # Natural saturation
        )
        
        return min(1.0, max(0.0, score))
    
    def _check_edges(self, face_img):
        """
        Check edge characteristics
        
        Real face: Smooth, natural edges
        Photo: Sharp boundaries, print artifacts
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Real faces typically have moderate edge density (0.04-0.12)
        # Photos/prints have higher density (sharp boundaries)
        if 0.04 <= edge_density <= 0.12:
            score = 1.0  # Optimal range
        elif edge_density < 0.04:
            score = 0.6  # Too smooth (might be blurred/fake)
        elif edge_density < 0.20:
            score = 0.7  # Slightly high
        else:
            score = 0.3  # Too high (likely print/screen)
        
        return score