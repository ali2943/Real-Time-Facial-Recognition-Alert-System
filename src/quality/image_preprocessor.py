"""
Image Preprocessing and Enhancement
Applies filters to improve face recognition accuracy
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Preprocesses face images for better recognition
    
    Applies:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    2. Gamma correction (lighting normalization)
    3. Bilateral filtering (noise reduction)
    4. Color normalization
    """
    
    def __init__(self):
        """Initialize preprocessor"""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print("[INFO] Image Preprocessor initialized")
    
    def preprocess(self, image):
        """
        Apply full preprocessing pipeline
        
        Args:
            image: Input face image (BGR)
            
        Returns:
            Preprocessed image
        """
        # 1. Lighting normalization using CLAHE
        enhanced = self.apply_clahe(image)
        
        # 2. Gamma correction
        enhanced = self.adjust_gamma(enhanced, gamma=1.2)
        
        # 3. Bilateral filtering (reduce noise while preserving edges)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # 4. Color normalization
        enhanced = self.normalize_color(enhanced)
        
        return enhanced
    
    def apply_clahe(self, image):
        """
        Apply CLAHE for adaptive contrast enhancement
        
        Works better than global histogram equalization
        Handles varying lighting conditions
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge back
        enhanced_lab = cv2.merge([l_clahe, a, b])
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def adjust_gamma(self, image, gamma=1.0):
        """
        Gamma correction for lighting normalization
        
        gamma < 1: Brighten image
        gamma > 1: Darken image
        gamma = 1: No change
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    def normalize_color(self, image):
        """
        Normalize color channels to reduce lighting effects
        """
        # Split channels
        b, g, r = cv2.split(image)
        
        # Normalize each channel
        b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
        
        # Merge back
        normalized = cv2.merge([b_norm, g_norm, r_norm])
        
        return normalized
    
    def enhance_for_recognition(self, face_img):
        """
        Quick enhancement for recognition (lighter pipeline)
        """
        # Just CLAHE for speed
        return self.apply_clahe(face_img)
