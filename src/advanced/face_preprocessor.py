"""
Face Preprocessor Module
Advanced preprocessing pipeline for better face embeddings
"""

import cv2
import numpy as np
from config import config


class FacePreprocessor:
    """
    Advanced face preprocessing for better face recognition accuracy
    
    Applies multiple enhancement techniques:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    2. Bilateral filtering (noise reduction with edge preservation)
    3. Gamma correction (lighting normalization)
    4. Face normalization (zero mean, unit variance)
    """
    
    def __init__(self):
        """Initialize face preprocessor"""
        # CLAHE parameters
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        print("[INFO] Face Preprocessor initialized")
    
    def preprocess(self, face_img):
        """
        Apply full preprocessing pipeline
        
        Args:
            face_img: Face image (BGR or RGB format)
            
        Returns:
            Preprocessed face image
        """
        # 1. Enhance contrast using CLAHE
        face = self.enhance_contrast(face_img)
        
        # 2. Apply bilateral filtering for noise reduction
        face = cv2.bilateralFilter(face, 9, 75, 75)
        
        # 3. Apply gamma correction for lighting normalization
        face = self.adjust_gamma(face, gamma=1.2)
        
        # 4. Normalize face (zero mean, unit variance)
        face = self.normalize_face(face)
        
        return face
    
    def enhance_contrast(self, face_img):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        CLAHE improves local contrast while preventing over-amplification of noise.
        Works on L channel in LAB color space to preserve color information.
        
        Args:
            face_img: Face image in BGR format (OpenCV default)
            
        Returns:
            Contrast-enhanced face image in BGR format
        """
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        
        # Split into L, A, B channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L (lightness) channel
        l = self.clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def adjust_gamma(self, face_img, gamma=1.0):
        """
        Apply gamma correction for lighting normalization
        
        Gamma correction adjusts the luminance of an image:
        - gamma < 1: Brighten the image
        - gamma > 1: Darken the image
        - gamma = 1: No change
        
        Args:
            face_img: Face image
            gamma: Gamma value (default: 1.0)
            
        Returns:
            Gamma-corrected face image
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction using lookup table
        corrected = cv2.LUT(face_img, table)
        
        return corrected
    
    def normalize_face(self, face_img):
        """
        Normalize face to zero mean and unit variance
        
        This standardizes pixel values to improve embedding consistency
        across different lighting conditions.
        
        Args:
            face_img: Face image
            
        Returns:
            Normalized face image
        """
        # Convert to float
        face_float = face_img.astype(np.float32)
        
        # Calculate mean and std per channel
        mean = np.mean(face_float, axis=(0, 1))
        std = np.std(face_float, axis=(0, 1))
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        # Normalize: (x - mean) / std
        normalized = (face_float - mean) / std
        
        # Scale back to [0, 255] range for display/processing
        # Clip to avoid overflow
        normalized = np.clip(normalized * 50 + 128, 0, 255).astype(np.uint8)
        
        return normalized
    
    def preprocess_light(self, face_img):
        """
        Light preprocessing (only CLAHE and bilateral filter)
        
        Faster than full preprocessing, good for real-time applications
        
        Args:
            face_img: Face image
            
        Returns:
            Preprocessed face image
        """
        # 1. Enhance contrast
        face = self.enhance_contrast(face_img)
        
        # 2. Bilateral filter
        face = cv2.bilateralFilter(face, 9, 75, 75)
        
        return face
