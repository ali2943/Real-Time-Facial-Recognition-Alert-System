"""
Advanced Preprocessing Pipeline
5-stage preprocessing for optimal recognition
"""

import cv2
import numpy as np


class AdvancedPreprocessingPipeline:
    """
    Multi-stage preprocessing pipeline
    
    Stages:
    1. Noise reduction
    2. Illumination normalization
    3. CLAHE (Contrast enhancement)
    4. Gamma correction
    5. Sharpening
    """
    
    def __init__(self):
        print("[INFO] Advanced Preprocessing Pipeline initialized")
    
    def process(self, face_img, mode='balanced'):
        """
        Apply full preprocessing pipeline
        
        Args:
            face_img: Input face image
            mode: 'light', 'balanced', 'aggressive'
            
        Returns:
            Preprocessed face image
        """
        if mode == 'light':
            return self._light_preprocessing(face_img)
        elif mode == 'balanced':
            return self._balanced_preprocessing(face_img)
        else:
            return self._aggressive_preprocessing(face_img)
    
    def _aggressive_preprocessing(self, face_img):
        """Aggressive preprocessing for difficult conditions"""
        
        # Stage 1: Advanced denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            face_img, None, 
            h=10, hColor=10, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
        
        # Stage 2: CLAHE
        clahe_enhanced = self._apply_clahe(denoised)
        
        # Stage 3: Gamma correction
        gamma_corrected = self._auto_gamma_correction(clahe_enhanced)
        
        # Stage 4: Sharpening
        sharpened = self._unsharp_mask(gamma_corrected)
        
        return sharpened
    
    def _balanced_preprocessing(self, face_img):
        """Balanced preprocessing for normal conditions"""
        
        # Stage 1: Light denoising
        denoised = cv2.bilateralFilter(face_img, 5, 50, 50)
        
        # Stage 2: CLAHE
        clahe_enhanced = self._apply_clahe(denoised)
        
        # Stage 3: Auto gamma
        gamma_corrected = self._auto_gamma_correction(clahe_enhanced)
        
        return gamma_corrected
    
    def _light_preprocessing(self, face_img):
        """Light preprocessing for good conditions"""
        
        # Just CLAHE and light sharpening
        clahe_enhanced = self._apply_clahe(face_img)
        sharpened = self._unsharp_mask(clahe_enhanced, amount=0.5)
        
        return sharpened
    
    def _apply_clahe(self, face_img):
        """Apply CLAHE for contrast enhancement"""
        
        # Convert to LAB
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return bgr_clahe
    
    def _auto_gamma_correction(self, face_img):
        """Auto gamma correction based on image brightness"""
        
        # Calculate mean brightness
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Determine gamma
        if mean_brightness < 80:
            gamma = 1.5  # Brighten dark images
        elif mean_brightness > 170:
            gamma = 0.7  # Darken bright images
        else:
            gamma = 1.0  # No adjustment needed
        
        if gamma == 1.0:
            return face_img
        
        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in range(256)]).astype(np.uint8)
        
        return cv2.LUT(face_img, table)
    
    def _unsharp_mask(self, face_img, amount=1.0):
        """Sharpen using unsharp masking"""
        
        # Create blurred version
        blurred = cv2.GaussianBlur(face_img, (0, 0), 2.0)
        
        # Sharpen
        sharpened = cv2.addWeighted(face_img, 1.0 + amount, blurred, -amount, 0)
        
        return sharpened


# Test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python advanced_preprocessing_pipeline.py <image_path>")
        exit(1)
    
    img = cv2.imread(sys.argv[1])
    
    if img is None:
        print("Failed to load image")
        exit(1)
    
    preprocessor = AdvancedPreprocessingPipeline()
    
    # Test all modes
    light = preprocessor.process(img, 'light')
    balanced = preprocessor.process(img, 'balanced')
    aggressive = preprocessor.process(img, 'aggressive')
    
    # Display
    cv2.imshow('Original', img)
    cv2.imshow('Light', light)
    cv2.imshow('Balanced', balanced)
    cv2.imshow('Aggressive', aggressive)
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()