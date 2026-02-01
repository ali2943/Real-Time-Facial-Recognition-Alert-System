"""
Frame Preprocessing
Prepares raw camera frames for face detection
"""

import cv2
import numpy as np


class FramePreprocessor:
    """
    Preprocesses camera frames before face detection
    
    Features:
    - Auto white balance
    - Noise reduction
    - Contrast enhancement
    - Frame stabilization
    """
    
    def __init__(self):
        self.prev_frame = None
        print("[INFO] Frame Preprocessor initialized")
    
    def preprocess(self, frame):
        """
        Apply preprocessing pipeline to frame
        
        Args:
            frame: Raw camera frame (BGR)
            
        Returns:
            Preprocessed frame
        """
        # Step 1: Denoise
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # Step 2: Auto white balance
        balanced = self.auto_white_balance(denoised)
        
        # Step 3: Contrast enhancement
        enhanced = self.enhance_contrast(balanced)
        
        # Step 4: Sharpen
        sharpened = self.sharpen(enhanced)
        
        return sharpened
    
    def auto_white_balance(self, frame):
        """Automatic white balance using Gray World assumption"""
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    def enhance_contrast(self, frame):
        """CLAHE contrast enhancement"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def sharpen(self, frame):
        """Sharpen image using unsharp masking"""
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        sharpened = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        return sharpened
