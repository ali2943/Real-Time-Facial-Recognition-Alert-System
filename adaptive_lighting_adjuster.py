"""
Adaptive Lighting Adjustment
Automatically adjusts image brightness and contrast
"""

import cv2
import numpy as np


class AdaptiveLightingAdjuster:
    """
    Automatically adjusts lighting in real-time
    
    Techniques:
    1. Histogram equalization (CLAHE)
    2. Gamma correction
    3. Adaptive brightness
    4. White balance
    """
    
    def __init__(self):
        # Target brightness range
        self.target_brightness = 128  # 0-255 scale
        self.brightness_tolerance = 30
        
        # CLAHE for local contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        print("[INFO] Adaptive Lighting Adjuster initialized")
    
    def adjust_lighting(self, frame, mode='auto'):
        """
        Adjust lighting in frame
        
        Args:
            frame: Input BGR image
            mode: 'auto', 'brighten', 'darken', 'balance'
            
        Returns:
            Adjusted frame
        """
        if mode == 'auto':
            return self._auto_adjust(frame)
        elif mode == 'brighten':
            return self._brighten(frame)
        elif mode == 'darken':
            return self._darken(frame)
        elif mode == 'balance':
            return self._balance_lighting(frame)
        else:
            return frame
    
    def _auto_adjust(self, frame):
        """
        Automatically adjust based on image analysis
        """
        # Calculate current brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)
        
        # Decide action based on brightness
        if current_brightness < self.target_brightness - self.brightness_tolerance:
            # Too dark - brighten
            adjusted = self._brighten(frame, current_brightness)
        elif current_brightness > self.target_brightness + self.brightness_tolerance:
            # Too bright - darken
            adjusted = self._darken(frame, current_brightness)
        else:
            # Just apply local contrast enhancement
            adjusted = self._enhance_contrast(frame)
        
        return adjusted
    
    def _brighten(self, frame, current_brightness=None):
        """Brighten dark images"""
        
        if current_brightness is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_brightness = np.mean(gray)
        
        # Calculate gamma for brightening
        # Darker image needs higher gamma
        # gamma > 1 brightens, gamma < 1 darkens
        
        if current_brightness < 60:
            gamma = 2.0  # Very dark
        elif current_brightness < 100:
            gamma = 1.5  # Dark
        else:
            gamma = 1.2  # Slightly dark
        
        # Apply gamma correction
        adjusted = self._gamma_correction(frame, gamma)
        
        # Also apply CLAHE for local details
        adjusted = self._enhance_contrast(adjusted)
        
        return adjusted
    
    def _darken(self, frame, current_brightness=None):
        """Darken bright images"""
        
        if current_brightness is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_brightness = np.mean(gray)
        
        # Calculate gamma for darkening
        if current_brightness > 200:
            gamma = 0.5  # Very bright
        elif current_brightness > 170:
            gamma = 0.7  # Bright
        else:
            gamma = 0.85  # Slightly bright
        
        # Apply gamma correction
        adjusted = self._gamma_correction(frame, gamma)
        
        return adjusted
    
    def _gamma_correction(self, frame, gamma):
        """
        Apply gamma correction
        
        gamma > 1: brighten
        gamma < 1: darken
        gamma = 1: no change
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        
        # Apply lookup table
        return cv2.LUT(frame, table)
    
    def _enhance_contrast(self, frame):
        """Apply CLAHE for local contrast enhancement"""
        
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _balance_lighting(self, frame):
        """
        Advanced: Balance lighting across face
        Corrects uneven lighting (one side bright, other dark)
        """
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate illumination map using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        
        # Estimate background illumination
        bg = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        
        # Normalize
        normalized = cv2.divide(l, bg, scale=255)
        
        # Merge back
        lab_balanced = cv2.merge([normalized, a, b])
        balanced = cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)
        
        # Apply CLAHE for final enhancement
        balanced = self._enhance_contrast(balanced)
        
        return balanced
    
    def get_brightness_info(self, frame):
        """Get brightness statistics"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        return {
            'mean': np.mean(gray),
            'median': np.median(gray),
            'std': np.std(gray),
            'min': np.min(gray),
            'max': np.max(gray)
        }