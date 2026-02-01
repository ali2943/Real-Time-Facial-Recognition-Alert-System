"""
Face Enhancement Module
Advanced face enhancement for better recognition accuracy
"""

import cv2
import numpy as np


class FaceEnhancer:
    """
    Face enhancement using advanced techniques
    
    Features:
    - Super-resolution simulation
    - Detail enhancement
    - Illumination normalization
    - Color correction
    """
    
    def __init__(self):
        print("[INFO] Face Enhancer initialized")
    
    def enhance(self, face_img):
        """
        Apply full enhancement pipeline
        
        Args:
            face_img: Face image (BGR)
            
        Returns:
            Enhanced face image
        """
        # Step 1: Illumination normalization
        normalized = self.normalize_illumination(face_img)
        
        # Step 2: Detail enhancement
        enhanced = self.enhance_details(normalized)
        
        # Step 3: Color correction
        corrected = self.correct_colors(enhanced)
        
        return corrected
    
    def normalize_illumination(self, face_img):
        """
        Normalize illumination using DoG filter
        
        Args:
            face_img: Face image
            
        Returns:
            Illumination-normalized image
        """
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Difference of Gaussians (DoG)
        gaussian1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gaussian2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
        
        dog = gaussian1.astype(np.float32) - gaussian2.astype(np.float32)
        
        # Normalize to [0, 255]
        dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # If original was color, convert back
        if len(face_img.shape) == 3:
            # Apply to all channels
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Replace L channel with DoG result
            lab_norm = cv2.merge([dog_norm, a, b])
            result = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
            return result
        else:
            return dog_norm
    
    def enhance_details(self, face_img):
        """
        Enhance facial details using unsharp masking
        
        Args:
            face_img: Face image
            
        Returns:
            Detail-enhanced image
        """
        # Create blurred version
        blurred = cv2.GaussianBlur(face_img, (0, 0), 1.0)
        
        # Unsharp mask
        enhanced = cv2.addWeighted(face_img, 1.5, blurred, -0.5, 0)
        
        return enhanced
    
    def correct_colors(self, face_img):
        """
        Apply automatic color correction
        
        Args:
            face_img: Face image
            
        Returns:
            Color-corrected image
        """
        if len(face_img.shape) != 3:
            return face_img
        
        # Gray world assumption
        avg_b = np.mean(face_img[:, :, 0])
        avg_g = np.mean(face_img[:, :, 1])
        avg_r = np.mean(face_img[:, :, 2])
        
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        # Calculate scaling factors
        scale_b = avg_gray / (avg_b + 1e-6)
        scale_g = avg_gray / (avg_g + 1e-6)
        scale_r = avg_gray / (avg_r + 1e-6)
        
        # Apply correction
        corrected = face_img.copy().astype(np.float32)
        corrected[:, :, 0] = np.clip(corrected[:, :, 0] * scale_b, 0, 255)
        corrected[:, :, 1] = np.clip(corrected[:, :, 1] * scale_g, 0, 255)
        corrected[:, :, 2] = np.clip(corrected[:, :, 2] * scale_r, 0, 255)
        
        return corrected.astype(np.uint8)
    
    def super_resolve(self, face_img, scale=2):
        """
        Simulate super-resolution using bicubic interpolation
        
        Args:
            face_img: Face image
            scale: Upscaling factor
            
        Returns:
            Upscaled image
        """
        h, w = face_img.shape[:2]
        new_size = (w * scale, h * scale)
        
        # Use INTER_CUBIC for better quality
        upscaled = cv2.resize(face_img, new_size, interpolation=cv2.INTER_CUBIC)
        
        return upscaled
    
    def denoise_preserve_edges(self, face_img):
        """
        Denoise while preserving edges using bilateral filter
        
        Args:
            face_img: Face image
            
        Returns:
            Denoised image
        """
        # Bilateral filter preserves edges while smoothing
        denoised = cv2.bilateralFilter(face_img, 9, 75, 75)
        
        return denoised
