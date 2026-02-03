"""
Multi-Sample Embedding Generation
Generate multiple embeddings and fuse them for robustness
"""

import cv2
import numpy as np


class MultiSampleEmbedder:
    """
    Generate multiple embeddings from single face
    Then average for robust embedding
    """
    
    def __init__(self, recognizer):
        self.recognizer = recognizer
        print("[INFO] Multi-Sample Embedder initialized")
    
    def generate_robust_embedding(self, face_img, num_samples=3):
        """
        Generate robust embedding from multiple samples
        
        Args:
            face_img: Input face
            num_samples: Number of variations to generate (1-5)
            
        Returns:
            Averaged embedding
        """
        embeddings = []
        
        # Sample 1: Original (center crop)
        emb1 = self.recognizer.get_embedding(face_img)
        embeddings.append(emb1)
        
        if num_samples >= 2:
            # Sample 2: Slight rotation (+3 degrees)
            rotated = self._rotate_image(face_img, 3)
            emb2 = self.recognizer.get_embedding(rotated)
            embeddings.append(emb2)
        
        if num_samples >= 3:
            # Sample 3: Slight rotation (-3 degrees)
            rotated = self._rotate_image(face_img, -3)
            emb3 = self.recognizer.get_embedding(rotated)
            embeddings.append(emb3)
        
        if num_samples >= 4:
            # Sample 4: Brightness adjusted (+20%)
            brightened = self._adjust_brightness(face_img, 1.2)
            emb4 = self.recognizer.get_embedding(brightened)
            embeddings.append(emb4)
        
        if num_samples >= 5:
            # Sample 5: Brightness adjusted (-20%)
            darkened = self._adjust_brightness(face_img, 0.8)
            emb5 = self.recognizer.get_embedding(darkened)
            embeddings.append(emb5)
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Renormalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return avg_embedding
    
    def _rotate_image(self, img, angle):
        """Rotate image by angle"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), 
                                 borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def _adjust_brightness(self, img, factor):
        """Adjust brightness by factor"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        v = np.clip(v * factor, 0, 255).astype(np.uint8)
        
        hsv = cv2.merge([h, s, v])
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return adjusted