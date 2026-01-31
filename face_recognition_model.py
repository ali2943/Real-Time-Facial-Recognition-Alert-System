"""
Face Recognition Module using FaceNet
Generates face embeddings for comparison
"""

import cv2
import numpy as np
from keras.models import load_model
from keras_facenet import FaceNet
import config


class FaceRecognitionModel:
    """Face recognition using FaceNet embeddings"""
    
    def __init__(self):
        """Initialize FaceNet model"""
        print("[INFO] Loading FaceNet model...")
        self.model = FaceNet()
        self.input_size = (160, 160)  # FaceNet input size
        print("[INFO] FaceNet model loaded successfully")
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for FaceNet
        
        Args:
            face_img: Face image
            
        Returns:
            Preprocessed face ready for embedding
        """
        # Resize to FaceNet input size
        face_resized = cv2.resize(face_img, self.input_size)
        
        # Normalize pixel values to [0, 1]
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Expand dimensions for batch processing
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def get_embedding(self, face_img):
        """
        Generate face embedding
        
        Args:
            face_img: Face image (BGR format from OpenCV)
            
        Returns:
            Face embedding vector
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        face_processed = self.preprocess_face(face_rgb)
        
        # Generate embedding
        embedding = self.model.embeddings(face_processed)
        
        # Return flattened embedding
        return embedding[0]
    
    def compare_embeddings(self, embedding1, embedding2):
        """
        Compare two face embeddings using Euclidean distance
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Distance between embeddings (lower = more similar)
        """
        # Calculate Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance
    
    def is_match(self, embedding1, embedding2):
        """
        Check if two embeddings match based on threshold
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            True if match, False otherwise
        """
        distance = self.compare_embeddings(embedding1, embedding2)
        return distance < config.RECOGNITION_THRESHOLD
