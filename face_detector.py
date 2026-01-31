"""
Face Detection Module using MTCNN
Detects faces in images and returns bounding box coordinates
"""

import cv2
import numpy as np
from mtcnn import MTCNN
import config


class FaceDetector:
    """Face detector using MTCNN for robust face detection"""
    
    def __init__(self):
        """Initialize MTCNN detector"""
        self.detector = MTCNN()
        print("[INFO] MTCNN Face Detector initialized")
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of dictionaries containing face information:
            - box: [x, y, width, height]
            - confidence: detection confidence
            - keypoints: facial landmarks
        """
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector.detect_faces(rgb_frame)
        
        # Filter by confidence and size
        valid_detections = []
        for detection in detections:
            confidence = detection['confidence']
            box = detection['box']
            
            # Check confidence threshold
            if confidence < config.FACE_DETECTION_CONFIDENCE:
                continue
            
            # Check minimum face size
            if box[2] < config.MIN_FACE_SIZE or box[3] < config.MIN_FACE_SIZE:
                continue
            
            valid_detections.append(detection)
        
        return valid_detections
    
    def extract_face(self, frame, box, margin=20):
        """
        Extract face region from frame with margin
        
        Args:
            frame: Original frame
            box: [x, y, width, height]
            margin: Pixels to add around face
            
        Returns:
            Extracted face image
        """
        x, y, w, h = box
        
        # Add margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        # Extract face
        face = frame[y1:y2, x1:x2]
        
        return face
