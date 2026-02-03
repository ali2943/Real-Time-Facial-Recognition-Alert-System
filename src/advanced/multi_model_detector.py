"""
Multi-Model Face Detection
Combines multiple detectors for robust detection
"""

import cv2
import numpy as np
from src.core.face_detector import FaceDetector


class MultiModelFaceDetector:
    """
    Ensemble face detector using multiple models
    
    Models:
    1. MTCNN (primary)
    2. YuNet (fast fallback)
    3. Haar Cascade (emergency fallback)
    """
    
    def __init__(self):
        # Primary: MTCNN
        self.mtcnn = FaceDetector()
        
        # Secondary: YuNet (OpenCV DNN)
        try:
            self.yunet = cv2.FaceDetectorYN.create(
                "face_detection_yunet_2023mar.onnx",
                "", (320, 320), 0.6, 0.3, 5000
            )
            self.has_yunet = True
        except (FileNotFoundError, cv2.error, AttributeError) as e:
            self.has_yunet = False
            print("[WARNING] YuNet not available")
        
        # Tertiary: Haar Cascade
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("[INFO] Multi-Model Face Detector initialized")
    
    def detect_faces(self, frame, mode='cascade'):
        """
        Detect faces using ensemble approach
        
        Modes:
        - 'cascade': Try models in order (fast)
        - 'ensemble': Use all, vote on results (accurate)
        
        Args:
            frame: Input frame
            mode: Detection mode
            
        Returns:
            List of face detections with confidence
        """
        if mode == 'cascade':
            return self._cascade_detect(frame)
        else:
            return self._ensemble_detect(frame)
    
    def _cascade_detect(self, frame):
        """Try detectors in order until success"""
        # Try MTCNN first (best quality)
        detections = self.mtcnn.detect_faces(frame)
        if len(detections) > 0:
            return detections
        
        # Try YuNet
        if self.has_yunet:
            yunet_dets = self._yunet_detect(frame)
            if len(yunet_dets) > 0:
                return yunet_dets
        
        # Try Haar cascade (last resort)
        haar_dets = self._haar_detect(frame)
        return haar_dets
    
    def _ensemble_detect(self, frame):
        """Combine all detectors with voting"""
        all_detections = []
        
        # MTCNN
        mtcnn_dets = self.mtcnn.detect_faces(frame)
        all_detections.extend([(d, 'mtcnn') for d in mtcnn_dets])
        
        # YuNet
        if self.has_yunet:
            yunet_dets = self._yunet_detect(frame)
            all_detections.extend([(d, 'yunet') for d in yunet_dets])
        
        # Haar
        haar_dets = self._haar_detect(frame)
        all_detections.extend([(d, 'haar') for d in haar_dets])
        
        # Cluster overlapping detections
        clustered = self._cluster_detections(all_detections)
        
        return clustered
    
    def _yunet_detect(self, frame):
        """YuNet detection"""
        h, w = frame.shape[:2]
        self.yunet.setInputSize((w, h))
        
        _, faces = self.yunet.detect(frame)
        
        detections = []
        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                conf = face[14]
                
                det = {
                    'box': [x, y, w, h],
                    'confidence': float(conf),
                    'keypoints': {
                        'right_eye': tuple(face[4:6]),
                        'left_eye': tuple(face[6:8]),
                        'nose': tuple(face[8:10]),
                        'mouth_right': tuple(face[10:12]),
                        'mouth_left': tuple(face[12:14]),
                    }
                }
                detections.append(det)
        
        return detections
    
    def _haar_detect(self, frame):
        """Haar cascade detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in faces:
            det = {
                'box': [x, y, w, h],
                'confidence': 0.7,  # Haar doesn't provide confidence
                'keypoints': None
            }
            detections.append(det)
        
        return detections
    
    def _cluster_detections(self, detections):
        """Cluster overlapping detections and merge"""
        if len(detections) == 0:
            return []
        
        # Simple clustering by IoU
        boxes = [d[0]['box'] for d in detections]
        
        # Non-maximum suppression
        final_detections = []
        used = set()
        
        for i, (det, source) in enumerate(detections):
            if i in used:
                continue
            
            # Find overlapping detections
            cluster = [det]
            used.add(i)
            
            for j, (det2, source2) in enumerate(detections):
                if j <= i or j in used:
                    continue
                
                if self._iou(det['box'], det2['box']) > 0.5:
                    cluster.append(det2)
                    used.add(j)
            
            # Merge cluster
            merged = self._merge_detections(cluster)
            merged['confidence'] *= (1.0 + 0.2 * (len(cluster) - 1))  # Boost if multiple agree
            merged['confidence'] = min(1.0, merged['confidence'])
            
            final_detections.append(merged)
        
        return final_detections
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _merge_detections(self, detections):
        """Average multiple detections"""
        boxes = np.array([d['box'] for d in detections])
        avg_box = np.mean(boxes, axis=0).astype(int).tolist()
        
        avg_conf = np.mean([d['confidence'] for d in detections])
        
        # Use landmarks from best detection
        best_det = max(detections, key=lambda d: d['confidence'])
        
        return {
            'box': avg_box,
            'confidence': avg_conf,
            'keypoints': best_det.get('keypoints')
        }
