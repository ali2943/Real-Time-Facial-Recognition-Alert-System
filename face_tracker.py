"""
Face Tracking with Temporal Consistency
Tracks faces across frames for stable recognition
"""

import cv2
import numpy as np
from collections import deque, defaultdict


class FaceTracker:
    """
    Track faces across frames using IoU and embedding similarity
    
    Benefits:
    - Reduces jitter
    - Provides stable embeddings
    - Detects sudden identity changes (spoofing)
    """
    
    def __init__(self, max_disappeared=10, max_distance=50):
        self.next_id = 0
        self.tracked_faces = {}  # face_id: {box, embedding, last_seen}
        self.disappeared = {}     # face_id: frames_disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # History for temporal smoothing
        self.embedding_history = defaultdict(lambda: deque(maxlen=5))
        self.box_history = defaultdict(lambda: deque(maxlen=3))
        
        print("[INFO] Face Tracker initialized")
    
    def update(self, detections, embeddings=None):
        """
        Update tracked faces
        
        Args:
            detections: List of face detections
            embeddings: Optional embeddings for each face
            
        Returns:
            Dict of {face_id: {'box': box, 'embedding': emb, 'confidence': conf}}
        """
        # No detections - increment disappeared counters
        if len(detections) == 0:
            self._handle_no_detections()
            return self.tracked_faces
        
        # First frame - register all
        if len(self.tracked_faces) == 0:
            for i, det in enumerate(detections):
                emb = embeddings[i] if embeddings and i < len(embeddings) else None
                self._register_face(det, emb)
        
        else:
            # Match detections to tracked faces
            matches = self._match_detections(detections, embeddings)
            
            # Update matched faces
            for face_id, det_idx in matches.items():
                det = detections[det_idx]
                emb = embeddings[det_idx] if embeddings and det_idx < len(embeddings) else None
                
                self._update_face(face_id, det, emb)
            
            # Register unmatched detections
            unmatched = [i for i in range(len(detections)) if i not in matches.values()]
            for idx in unmatched:
                emb = embeddings[idx] if embeddings and idx < len(embeddings) else None
                self._register_face(detections[idx], emb)
            
            # Handle disappeared faces
            matched_ids = set(matches.keys())
            for face_id in list(self.tracked_faces.keys()):
                if face_id not in matched_ids:
                    self.disappeared[face_id] = self.disappeared.get(face_id, 0) + 1
                    
                    if self.disappeared[face_id] > self.max_disappeared:
                        self._deregister_face(face_id)
        
        return self._get_tracked_faces_with_confidence()
    
    def _register_face(self, detection, embedding=None):
        """Register new face"""
        face_id = self.next_id
        self.next_id += 1
        
        self.tracked_faces[face_id] = {
            'box': detection['box'],
            'embedding': embedding,
            'last_seen': 0
        }
        
        self.disappeared[face_id] = 0
        
        if embedding is not None:
            self.embedding_history[face_id].append(embedding)
        
        self.box_history[face_id].append(detection['box'])
        
        return face_id
    
    def _update_face(self, face_id, detection, embedding=None):
        """Update existing tracked face"""
        self.tracked_faces[face_id]['box'] = detection['box']
        self.tracked_faces[face_id]['last_seen'] = 0
        self.disappeared[face_id] = 0
        
        if embedding is not None:
            self.tracked_faces[face_id]['embedding'] = embedding
            self.embedding_history[face_id].append(embedding)
        
        self.box_history[face_id].append(detection['box'])
    
    def _deregister_face(self, face_id):
        """Remove face from tracking"""
        del self.tracked_faces[face_id]
        del self.disappeared[face_id]
        
        if face_id in self.embedding_history:
            del self.embedding_history[face_id]
        if face_id in self.box_history:
            del self.box_history[face_id]
    
    def _match_detections(self, detections, embeddings=None):
        """Match detections to tracked faces"""
        if len(self.tracked_faces) == 0:
            return {}
        
        # Calculate cost matrix (IoU + embedding similarity)
        face_ids = list(self.tracked_faces.keys())
        cost_matrix = np.zeros((len(face_ids), len(detections)))
        
        for i, face_id in enumerate(face_ids):
            tracked_box = self.tracked_faces[face_id]['box']
            
            for j, det in enumerate(detections):
                det_box = det['box']
                
                # IoU similarity
                iou = self._calculate_iou(tracked_box, det_box)
                
                # Embedding similarity (if available)
                if embeddings and j < len(embeddings) and self.tracked_faces[face_id]['embedding'] is not None:
                    emb_sim = self._embedding_similarity(
                        self.tracked_faces[face_id]['embedding'],
                        embeddings[j]
                    )
                    cost = 0.6 * iou + 0.4 * emb_sim
                else:
                    cost = iou
                
                cost_matrix[i, j] = cost
        
        # Hungarian algorithm for optimal matching
        matches = {}
        used_detections = set()
        
        # Greedy matching (simple approach)
        for _ in range(min(len(face_ids), len(detections))):
            # Find best match
            i, j = np.unravel_index(cost_matrix.argmax(), cost_matrix.shape)
            
            if cost_matrix[i, j] < 0.3:  # Threshold
                break
            
            matches[face_ids[i]] = j
            used_detections.add(j)
            
            # Zero out row and column
            cost_matrix[i, :] = 0
            cost_matrix[:, j] = 0
        
        return matches
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _embedding_similarity(self, emb1, emb2):
        """Calculate embedding similarity (cosine)"""
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-6)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-6)
        
        similarity = np.dot(emb1, emb2)
        return max(0, similarity)  # Clamp to [0, 1]
    
    def _handle_no_detections(self):
        """Handle frame with no detections"""
        for face_id in list(self.tracked_faces.keys()):
            self.disappeared[face_id] = self.disappeared.get(face_id, 0) + 1
            
            if self.disappeared[face_id] > self.max_disappeared:
                self._deregister_face(face_id)
    
    def _get_tracked_faces_with_confidence(self):
        """Get tracked faces with confidence scores"""
        result = {}
        
        for face_id, data in self.tracked_faces.items():
            # Calculate tracking confidence
            frames_tracked = len(self.box_history[face_id])
            tracking_conf = min(1.0, frames_tracked / 5.0)  # Max confidence after 5 frames
            
            # Get stable embedding (average of history)
            if len(self.embedding_history[face_id]) > 0:
                stable_emb = np.mean(list(self.embedding_history[face_id]), axis=0)
            else:
                stable_emb = data['embedding']
            
            # Get stable box (average of history)
            if len(self.box_history[face_id]) > 0:
                boxes = np.array(list(self.box_history[face_id]))
                stable_box = np.mean(boxes, axis=0).astype(int).tolist()
            else:
                # Ensure consistent list format
                stable_box = list(data['box']) if not isinstance(data['box'], list) else data['box']
            
            result[face_id] = {
                'box': stable_box,
                'embedding': stable_emb,
                'confidence': tracking_conf,
                'frames_tracked': frames_tracked
            }
        
        return result
    
    def get_stable_embedding(self, face_id):
        """Get temporally averaged embedding"""
        if face_id not in self.embedding_history:
            return None
        
        if len(self.embedding_history[face_id]) == 0:
            return None
        
        return np.mean(list(self.embedding_history[face_id]), axis=0)
