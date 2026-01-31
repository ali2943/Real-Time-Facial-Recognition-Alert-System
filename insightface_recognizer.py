"""
InsightFace Recognition Module using ArcFace
Provides state-of-the-art face recognition with 512-dimensional embeddings
"""

import cv2
import numpy as np


class InsightFaceRecognizer:
    """Face recognition using InsightFace (ArcFace) for mobile-phone level accuracy"""
    
    def __init__(self, model_name='buffalo_l', gpu_enabled=True):
        """
        Initialize InsightFace model
        
        Args:
            model_name: Model to use ('buffalo_l', 'buffalo_s', 'antelopev2')
            gpu_enabled: Enable GPU acceleration if available
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            print(f"[INFO] Loading InsightFace model: {model_name}...")
            
            # Determine device
            ctx_id = 0 if gpu_enabled else -1
            
            # Initialize FaceAnalysis app
            self.app = FaceAnalysis(
                name=model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu_enabled else ['CPUExecutionProvider']
            )
            
            # Prepare model with input size
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            self.embedding_size = 512  # ArcFace uses 512-d embeddings
            self.input_size = (112, 112)  # Standard aligned face size
            
            print(f"[INFO] InsightFace model loaded successfully (embedding size: {self.embedding_size})")
            
        except ImportError:
            raise ImportError(
                "InsightFace not installed. Install with: pip install insightface onnxruntime"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InsightFace: {e}")
    
    def detect_faces(self, frame):
        """
        Detect faces using InsightFace's RetinaFace detector
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of dictionaries containing face information:
            - box: [x, y, width, height]
            - confidence: detection confidence
            - keypoints: facial landmarks (left_eye, right_eye, nose, mouth_left, mouth_right)
        """
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                return []
            
            # Detect faces
            faces = self.app.get(frame)
            
            if faces is None or len(faces) == 0:
                return []
            
            # Convert to standard format
            detections = []
            for face in faces:
                # Extract bounding box
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                w = x2 - x
                h = y2 - y
                
                # Extract keypoints
                kps = face.kps.astype(int)
                keypoints = {
                    'left_eye': tuple(kps[0]),
                    'right_eye': tuple(kps[1]),
                    'nose': tuple(kps[2]),
                    'mouth_left': tuple(kps[3]),
                    'mouth_right': tuple(kps[4])
                }
                
                detection = {
                    'box': [x, y, w, h],
                    'confidence': float(face.det_score),
                    'keypoints': keypoints,
                    'face_object': face  # Store for embedding extraction
                }
                
                detections.append(detection)
            
            return detections
        
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            return []
    
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
    
    def get_embedding(self, face_img=None, face_object=None):
        """
        Generate face embedding using ArcFace
        
        Args:
            face_img: Face image (BGR format from OpenCV) - optional if face_object provided
            face_object: Face object from detect_faces - preferred method
            
        Returns:
            Face embedding vector (512-d)
        """
        try:
            # If face_object is provided (from detect_faces), use its embedding directly
            if face_object is not None:
                embedding = face_object.normed_embedding
                return embedding
            
            # Otherwise, detect and extract embedding from face image
            if face_img is not None:
                # Detect face in the cropped image
                faces = self.app.get(face_img)
                
                if faces is None or len(faces) == 0:
                    raise ValueError("No face detected in the provided image")
                
                # Use the first detected face
                embedding = faces[0].normed_embedding
                return embedding
            
            raise ValueError("Either face_img or face_object must be provided")
        
        except Exception as e:
            print(f"[ERROR] Failed to generate embedding: {e}")
            raise
    
    def compare_embeddings(self, embedding1, embedding2):
        """
        Compare two face embeddings using cosine similarity (converted to distance)
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Distance between embeddings (lower = more similar)
            For ArcFace, this is 1 - cosine_similarity
        """
        # Normalize embeddings (should already be normalized from InsightFace)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        cosine_sim = np.dot(embedding1, embedding2)
        
        # Convert to distance (0 = identical, 2 = opposite)
        distance = 1.0 - cosine_sim
        
        return distance
    
    def is_match(self, embedding1, embedding2, threshold=0.6):
        """
        Check if two embeddings match based on threshold
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Distance threshold (lower for ArcFace than FaceNet)
            
        Returns:
            True if match, False otherwise
        """
        distance = self.compare_embeddings(embedding1, embedding2)
        return distance < threshold
