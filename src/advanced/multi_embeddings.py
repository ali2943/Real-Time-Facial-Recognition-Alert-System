"""
Multi-Face Embeddings Module
Generate embeddings using multiple models for robustness
"""

import numpy as np


class MultiEmbeddingGenerator:
    """
    Generate face embeddings using multiple models
    
    Combines embeddings from:
    1. FaceNet (primary)
    2. InsightFace (if available)
    3. Ensemble averaging
    """
    
    def __init__(self, face_recognition_model=None, insightface_model=None):
        """
        Initialize multi-embedding generator
        
        Args:
            face_recognition_model: FaceNet model instance
            insightface_model: InsightFace model instance (optional)
        """
        self.facenet = face_recognition_model
        self.insightface = insightface_model
        
        print("[INFO] Multi-Embedding Generator initialized")
        if self.facenet:
            print("[INFO] - FaceNet model: Active")
        if self.insightface:
            print("[INFO] - InsightFace model: Active")
    
    def generate_embedding(self, face_img, mode='ensemble'):
        """
        Generate face embedding
        
        Modes:
        - 'facenet': Use FaceNet only
        - 'insightface': Use InsightFace only
        - 'ensemble': Combine both (if available)
        
        Args:
            face_img: Aligned and preprocessed face image
            mode: Embedding mode
            
        Returns:
            Face embedding vector
        """
        if mode == 'facenet' or not self.insightface:
            return self._generate_facenet_embedding(face_img)
        elif mode == 'insightface' and self.insightface:
            return self._generate_insightface_embedding(face_img)
        elif mode == 'ensemble':
            return self._generate_ensemble_embedding(face_img)
        else:
            return self._generate_facenet_embedding(face_img)
    
    def _generate_facenet_embedding(self, face_img):
        """Generate embedding using FaceNet"""
        if not self.facenet:
            return None
        
        try:
            embedding = self.facenet.get_embedding(face_img)
            return embedding
        except Exception as e:
            print(f"[WARNING] FaceNet embedding failed: {e}")
            return None
    
    def _generate_insightface_embedding(self, face_img):
        """Generate embedding using InsightFace"""
        if not self.insightface:
            return None
        
        try:
            embedding = self.insightface.get_embedding(face_img)
            return embedding
        except Exception as e:
            print(f"[WARNING] InsightFace embedding failed: {e}")
            return None
    
    def _generate_ensemble_embedding(self, face_img):
        """
        Generate ensemble embedding by combining models
        
        Args:
            face_img: Face image
            
        Returns:
            Combined embedding
        """
        embeddings = []
        
        # Get FaceNet embedding
        if self.facenet:
            facenet_emb = self._generate_facenet_embedding(face_img)
            if facenet_emb is not None:
                embeddings.append(facenet_emb)
        
        # Get InsightFace embedding
        if self.insightface:
            insight_emb = self._generate_insightface_embedding(face_img)
            if insight_emb is not None:
                embeddings.append(insight_emb)
        
        if len(embeddings) == 0:
            return None
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            # Concatenate embeddings
            combined = np.concatenate(embeddings)
            
            # Normalize
            combined = combined / (np.linalg.norm(combined) + 1e-6)
            
            return combined
    
    def generate_multiple_embeddings(self, face_img):
        """
        Generate embeddings from all available models
        
        Args:
            face_img: Face image
            
        Returns:
            Dictionary of {model_name: embedding}
        """
        embeddings = {}
        
        if self.facenet:
            facenet_emb = self._generate_facenet_embedding(face_img)
            if facenet_emb is not None:
                embeddings['facenet'] = facenet_emb
        
        if self.insightface:
            insight_emb = self._generate_insightface_embedding(face_img)
            if insight_emb is not None:
                embeddings['insightface'] = insight_emb
        
        return embeddings
    
    def calculate_similarity(self, emb1, emb2, metric='cosine'):
        """
        Calculate similarity between two embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        if metric == 'cosine':
            # Cosine similarity
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-6)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-6)
            
            similarity = np.dot(emb1_norm, emb2_norm)
            return similarity
        
        elif metric == 'euclidean':
            # Euclidean distance (convert to similarity)
            distance = np.linalg.norm(emb1 - emb2)
            
            # Convert distance to similarity (0 = same, higher = different)
            # Normalize to [0, 1] range
            similarity = 1.0 / (1.0 + distance)
            return similarity
        
        else:
            return 0.0
