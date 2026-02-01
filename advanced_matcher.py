"""
Advanced Matching Module
Sophisticated matching algorithms for face recognition
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class AdvancedMatcher:
    """
    Advanced face matching using multiple strategies
    
    Features:
    - Multi-metric matching (cosine + euclidean)
    - Adaptive thresholding
    - Confidence calibration
    - Outlier rejection
    """
    
    def __init__(self, threshold=0.6):
        """
        Initialize matcher
        
        Args:
            threshold: Base matching threshold
        """
        self.threshold = threshold
        self.cosine_weight = 0.6
        self.euclidean_weight = 0.4
        
        # Confidence calibration parameters
        self.MAX_CONFIDENCE_BOOST = 0.2
        self.GAP_MULTIPLIER = 2
        
        print("[INFO] Advanced Matcher initialized")
    
    def match_embedding(self, query_emb, database_embeddings, names, method='hybrid'):
        """
        Match query embedding against database
        
        Args:
            query_emb: Query face embedding
            database_embeddings: List of database embeddings
            names: List of names corresponding to embeddings
            method: Matching method ('cosine', 'euclidean', 'hybrid')
            
        Returns:
            Tuple of (best_match_name, confidence, all_similarities)
        """
        if query_emb is None or len(database_embeddings) == 0:
            return None, 0.0, []
        
        # Calculate similarities
        if method == 'cosine':
            similarities = self._cosine_similarities(query_emb, database_embeddings)
        elif method == 'euclidean':
            similarities = self._euclidean_similarities(query_emb, database_embeddings)
        elif method == 'hybrid':
            similarities = self._hybrid_similarities(query_emb, database_embeddings)
        else:
            similarities = self._cosine_similarities(query_emb, database_embeddings)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_name = names[best_idx]
        
        # Apply adaptive threshold
        adaptive_threshold = self._calculate_adaptive_threshold(similarities)
        
        # Check if match is confident enough
        if best_similarity >= max(self.threshold, adaptive_threshold):
            confidence = self._calibrate_confidence(best_similarity, similarities)
            return best_name, confidence, similarities
        else:
            return None, 0.0, similarities
    
    def _cosine_similarities(self, query_emb, database_embeddings):
        """Calculate cosine similarities"""
        query_emb = query_emb.reshape(1, -1)
        db_embs = np.array(database_embeddings)
        
        # Normalize embeddings
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-6)
        db_norm = db_embs / (np.linalg.norm(db_embs, axis=1, keepdims=True) + 1e-6)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_norm, db_norm)[0]
        
        return similarities
    
    def _euclidean_similarities(self, query_emb, database_embeddings):
        """Calculate euclidean distance-based similarities"""
        query_emb = query_emb.reshape(1, -1)
        db_embs = np.array(database_embeddings)
        
        # Compute euclidean distances
        distances = euclidean_distances(query_emb, db_embs)[0]
        
        # Convert to similarities (lower distance = higher similarity)
        similarities = 1.0 / (1.0 + distances)
        
        return similarities
    
    def _hybrid_similarities(self, query_emb, database_embeddings):
        """Combine cosine and euclidean similarities"""
        cosine_sims = self._cosine_similarities(query_emb, database_embeddings)
        euclidean_sims = self._euclidean_similarities(query_emb, database_embeddings)
        
        # Weighted combination
        hybrid_sims = (self.cosine_weight * cosine_sims + 
                      self.euclidean_weight * euclidean_sims)
        
        return hybrid_sims
    
    def _calculate_adaptive_threshold(self, similarities):
        """
        Calculate adaptive threshold based on similarity distribution
        
        Args:
            similarities: Array of similarity scores
            
        Returns:
            Adaptive threshold
        """
        if len(similarities) < 2:
            return self.threshold
        
        # Use mean + std as adaptive threshold
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Adaptive threshold: mean + 0.5 * std
        # This ensures the best match is significantly better than average
        adaptive_threshold = mean_sim + 0.5 * std_sim
        
        return min(adaptive_threshold, 0.9)  # Cap at 0.9
    
    def _calibrate_confidence(self, best_similarity, all_similarities):
        """
        Calibrate confidence based on separation from other matches
        
        Args:
            best_similarity: Best match similarity
            all_similarities: All similarity scores
            
        Returns:
            Calibrated confidence score
        """
        if len(all_similarities) < 2:
            return best_similarity
        
        # Sort similarities
        sorted_sims = np.sort(all_similarities)[::-1]
        
        # Gap between best and second-best
        gap = sorted_sims[0] - sorted_sims[1]
        
        # Larger gap = higher confidence
        # Confidence boost: 0 to MAX_CONFIDENCE_BOOST based on gap
        confidence_boost = min(self.MAX_CONFIDENCE_BOOST, gap * self.GAP_MULTIPLIER)
        
        calibrated_conf = min(1.0, best_similarity + confidence_boost)
        
        return calibrated_conf
    
    def match_multiple(self, query_emb, database_embeddings, names, top_k=3):
        """
        Find top-k matches
        
        Args:
            query_emb: Query embedding
            database_embeddings: Database embeddings
            names: Names list
            top_k: Number of top matches to return
            
        Returns:
            List of (name, confidence) tuples
        """
        if query_emb is None or len(database_embeddings) == 0:
            return []
        
        # Calculate similarities
        similarities = self._hybrid_similarities(query_emb, database_embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.threshold:
                results.append((names[idx], similarities[idx]))
        
        return results
    
    def set_threshold(self, new_threshold):
        """Update matching threshold"""
        self.threshold = new_threshold
        print(f"[INFO] Matching threshold updated to {new_threshold}")
