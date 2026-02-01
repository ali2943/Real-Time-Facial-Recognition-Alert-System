"""
Post-Processing & Verification Module
Final verification and decision logic
"""

import numpy as np
from collections import deque


class PostProcessor:
    """
    Post-processing and verification for face recognition
    
    Features:
    - Temporal consistency verification
    - Multi-factor decision making
    - Confidence aggregation
    - False positive rejection
    """
    
    def __init__(self, consistency_threshold=0.7, min_frames=3):
        """
        Initialize post-processor
        
        Args:
            consistency_threshold: Minimum consistency ratio
            min_frames: Minimum frames for verification
        """
        self.consistency_threshold = consistency_threshold
        self.min_frames = min_frames
        
        # History tracking
        self.recognition_history = deque(maxlen=10)
        
        print("[INFO] Post-Processor initialized")
    
    def verify_recognition(self, name, confidence, quality_score=None, 
                          liveness_passed=True, tracking_confidence=None):
        """
        Verify recognition result using multiple factors
        
        Args:
            name: Recognized name
            confidence: Recognition confidence
            quality_score: Optional quality score (0-100)
            liveness_passed: Liveness check result
            tracking_confidence: Optional tracking confidence
            
        Returns:
            Tuple of (verified, final_confidence, reason)
        """
        # Factor 1: Recognition confidence
        if confidence < 0.6:
            return False, confidence, "Low recognition confidence"
        
        # Factor 2: Liveness check
        if not liveness_passed:
            return False, confidence, "Liveness check failed"
        
        # Factor 3: Quality check (if provided)
        if quality_score is not None and quality_score < 60:
            return False, confidence, "Low image quality"
        
        # Factor 4: Temporal consistency (if history available)
        temporal_conf = self._check_temporal_consistency(name)
        
        # Calculate final confidence
        factors = [confidence]
        
        if quality_score is not None:
            # Normalize quality score to 0-1
            factors.append(quality_score / 100.0)
        
        if tracking_confidence is not None:
            factors.append(tracking_confidence)
        
        if temporal_conf is not None:
            factors.append(temporal_conf)
        
        # Weighted average
        final_confidence = np.mean(factors)
        
        # Final decision
        if final_confidence >= self.consistency_threshold:
            return True, final_confidence, "Verified"
        else:
            return False, final_confidence, "Inconsistent verification"
    
    def _check_temporal_consistency(self, name):
        """
        Check consistency across recent frames
        
        Args:
            name: Current recognition result
            
        Returns:
            Temporal consistency score (0-1)
        """
        if len(self.recognition_history) < self.min_frames:
            # Not enough history yet
            return None
        
        # Count occurrences of this name
        name_count = sum(1 for n in self.recognition_history if n == name)
        
        # Calculate consistency ratio
        consistency = name_count / len(self.recognition_history)
        
        return consistency
    
    def add_recognition_result(self, name):
        """
        Add recognition result to history
        
        Args:
            name: Recognized name
        """
        self.recognition_history.append(name)
    
    def clear_history(self):
        """Clear recognition history"""
        self.recognition_history.clear()
    
    def aggregate_multi_frame_results(self, results):
        """
        Aggregate results from multiple frames
        
        Args:
            results: List of (name, confidence) tuples
            
        Returns:
            Tuple of (consensus_name, aggregated_confidence)
        """
        if len(results) == 0:
            return None, 0.0
        
        # Group by name
        name_groups = {}
        for name, conf in results:
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(conf)
        
        # Find most frequent name with highest average confidence
        best_name = None
        best_score = 0.0
        
        for name, confidences in name_groups.items():
            # Score = frequency * average confidence
            frequency = len(confidences) / len(results)
            avg_confidence = np.mean(confidences)
            score = frequency * avg_confidence
            
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_name:
            avg_conf = np.mean(name_groups[best_name])
            return best_name, avg_conf
        
        return None, 0.0
    
    def detect_spoofing_attempt(self, embeddings_history):
        """
        Detect potential spoofing by analyzing embedding stability
        
        Args:
            embeddings_history: List of recent embeddings
            
        Returns:
            Tuple of (is_spoofing, reason)
        """
        if len(embeddings_history) < 3:
            return False, "Insufficient history"
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings_history) - 1):
            emb1 = embeddings_history[i]
            emb2 = embeddings_history[i + 1]
            
            # Cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6)
            similarities.append(sim)
        
        # Check for unusual patterns
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        # Too stable = possible photo
        if avg_similarity > 0.99 and std_similarity < 0.01:
            return True, "Embeddings too stable (possible photo)"
        
        # Too unstable = possible manipulation
        if std_similarity > 0.3:
            return True, "Embeddings too unstable (possible manipulation)"
        
        return False, "Normal variation"
    
    def calculate_overall_score(self, recognition_conf, quality_score, 
                               liveness_conf, tracking_conf):
        """
        Calculate overall system confidence score
        
        Args:
            recognition_conf: Recognition confidence (0-1)
            quality_score: Quality score (0-100)
            liveness_conf: Liveness confidence (0-1)
            tracking_conf: Tracking confidence (0-1)
            
        Returns:
            Overall confidence score (0-100)
        """
        # Weights
        weights = {
            'recognition': 0.4,
            'quality': 0.25,
            'liveness': 0.2,
            'tracking': 0.15
        }
        
        # Normalize quality score
        quality_norm = quality_score / 100.0 if quality_score else 0.0
        
        # Calculate weighted score
        overall = (weights['recognition'] * recognition_conf +
                  weights['quality'] * quality_norm +
                  weights['liveness'] * liveness_conf +
                  weights['tracking'] * tracking_conf)
        
        # Convert to 0-100 scale
        overall_score = overall * 100.0
        
        return round(overall_score, 2)
