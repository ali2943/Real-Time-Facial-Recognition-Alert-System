"""
Enhanced Database Manager with Advanced Matching
Implements KNN matching, adaptive thresholds, and confidence scoring
"""

import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import config


class EnhancedDatabaseManager:
    """Enhanced database manager with advanced matching strategies"""
    
    def __init__(self):
        """Initialize enhanced database manager"""
        self.database_dir = config.DATABASE_DIR
        self.embeddings_file = os.path.join(self.database_dir, config.EMBEDDINGS_FILE)
        
        # Create database directory if it doesn't exist
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir)
            print(f"[INFO] Created database directory: {self.database_dir}")
        
        # Load existing embeddings or create new database
        self.embeddings_db = self.load_database()
        
        # KNN model cache
        self.knn_model = None
        self.embedding_to_name = []  # Maps index to user name
        
        if len(self.embeddings_db) > 0:
            self._build_knn_model()
        
        print(f"[INFO] Enhanced Database Manager initialized with {len(self.embeddings_db)} authorized users")
    
    def load_database(self):
        """
        Load embeddings database from file
        
        Returns:
            Dictionary of {name: [embeddings]}
        """
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                embeddings_db = pickle.load(f)
            print(f"[INFO] Loaded embeddings from {self.embeddings_file}")
            return embeddings_db
        else:
            print("[INFO] No existing database found, creating new one")
            return {}
    
    def save_database(self):
        """Save embeddings database to file"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings_db, f)
        print(f"[INFO] Database saved to {self.embeddings_file}")
    
    def add_user(self, name, embedding):
        """
        Add a new user or update existing user embeddings
        
        Args:
            name: User name
            embedding: Face embedding vector
        """
        if name not in self.embeddings_db:
            self.embeddings_db[name] = []
        
        self.embeddings_db[name].append(embedding)
        self.save_database()
        
        # Rebuild KNN model
        self._build_knn_model()
        
        print(f"[INFO] Added embedding for user: {name}")
    
    def remove_user(self, name):
        """
        Remove a user from database
        
        Args:
            name: User name to remove
        """
        if name in self.embeddings_db:
            del self.embeddings_db[name]
            self.save_database()
            
            # Rebuild KNN model
            self._build_knn_model()
            
            print(f"[INFO] Removed user: {name}")
        else:
            print(f"[WARNING] User not found: {name}")
    
    def get_all_users(self):
        """
        Get list of all authorized users
        
        Returns:
            List of user names
        """
        return list(self.embeddings_db.keys())
    
    def get_user_embeddings(self, name):
        """
        Get embeddings for a specific user
        
        Args:
            name: User name
            
        Returns:
            List of embeddings for the user
        """
        return self.embeddings_db.get(name, [])
    
    def _build_knn_model(self):
        """Build KNN model from all embeddings"""
        if len(self.embeddings_db) == 0:
            self.knn_model = None
            self.embedding_to_name = []
            return
        
        # Flatten all embeddings with labels
        all_embeddings = []
        self.embedding_to_name = []
        
        for name, embeddings in self.embeddings_db.items():
            for embedding in embeddings:
                all_embeddings.append(embedding)
                self.embedding_to_name.append(name)
        
        # Build KNN model
        all_embeddings = np.array(all_embeddings)
        self.knn_model = NearestNeighbors(
            n_neighbors=min(config.KNN_K, len(all_embeddings)),
            metric='euclidean'
        )
        self.knn_model.fit(all_embeddings)
    
    def find_knn_match(self, embedding, k=None):
        """
        Find k nearest neighbors and use voting
        
        Args:
            embedding: Query embedding
            k: Number of neighbors (default from config)
            
        Returns:
            Tuple of (matched_name, avg_distance, vote_count)
        """
        if k is None:
            k = config.KNN_K
        
        if self.knn_model is None:
            return None, float('inf'), 0
        
        # Find k nearest neighbors
        k = min(k, len(self.embedding_to_name))
        distances, indices = self.knn_model.kneighbors([embedding], n_neighbors=k)
        
        # Count votes for each name
        vote_counts = {}
        distance_sums = {}
        
        for i, idx in enumerate(indices[0]):
            name = self.embedding_to_name[idx]
            distance = distances[0][i]
            
            if name not in vote_counts:
                vote_counts[name] = 0
                distance_sums[name] = 0
            
            vote_counts[name] += 1
            distance_sums[name] += distance
        
        # Find name with most votes
        best_name = max(vote_counts.items(), key=lambda x: x[1])[0]
        best_vote_count = vote_counts[best_name]
        avg_distance = distance_sums[best_name] / best_vote_count
        
        return best_name, avg_distance, best_vote_count
    
    def calculate_adaptive_threshold(self, user_name):
        """
        Calculate personalized threshold based on intra-class variance
        
        Args:
            user_name: User name
            
        Returns:
            Adaptive threshold for this user
        """
        embeddings = self.get_user_embeddings(user_name)
        
        if len(embeddings) < 2:
            # Not enough samples, use default threshold
            return config.RECOGNITION_THRESHOLD
        
        # Calculate pairwise distances within user's samples
        embeddings = np.array(embeddings)
        intra_distances = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = np.linalg.norm(embeddings[i] - embeddings[j])
                intra_distances.append(distance)
        
        # Set threshold as mean + 2 * std of intra-class distances
        # This represents approximately 95% confidence interval assuming normal distribution
        # Allows for natural variation within user's samples while rejecting outliers
        mean_intra = np.mean(intra_distances)
        std_intra = np.std(intra_distances)
        
        adaptive_threshold = mean_intra + 2 * std_intra
        
        # Clamp to reasonable range
        min_threshold = config.RECOGNITION_THRESHOLD * 0.5
        max_threshold = config.RECOGNITION_THRESHOLD * 1.5
        
        adaptive_threshold = np.clip(adaptive_threshold, min_threshold, max_threshold)
        
        return adaptive_threshold
    
    def get_match_confidence(self, distances, matched_name):
        """
        Calculate confidence score based on distance and separation
        
        Args:
            distances: List of distances to k nearest neighbors
            matched_name: Name of matched user
            
        Returns:
            Confidence score (0-1)
        """
        if len(distances) == 0:
            return 0.0
        
        # Best distance (to matched user)
        best_distance = distances[0]
        
        # Convert distance to similarity (0 = identical, higher = different)
        # For normalized embeddings, distance typically ranges 0-2
        similarity = max(0, 1.0 - (best_distance / 2.0))
        
        # Calculate separation from second-best match (if different user)
        separation_bonus = 0.0
        if len(distances) > 1:
            second_distance = distances[1]
            separation = second_distance - best_distance
            separation_bonus = min(0.2, separation / 2.0)  # Up to 20% bonus
        
        # Final confidence
        confidence = min(1.0, similarity + separation_bonus)
        
        return confidence
    
    def find_match_advanced(self, query_embedding, recognizer):
        """
        Advanced matching with KNN, adaptive thresholds, and confidence scoring
        
        Args:
            query_embedding: Face embedding to match
            recognizer: Face recognition model instance
            
        Returns:
            Tuple of (user_name, distance, confidence) or (None, distance, 0.0) if no match
        """
        if len(self.embeddings_db) == 0:
            return None, float('inf'), 0.0
        
        if config.USE_KNN_MATCHING:
            # Use KNN matching with voting
            matched_name, avg_distance, vote_count = self.find_knn_match(query_embedding)
            
            # Get threshold
            if config.ADAPTIVE_THRESHOLD_PER_USER:
                threshold = self.calculate_adaptive_threshold(matched_name)
            else:
                threshold = config.RECOGNITION_THRESHOLD
            
            # Check if match is within threshold
            if avg_distance < threshold:
                # Calculate confidence
                # Find all distances for confidence calculation
                k = min(config.KNN_K, len(self.embedding_to_name))
                distances, _ = self.knn_model.kneighbors([query_embedding], n_neighbors=k)
                confidence = self.get_match_confidence(distances[0], matched_name)
                
                return matched_name, avg_distance, confidence
            else:
                return None, avg_distance, 0.0
        
        else:
            # Use traditional closest match
            best_match = None
            best_distance = float('inf')
            
            for name, stored_embeddings in self.embeddings_db.items():
                for stored_embedding in stored_embeddings:
                    distance = recognizer.compare_embeddings(query_embedding, stored_embedding)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name
            
            # Get threshold
            if config.ADAPTIVE_THRESHOLD_PER_USER and best_match:
                threshold = self.calculate_adaptive_threshold(best_match)
            else:
                threshold = config.RECOGNITION_THRESHOLD
            
            # Check if best match is within threshold
            if best_distance < threshold:
                # Simple confidence based on distance
                confidence = max(0, 1.0 - (best_distance / threshold))
                return best_match, best_distance, confidence
            else:
                return None, best_distance, 0.0
    
    def find_match(self, embedding, face_recognizer):
        """
        Find matching user for a given embedding (backward compatible)
        
        Args:
            embedding: Face embedding to match
            face_recognizer: FaceRecognitionModel instance
            
        Returns:
            Tuple of (user_name, distance) or (None, distance) if no match
        """
        # Use advanced matching but return only name and distance for compatibility
        matched_name, distance, confidence = self.find_match_advanced(embedding, face_recognizer)
        return matched_name, distance
