"""
Adaptive Threshold Management
Per-user threshold calibration
"""

import numpy as np
import pickle
import os


class AdaptiveThresholdManager:
    """
    Manage per-user adaptive thresholds
    
    Each user gets custom threshold based on their enrollment variance
    """
    
    def __init__(self, database_manager):
        self.db_manager = database_manager
        self.user_thresholds = {}
        self.user_stats = {}
        
        self._load_thresholds()
        self._calculate_all_thresholds()
        
        print("[INFO] Adaptive Threshold Manager initialized")
    
    def _calculate_all_thresholds(self):
        """Calculate thresholds for all users"""
        
        users = self.db_manager.get_all_users()
        
        for user in users:
            embeddings = self.db_manager.get_user_embeddings(user)
            
            if len(embeddings) < 2:
                # Not enough data, use default
                self.user_thresholds[user] = 0.6
                print(f"[THRESHOLD] {user}: 0.6000 (default - insufficient samples)")
                continue
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    distances.append(dist)
            
            # Statistics
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            max_dist = np.max(distances)
            
            # Adaptive threshold = mean + 3*std
            # This covers 99.7% of genuine attempts
            adaptive_threshold = mean_dist + 3.0 * std_dist
            
            # Clamp to reasonable range
            adaptive_threshold = np.clip(adaptive_threshold, 0.4, 0.9)
            
            self.user_thresholds[user] = adaptive_threshold
            self.user_stats[user] = {
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'max_distance': max_dist,
                'num_samples': len(embeddings)
            }
            
            print(f"[THRESHOLD] {user}: {adaptive_threshold:.4f} "
                  f"(mean: {mean_dist:.4f}, std: {std_dist:.4f}, samples: {len(embeddings)})")
        
        self._save_thresholds()
    
    def get_threshold(self, user):
        """Get threshold for user"""
        return self.user_thresholds.get(user, 0.6)
    
    def get_stats(self, user):
        """Get statistics for user"""
        return self.user_stats.get(user, {})
    
    def update_threshold(self, user, new_match_distance):
        """
        Update threshold based on new successful match
        
        Online learning: adjust threshold based on actual usage
        """
        if user not in self.user_stats:
            return
        
        stats = self.user_stats[user]
        
        # Update running statistics using exponential moving average
        alpha = 0.1  # Learning rate
        
        stats['mean_distance'] = (1 - alpha) * stats['mean_distance'] + alpha * new_match_distance
        stats['max_distance'] = max(stats['max_distance'], new_match_distance)
        
        # Recalculate threshold
        new_threshold = stats['mean_distance'] + 3.0 * stats['std_distance']
        new_threshold = np.clip(new_threshold, 0.4, 0.9)
        
        self.user_thresholds[user] = new_threshold
        
        self._save_thresholds()
    
    def _save_thresholds(self):
        """Save thresholds to file"""
        os.makedirs('database', exist_ok=True)
        
        data = {
            'thresholds': self.user_thresholds,
            'stats': self.user_stats
        }
        
        try:
            with open('database/adaptive_thresholds.pkl', 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"[WARNING] Failed to save thresholds: {e}")
    
    def _load_thresholds(self):
        """Load thresholds from file"""
        if os.path.exists('database/adaptive_thresholds.pkl'):
            try:
                with open('database/adaptive_thresholds.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.user_thresholds = data.get('thresholds', {})
                    self.user_stats = data.get('stats', {})
                print(f"[INFO] Loaded adaptive thresholds for {len(self.user_thresholds)} users")
            except Exception as e:
                print(f"[WARNING] Failed to load thresholds: {e}")