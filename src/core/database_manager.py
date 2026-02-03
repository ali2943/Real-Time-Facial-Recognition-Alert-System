"""
Database Manager for storing and retrieving authorized user face embeddings
"""

import os
import pickle
import numpy as np
from config import config


class DatabaseManager:
    """Manages the database of authorized user face embeddings"""
    
    def __init__(self):
        """Initialize database manager"""
        self.database_dir = config.DATABASE_DIR
        self.embeddings_file = os.path.join(self.database_dir, config.EMBEDDINGS_FILE)
        
        # Create database directory if it doesn't exist
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir)
            print(f"[INFO] Created database directory: {self.database_dir}")
        
        # Load existing embeddings or create new database
        self.embeddings_db = self.load_database()
        print(f"[INFO] Database initialized with {len(self.embeddings_db)} authorized users")
    
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
    
    def find_match(self, embedding, face_recognizer):
        """
        Find matching user for a given embedding
        
        Args:
            embedding: Face embedding to match
            face_recognizer: FaceRecognitionModel instance
            
        Returns:
            Tuple of (user_name, distance) or (None, distance) if no match
        """
        # Check if database is empty
        if len(self.embeddings_db) == 0:
            print("[ERROR] Database is empty! No users to match against.")
            return None, float('inf')
        
        best_match = None
        best_distance = float('inf')
        
        for name, stored_embeddings in self.embeddings_db.items():
            for stored_embedding in stored_embeddings:
                distance = face_recognizer.compare_embeddings(embedding, stored_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
        
        # CRITICAL: Only return match if within threshold
        if best_distance < config.RECOGNITION_THRESHOLD:
            if config.DEBUG_MODE:
                print(f"[DEBUG] Match found: {best_match}, distance: {best_distance:.4f} < threshold: {config.RECOGNITION_THRESHOLD}")
            return best_match, best_distance
        else:
            # Distance too large - unknown person
            if config.DEBUG_MODE:
                print(f"[DEBUG] No match: Best distance {best_distance:.4f} >= threshold {config.RECOGNITION_THRESHOLD}")
            return None, best_distance
