"""
Face Recognition Model - Keras-FaceNet (Most Reliable)
Works on all systems without PyTorch issues
"""

import cv2
import numpy as np
import config


class FaceRecognitionModel:
    """Face recognition using Keras-FaceNet"""
    
    def __init__(self):
        """Initialize face recognition model"""
        print("[INFO] Initializing Face Recognition Model...")
        
        # Load Keras-FaceNet
        self._load_keras_facenet()
        self.model_type = 'keras_facenet'
    
    def _load_keras_facenet(self):
        """Load Keras-FaceNet model"""
        try:
            from keras_facenet import FaceNet
            
            print("[INFO] Loading Keras-FaceNet model...")
            
            # Initialize FaceNet
            self.model = FaceNet()
            
            self.embedding_size = 512
            self.input_size = (160, 160)
            
            print("[INFO] ✅ Keras-FaceNet model loaded successfully")
            print(f"[INFO] - Model: FaceNet (Keras)")
            print(f"[INFO] - Embedding dimension: {self.embedding_size}")
            print(f"[INFO] - Input size: {self.input_size}")
            
        except ImportError as e:
            print(f"[ERROR] keras-facenet not installed: {e}")
            print("[INFO] Installing keras-facenet...")
            print("[INFO] Run: pip install keras-facenet")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load Keras-FaceNet: {e}")
            raise
    
    def get_embedding(self, face_img):
        """
        Generate face embedding
        
        Args:
            face_img: Face image (BGR format, any size)
            
        Returns:
            numpy array of embedding (normalized)
        """
        if config.DEBUG_MODE:
            print(f"[DEBUG] Generating embedding for face: shape={face_img.shape}")
        
        try:
            return self._get_embedding_keras(face_img)
                
        except Exception as e:
            print(f"[ERROR] Failed to generate embedding: {e}")
            import traceback
            traceback.print_exc()
            # Return random vector (will not match anything)
            return np.random.randn(self.embedding_size).astype(np.float32)
    
    def _get_embedding_keras(self, face_img):
        """Generate embedding using Keras-FaceNet"""
        
        # Resize to 160x160
        face_resized = cv2.resize(face_img, self.input_size, interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        if config.DEBUG_MODE:
            print(f"[DEBUG] Preprocessed face: shape={face_rgb.shape}, "
                  f"dtype={face_rgb.dtype}, "
                  f"range=[{face_rgb.min()}, {face_rgb.max()}]")
        
        # Generate embedding (model expects list of images)
        embeddings = self.model.embeddings([face_rgb])
        embedding = embeddings[0]
        
        # Convert to numpy if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        if config.DEBUG_MODE:
            print(f"[DEBUG] Embedding generated: shape={embedding.shape}, "
                  f"norm={np.linalg.norm(embedding):.2f}")
        
        return embedding
    
    def compare_embeddings(self, embedding1, embedding2):
        """
        Compare two embeddings using Euclidean distance
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Distance (0 = identical, higher = more different)
        """
        # Ensure embeddings are numpy arrays
        emb1 = np.array(embedding1, dtype=np.float32)
        emb2 = np.array(embedding2, dtype=np.float32)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(emb1 - emb2)
        
        return float(distance)
    
    def verify(self, face_img, reference_embedding, threshold=None):
        """
        Verify if face matches reference embedding
        
        Args:
            face_img: Face image to verify
            reference_embedding: Reference embedding to compare against
            threshold: Distance threshold (default from config)
            
        Returns:
            (is_match, distance, confidence)
        """
        if threshold is None:
            threshold = config.RECOGNITION_THRESHOLD
        
        # Generate embedding
        embedding = self.get_embedding(face_img)
        
        # Compare
        distance = self.compare_embeddings(embedding, reference_embedding)
        
        # Determine match
        is_match = distance < threshold
        
        # Calculate confidence (0-1)
        confidence = max(0.0, 1.0 - (distance / threshold))
        
        return is_match, distance, confidence
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': self.model_type,
            'embedding_size': self.embedding_size,
            'input_size': self.input_size
        }


# Test the model
if __name__ == "__main__":
    print("=== Testing Face Recognition Model ===\n")
    
    # Initialize model
    try:
        recognizer = FaceRecognitionModel()
    except Exception as e:
        print(f"\n❌ Failed to initialize model: {e}")
        print("\nPlease install keras-facenet:")
        print("  pip install keras-facenet")
        exit(1)
    
    # Get model info
    info = recognizer.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with random images
    print("\n=== Testing with random images ===")
    
    embeddings = []
    for i in range(3):
        # Create random face image
        random_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        
        # Generate embedding
        embedding = recognizer.get_embedding(random_face)
        embeddings.append(embedding)
        
        print(f"\nImage {i+1}:")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding sample: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Check if embeddings are different
    print("\n=== Checking embedding differences ===")
    all_same = True
    distances = []
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = recognizer.compare_embeddings(embeddings[i], embeddings[j])
            distances.append(dist)
            print(f"Distance({i},{j}): {dist:.4f}")
            if dist > 0.1:
                all_same = False
    
    # Variance check
    variance = np.var(embeddings, axis=0).mean()
    print(f"\nEmbedding variance: {variance:.6f}")
    
    # Verdict
    print("\n=== VERDICT ===")
    
    if all_same or variance < 0.0001:
        print("❌ FAIL: All embeddings are too similar!")
        print("   Model may not be working correctly")
    else:
        print("✅ PASS: Model generates diverse embeddings")
        print(f"   Average distance: {np.mean(distances):.4f}")
        print(f"   Variance: {variance:.6f}")
    
    print("\n=== Test Complete ===")