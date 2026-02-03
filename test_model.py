"""
Test if model generates different embeddings
"""

import cv2
import numpy as np
from face_recognition_model import FaceRecognitionModel

def test_model():
    """Test model with random images"""
    
    recognizer = FaceRecognitionModel()
    
    print("=== MODEL TEST ===\n")
    
    # Test 1: Different random images should give different embeddings
    print("Test 1: Random images...")
    
    embeddings = []
    for i in range(5):
        # Create random "face" image
        random_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        embedding = recognizer.get_embedding(random_face)
        embeddings.append(embedding)
        
        print(f"  Image {i+1}: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
        print(f"           Norm: {np.linalg.norm(embedding):.4f}")
    
    # Check if all different
    print("\nPairwise distances:")
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            print(f"  Dist({i},{j}): {dist:.4f}")
    
    # Test 2: Same image should give same embedding
    print("\nTest 2: Same image...")
    
    same_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    emb1 = recognizer.get_embedding(same_face)
    emb2 = recognizer.get_embedding(same_face)
    
    dist_same = np.linalg.norm(emb1 - emb2)
    
    print(f"  Distance between same image: {dist_same:.6f}")
    
    if dist_same > 0.0001:
        print("  ⚠️  WARNING: Same image gives different embeddings!")
        print("     Model may not be deterministic")
    else:
        print("  ✅ Same image gives same embedding")
    
    # Verdict
    print("\n=== VERDICT ===")
    
    # Check variance
    variance = np.var(embeddings, axis=0).mean()
    print(f"Embedding variance: {variance:.6f}")
    
    if variance < 0.0001:
        print("❌ FAIL: Model is NOT working properly!")
        print("   All embeddings are too similar")
        print("   Model may not be loaded or trained")
    else:
        print("✅ PASS: Model is generating diverse embeddings")


if __name__ == "__main__":
    test_model()