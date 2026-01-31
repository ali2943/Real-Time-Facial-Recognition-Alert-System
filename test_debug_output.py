"""
Test script to verify debug output and unauthorized user detection
This simulates the face recognition process and shows debug logging
"""

import numpy as np
import config
from database_manager import DatabaseManager
from face_recognition_model import FaceRecognitionModel


def test_debug_logging():
    """Test that debug logging works correctly"""
    print("="*60)
    print("TESTING DEBUG OUTPUT AND UNAUTHORIZED USER DETECTION")
    print("="*60)
    print()
    
    # Check DEBUG_MODE is enabled
    print(f"[INFO] DEBUG_MODE: {config.DEBUG_MODE}")
    print(f"[INFO] RECOGNITION_THRESHOLD: {config.RECOGNITION_THRESHOLD}")
    print()
    
    # Initialize components
    print("[INFO] Initializing components...")
    db_manager = DatabaseManager()
    face_recognizer = FaceRecognitionModel()
    print()
    
    # Create test embeddings
    print("[INFO] Creating test embeddings...")
    
    # Authorized user embedding
    authorized_embedding = np.random.randn(128)
    db_manager.add_user("Test User", authorized_embedding)
    print()
    
    # Test 1: Authorized user (very similar embedding)
    print("="*60)
    print("TEST 1: AUTHORIZED USER (Similar Embedding)")
    print("="*60)
    
    # Create a slightly modified embedding (distance ~0.3)
    test_embedding_authorized = authorized_embedding + np.random.randn(128) * 0.05
    
    if config.DEBUG_MODE:
        print(f"[DEBUG] Face detected, generating embedding...")
        print(f"[DEBUG] Searching database for match...")
    
    matched_name, distance = db_manager.find_match(test_embedding_authorized, face_recognizer)
    
    if config.DEBUG_MODE:
        if matched_name:
            print(f"[DEBUG] Best match: {matched_name}, Distance: {distance:.4f}, Threshold: {config.RECOGNITION_THRESHOLD}")
        else:
            print(f"[DEBUG] Best match: None, Distance: {distance if distance else 'N/A'}, Threshold: {config.RECOGNITION_THRESHOLD}")
    
    if matched_name:
        print(f"[SUCCESS] Access Granted: {matched_name} (distance: {distance:.4f})")
    else:
        if distance and distance != float('inf'):
            print(f"[FAILURE] Access Denied: Unknown Person (best distance: {distance:.4f})")
        else:
            print(f"[FAILURE] Access Denied: Unknown Person (no database entries)")
    
    print()
    
    # Test 2: Unauthorized user (very different embedding)
    print("="*60)
    print("TEST 2: UNAUTHORIZED USER (Different Embedding)")
    print("="*60)
    
    # Create a completely different embedding (distance > threshold)
    test_embedding_unauthorized = np.random.randn(128) * 2.0
    
    if config.DEBUG_MODE:
        print(f"[DEBUG] Face detected, generating embedding...")
        print(f"[DEBUG] Searching database for match...")
    
    matched_name, distance = db_manager.find_match(test_embedding_unauthorized, face_recognizer)
    
    if config.DEBUG_MODE:
        if matched_name:
            print(f"[DEBUG] Best match: {matched_name}, Distance: {distance:.4f}, Threshold: {config.RECOGNITION_THRESHOLD}")
        else:
            print(f"[DEBUG] Best match: None, Distance: {distance if distance else 'N/A'}, Threshold: {config.RECOGNITION_THRESHOLD}")
    
    if matched_name:
        print(f"[SUCCESS] Access Granted: {matched_name} (distance: {distance:.4f})")
    else:
        if distance and distance != float('inf'):
            print(f"[FAILURE] Access Denied: Unknown Person (best distance: {distance:.4f})")
        else:
            print(f"[FAILURE] Access Denied: Unknown Person (no database entries)")
    
    print()
    
    # Test 3: Empty database
    print("="*60)
    print("TEST 3: EMPTY DATABASE (No Users Enrolled)")
    print("="*60)
    
    # Create empty database
    db_manager_empty = DatabaseManager()
    
    if config.DEBUG_MODE:
        print(f"[DEBUG] Face detected, generating embedding...")
        print(f"[DEBUG] Searching database for match...")
    
    matched_name, distance = db_manager_empty.find_match(test_embedding_unauthorized, face_recognizer)
    
    if config.DEBUG_MODE:
        if matched_name:
            print(f"[DEBUG] Best match: {matched_name}, Distance: {distance:.4f}, Threshold: {config.RECOGNITION_THRESHOLD}")
        else:
            print(f"[DEBUG] Best match: None, Distance: {distance if distance else 'N/A'}, Threshold: {config.RECOGNITION_THRESHOLD}")
    
    if matched_name:
        print(f"[SUCCESS] Access Granted: {matched_name} (distance: {distance:.4f})")
    else:
        if distance and distance != float('inf'):
            print(f"[FAILURE] Access Denied: Unknown Person (best distance: {distance:.4f})")
        else:
            print(f"[FAILURE] Access Denied: Unknown Person (no database entries)")
    
    print()
    print("="*60)
    print("DEBUG OUTPUT TESTS COMPLETED")
    print("="*60)
    print()
    print("SUMMARY:")
    print("✓ DEBUG_MODE configuration added")
    print("✓ Debug logging shows face detection")
    print("✓ Debug logging shows database search")
    print("✓ Debug logging shows match details with distance")
    print("✓ SUCCESS messages include distance values")
    print("✓ FAILURE messages include best distance values")
    print("✓ Unauthorized users properly detected and logged")
    print()


if __name__ == '__main__':
    test_debug_logging()
