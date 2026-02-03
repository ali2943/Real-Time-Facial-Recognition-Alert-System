"""
Test script to verify all modules work correctly without requiring a camera
"""

import sys
import numpy as np


def test_imports():
    """Test that all modules can be imported"""
    print("\n[TEST] Testing module imports...")
    
    try:
from config import config
        print("✓ config module imported")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
from src.core import database_manager
        print("✓ database_manager module imported")
    except Exception as e:
        print(f"✗ Failed to import database_manager: {e}")
        return False
    
    try:
from src.utils import utils
        print("✓ utils module imported")
    except Exception as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    print("\n[TEST] All basic modules imported successfully!")
    return True


def test_database():
    """Test database manager functionality"""
    print("\n[TEST] Testing database manager...")
    
    try:
from src.core.database_manager import DatabaseManager
        
        # Initialize database
        db = DatabaseManager()
        print("✓ DatabaseManager initialized")
        
        # Test adding a user
        dummy_embedding = np.random.rand(128)
        db.add_user("test_user", dummy_embedding)
        print("✓ Added test user")
        
        # Test getting users
        users = db.get_all_users()
        assert "test_user" in users, "Test user not found in database"
        print(f"✓ Retrieved users: {users}")
        
        # Test getting embeddings
        embeddings = db.get_user_embeddings("test_user")
        assert len(embeddings) > 0, "No embeddings found for test user"
        print(f"✓ Retrieved {len(embeddings)} embedding(s)")
        
        # Test removing user
        db.remove_user("test_user")
        users = db.get_all_users()
        assert "test_user" not in users, "Test user still in database after removal"
        print("✓ Removed test user")
        
        print("\n[TEST] Database manager tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Database manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration values"""
    print("\n[TEST] Testing configuration...")
    
    try:
from config import config
        
        # Check essential config values
        assert hasattr(config, 'FACE_DETECTION_CONFIDENCE'), "Missing FACE_DETECTION_CONFIDENCE"
        assert hasattr(config, 'RECOGNITION_THRESHOLD'), "Missing RECOGNITION_THRESHOLD"
        assert hasattr(config, 'CAMERA_INDEX'), "Missing CAMERA_INDEX"
        assert hasattr(config, 'BBOX_COLOR_LEGIT'), "Missing BBOX_COLOR_LEGIT"
        assert hasattr(config, 'BBOX_COLOR_UNKNOWN'), "Missing BBOX_COLOR_UNKNOWN"
        
        print(f"✓ FACE_DETECTION_CONFIDENCE: {config.FACE_DETECTION_CONFIDENCE}")
        print(f"✓ RECOGNITION_THRESHOLD: {config.RECOGNITION_THRESHOLD}")
        print(f"✓ CAMERA_INDEX: {config.CAMERA_INDEX}")
        print(f"✓ BBOX_COLOR_LEGIT: {config.BBOX_COLOR_LEGIT}")
        print(f"✓ BBOX_COLOR_UNKNOWN: {config.BBOX_COLOR_UNKNOWN}")
        
        print("\n[TEST] Configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        return False


def test_utils():
    """Test utility functions"""
    print("\n[TEST] Testing utility functions...")
    
    try:
        import cv2
        import numpy as np
from src.utils.utils import draw_face_box, display_stats
        
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test draw_face_box
        box = [100, 100, 200, 200]
        draw_face_box(frame, box, "Test User", is_authorized=True)
        print("✓ draw_face_box (authorized) works")
        
        draw_face_box(frame, box, "Unknown", is_authorized=False)
        print("✓ draw_face_box (unauthorized) works")
        
        # Test display_stats
        display_stats(frame, 30.0)
        print("✓ display_stats works")
        
        print("\n[TEST] Utility function tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Utility function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("FACIAL RECOGNITION SYSTEM - MODULE TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Utilities", test_utils()))
    results.append(("Database", test_database()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("="*60)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
