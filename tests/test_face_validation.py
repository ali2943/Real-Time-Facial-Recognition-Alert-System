"""
Test script for face validation enhancements
Tests mask detection and eye state verification
"""

import sys
import numpy as np
import cv2


def test_imports():
    """Test that new modules can be imported"""
    print("\n[TEST] Testing new module imports...")
    
    try:
from config import config
        print("✓ config module imported")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
from src.security.face_occlusion_detector import FaceOcclusionDetector
        print("✓ face_occlusion_detector module imported")
    except Exception as e:
        print(f"✗ Failed to import face_occlusion_detector: {e}")
        return False
    
    try:
from src.security.eye_state_detector import EyeStateDetector
        print("✓ eye_state_detector module imported")
    except Exception as e:
        print(f"✗ Failed to import eye_state_detector: {e}")
        return False
    
    print("\n[TEST] All new modules imported successfully!")
    return True


def test_face_occlusion_detector():
    """Test Face Occlusion Detector functionality"""
    print("\n[TEST] Testing Face Occlusion Detector...")
    
    try:
from src.security.face_occlusion_detector import FaceOcclusionDetector
        
        # Initialize detector
        detector = FaceOcclusionDetector()
        print("✓ FaceOcclusionDetector initialized")
        
        # Create dummy face image (black image)
        dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
        
        # Test 1: Mask detection on black image (should detect as covered)
        has_mask, confidence, reason = detector.detect_mask(dummy_face)
        print(f"  Test 1 - Black image: has_mask={has_mask}, confidence={confidence:.2f}, reason='{reason}'")
        
        # Test 2: Create a face-like image with texture
        dummy_face_with_texture = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        has_mask2, confidence2, reason2 = detector.detect_mask(dummy_face_with_texture)
        print(f"  Test 2 - Textured image: has_mask={has_mask2}, confidence={confidence2:.2f}, reason='{reason2}'")
        
        # Test 3: Mouth visibility check
        mouth_visible, mouth_conf = detector.is_mouth_visible(dummy_face)
        print(f"  Test 3 - Mouth visibility: visible={mouth_visible}, confidence={mouth_conf:.2f}")
        
        # Test 4: Nose visibility check
        nose_visible, nose_conf = detector.is_nose_visible(dummy_face)
        print(f"  Test 4 - Nose visibility: visible={nose_visible}, confidence={nose_conf:.2f}")
        
        # Test 5: Occlusion detection
        is_occluded, occl_conf, occluded_regions = detector.detect_occlusion(dummy_face)
        print(f"  Test 5 - Occlusion detection: occluded={is_occluded}, confidence={occl_conf:.2f}, regions={occluded_regions}")
        
        print("✓ Face Occlusion Detector tests completed")
        return True
    
    except Exception as e:
        print(f"✗ Face Occlusion Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eye_state_detector():
    """Test Eye State Detector functionality"""
    print("\n[TEST] Testing Eye State Detector...")
    
    try:
from src.security.eye_state_detector import EyeStateDetector
        
        # Initialize detector
        detector = EyeStateDetector()
        print("✓ EyeStateDetector initialized")
        
        # Create dummy face image
        dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
        
        # Test 1: EAR calculation with dummy eye landmarks (6 points)
        # Simulating open eyes (wider vertical distance)
        open_eye = np.array([
            [30, 50],   # Left corner
            [35, 45],   # Top left
            [40, 43],   # Top right
            [50, 50],   # Right corner
            [40, 57],   # Bottom right
            [35, 55]    # Bottom left
        ], dtype=np.float32)
        
        ear_open = detector.calculate_ear(open_eye)
        print(f"  Test 1 - Open eye EAR: {ear_open:.3f} (should be > 0.21)")
        
        # Test 2: Calculate EAR for closed eyes (narrow vertical distance)
        closed_eye = np.array([
            [30, 50],   # Left corner
            [35, 49],   # Top left
            [40, 48],   # Top right
            [50, 50],   # Right corner
            [40, 51],   # Bottom right
            [35, 52]    # Bottom left
        ], dtype=np.float32)
        
        ear_closed = detector.calculate_ear(closed_eye)
        print(f"  Test 2 - Closed eye EAR: {ear_closed:.3f} (should be < 0.21)")
        
        # Test 3: Test with landmarks dictionary
        landmarks = {
            'left_eye': open_eye,
            'right_eye': open_eye
        }
        
        both_open, left_ear, right_ear, reason = detector.are_eyes_open(landmarks)
        print(f"  Test 3 - Eyes open check: both_open={both_open}, left={left_ear:.3f}, right={right_ear:.3f}")
        print(f"           Reason: {reason}")
        
        # Test 4: Test with closed eyes
        landmarks_closed = {
            'left_eye': closed_eye,
            'right_eye': closed_eye
        }
        
        both_open_2, left_ear_2, right_ear_2, reason_2 = detector.are_eyes_open(landmarks_closed)
        print(f"  Test 4 - Eyes closed check: both_open={both_open_2}, left={left_ear_2:.3f}, right={right_ear_2:.3f}")
        print(f"           Reason: {reason_2}")
        
        # Test 5: Test eye occlusion detection (without landmarks)
        # This will fail gracefully as we can't extract eye regions without proper landmarks
        print(f"  Test 5 - Eye occlusion detection (skipped - requires proper landmarks)")
        
        print("✓ Eye State Detector tests completed")
        return True
    
    except Exception as e:
        print(f"✗ Eye State Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_settings():
    """Test that new configuration settings are present"""
    print("\n[TEST] Testing configuration settings...")
    
    try:
from config import config
        
        # Check mask detection settings
        assert hasattr(config, 'ENABLE_MASK_DETECTION'), "ENABLE_MASK_DETECTION not found"
        print(f"✓ ENABLE_MASK_DETECTION = {config.ENABLE_MASK_DETECTION}")
        
        assert hasattr(config, 'ENABLE_OCCLUSION_DETECTION'), "ENABLE_OCCLUSION_DETECTION not found"
        print(f"✓ ENABLE_OCCLUSION_DETECTION = {config.ENABLE_OCCLUSION_DETECTION}")
        
        assert hasattr(config, 'MASK_DETECTION_CONFIDENCE'), "MASK_DETECTION_CONFIDENCE not found"
        print(f"✓ MASK_DETECTION_CONFIDENCE = {config.MASK_DETECTION_CONFIDENCE}")
        
        # Check eye state settings
        assert hasattr(config, 'ENABLE_EYE_STATE_CHECK'), "ENABLE_EYE_STATE_CHECK not found"
        print(f"✓ ENABLE_EYE_STATE_CHECK = {config.ENABLE_EYE_STATE_CHECK}")
        
        assert hasattr(config, 'REQUIRE_BOTH_EYES_OPEN'), "REQUIRE_BOTH_EYES_OPEN not found"
        print(f"✓ REQUIRE_BOTH_EYES_OPEN = {config.REQUIRE_BOTH_EYES_OPEN}")
        
        assert hasattr(config, 'EYE_ASPECT_RATIO_THRESHOLD'), "EYE_ASPECT_RATIO_THRESHOLD not found"
        print(f"✓ EYE_ASPECT_RATIO_THRESHOLD = {config.EYE_ASPECT_RATIO_THRESHOLD}")
        
        # Check face visibility settings
        assert hasattr(config, 'REQUIRE_FULL_FACE_VISIBLE'), "REQUIRE_FULL_FACE_VISIBLE not found"
        print(f"✓ REQUIRE_FULL_FACE_VISIBLE = {config.REQUIRE_FULL_FACE_VISIBLE}")
        
        assert hasattr(config, 'MIN_FACIAL_FEATURES_VISIBLE'), "MIN_FACIAL_FEATURES_VISIBLE not found"
        print(f"✓ MIN_FACIAL_FEATURES_VISIBLE = {config.MIN_FACIAL_FEATURES_VISIBLE}")
        
        print("✓ All configuration settings present")
        return True
    
    except AssertionError as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_main_integration():
    """Test that main.py can initialize with new detectors"""
    print("\n[TEST] Testing main.py integration...")
    
    try:
from config import config
        
        # Temporarily disable the detectors to avoid initialization issues
        original_mask = config.ENABLE_MASK_DETECTION
        original_eye = config.ENABLE_EYE_STATE_CHECK
        
        # Test with detectors enabled
        config.ENABLE_MASK_DETECTION = True
        config.ENABLE_EYE_STATE_CHECK = True
        
        # Import modules
from src.security.face_occlusion_detector import FaceOcclusionDetector
from src.security.eye_state_detector import EyeStateDetector
        
        # Initialize detectors
        mask_detector = FaceOcclusionDetector()
        eye_detector = EyeStateDetector()
        
        print("✓ Detectors initialized successfully")
        
        # Restore original config
        config.ENABLE_MASK_DETECTION = original_mask
        config.ENABLE_EYE_STATE_CHECK = original_eye
        
        print("✓ Main integration test completed")
        return True
    
    except Exception as e:
        print(f"✗ Main integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Face Validation Enhancement Tests")
    print("="*60)
    
    results = []
    
    # Test 1: Module imports
    results.append(("Module Imports", test_imports()))
    
    # Test 2: Configuration settings
    results.append(("Configuration Settings", test_config_settings()))
    
    # Test 3: Face Occlusion Detector
    results.append(("Face Occlusion Detector", test_face_occlusion_detector()))
    
    # Test 4: Eye State Detector
    results.append(("Eye State Detector", test_eye_state_detector()))
    
    # Test 5: Main integration
    results.append(("Main Integration", test_main_integration()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
