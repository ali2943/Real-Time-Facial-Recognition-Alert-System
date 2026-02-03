"""
Test Suite for Accuracy Enhancement Features
Tests quality checker, face alignment, liveness detection, and enhanced matching
"""

import sys
import numpy as np
import cv2


def test_face_quality_checker():
    """Test face quality checker functionality"""
    print("\n[TEST] Testing Face Quality Checker...")
    
    try:
from src.quality.face_quality_checker import FaceQualityChecker
from config import config
        
        # Initialize checker
        checker = FaceQualityChecker()
        print("✓ FaceQualityChecker initialized")
        
        # Create test images
        # 1. Good quality image (sharp, well-lit, good contrast)
        good_image = np.random.randint(50, 200, (160, 160, 3), dtype=np.uint8)
        good_image = cv2.GaussianBlur(good_image, (3, 3), 0)  # Slight blur for realism
        
        # 2. Blurry image
        blurry_image = cv2.GaussianBlur(good_image, (15, 15), 0)
        
        # 3. Dark image
        dark_image = (good_image * 0.3).astype(np.uint8)
        
        # 4. Low contrast image
        low_contrast = np.full((160, 160, 3), 128, dtype=np.uint8)
        
        # Test blur detection
        is_sharp, variance = checker.check_blur_laplacian(good_image)
        print(f"✓ Blur check (good): sharp={is_sharp}, variance={variance:.2f}")
        
        is_sharp_blur, variance_blur = checker.check_blur_laplacian(blurry_image)
        print(f"✓ Blur check (blurry): sharp={is_sharp_blur}, variance={variance_blur:.2f}")
        assert variance > variance_blur, "Good image should have higher variance than blurry"
        
        # Test brightness check
        is_bright, brightness = checker.check_brightness(good_image)
        print(f"✓ Brightness check: adequate={is_bright}, value={brightness:.2f}")
        
        is_bright_dark, brightness_dark = checker.check_brightness(dark_image)
        print(f"✓ Brightness check (dark): adequate={is_bright_dark}, value={brightness_dark:.2f}")
        assert brightness > brightness_dark, "Good image should be brighter than dark image"
        
        # Test contrast check
        is_contrast, contrast = checker.check_contrast(good_image)
        print(f"✓ Contrast check: adequate={is_contrast}, value={contrast:.2f}")
        
        is_contrast_low, contrast_low = checker.check_contrast(low_contrast)
        print(f"✓ Contrast check (low): adequate={is_contrast_low}, value={contrast_low:.2f}")
        assert contrast > contrast_low, "Good image should have higher contrast"
        
        # Test resolution check
        is_res_ok, min_dim = checker.check_resolution(good_image)
        print(f"✓ Resolution check: adequate={is_res_ok}, min_dimension={min_dim}")
        
        # Test overall quality score
        quality_score = checker.get_quality_score(good_image)
        print(f"✓ Overall quality score: {quality_score}/100")
        assert 0 <= quality_score <= 100, "Quality score should be 0-100"
        
        print("\n[TEST] Face Quality Checker tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Face Quality Checker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_face_aligner():
    """Test face alignment functionality"""
    print("\n[TEST] Testing Face Aligner...")
    
    try:
from src.core.face_aligner import FaceAligner
        
        # Initialize aligner
        aligner = FaceAligner()
        print("✓ FaceAligner initialized")
        
        # Create test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Create test landmarks
        landmarks = {
            'left_eye': (60, 80),
            'right_eye': (140, 80),
            'nose': (100, 120),
            'mouth_left': (70, 150),
            'mouth_right': (130, 150)
        }
        
        # Test alignment
        aligned = aligner.align_face(test_image, landmarks)
        print(f"✓ Face aligned: input shape {test_image.shape}, output shape {aligned.shape}")
        
        assert aligned.shape[:2] == (112, 112), "Aligned face should be 112x112"
        
        # Test with only eyes (minimal landmarks)
        minimal_landmarks = {
            'left_eye': (60, 80),
            'right_eye': (140, 80)
        }
        aligned_minimal = aligner.align_face(test_image, minimal_landmarks)
        print(f"✓ Face aligned with minimal landmarks: {aligned_minimal.shape}")
        
        # Test fallback (no landmarks)
        aligned_no_landmarks = aligner.align_face(test_image, None)
        print(f"✓ Face aligned without landmarks (fallback): {aligned_no_landmarks.shape}")
        
        print("\n[TEST] Face Aligner tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Face Aligner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_liveness_detector():
    """Test liveness detection functionality"""
    print("\n[TEST] Testing Liveness Detector...")
    
    try:
from src.security.liveness_detector import LivenessDetector
        
        # Initialize detector
        detector = LivenessDetector()
        print("✓ LivenessDetector initialized")
        
        # Create test frames
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test face boxes (simulating movement)
        box1 = [100, 100, 150, 150]
        box2 = [105, 102, 150, 150]  # Slight movement
        
        # Test landmarks
        landmarks = {
            'left_eye': (120, 130),
            'right_eye': (160, 130),
            'nose': (140, 160)
        }
        
        # First call
        is_live1, conf1, reason1 = detector.is_live(frame1, box1, landmarks)
        print(f"✓ First liveness check: live={is_live1}, confidence={conf1:.2f}, reason={reason1}")
        
        # Second call (with movement)
        is_live2, conf2, reason2 = detector.is_live(frame2, box2, landmarks)
        print(f"✓ Second liveness check: live={is_live2}, confidence={conf2:.2f}, reason={reason2}")
        
        # Test texture analysis
        is_real, texture_score = detector.analyze_texture(frame1[100:250, 100:250])
        print(f"✓ Texture analysis: is_real={is_real}, score={texture_score:.2f}")
        
        # Reset detector
        detector.reset()
        print("✓ Detector reset successful")
        
        print("\n[TEST] Liveness Detector tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Liveness Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_database_manager():
    """Test enhanced database manager functionality"""
    print("\n[TEST] Testing Enhanced Database Manager...")
    
    try:
from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.face_recognition_model import FaceRecognitionModel
        
        # Initialize
        db = EnhancedDatabaseManager()
        print("✓ EnhancedDatabaseManager initialized")
        
        recognizer = FaceRecognitionModel()
        print("✓ FaceRecognitionModel initialized")
        
        # Create test embeddings
        embedding1 = np.random.rand(128).astype(np.float32)
        embedding2 = np.random.rand(128).astype(np.float32)
        embedding3 = embedding1 + np.random.rand(128) * 0.1  # Similar to embedding1
        
        # Add test users
        db.add_user("test_user_1", embedding1)
        db.add_user("test_user_1", embedding3)  # Second sample
        db.add_user("test_user_2", embedding2)
        print("✓ Added test users")
        
        # Test find_match_advanced
        query_embedding = embedding1 + np.random.rand(128) * 0.05  # Very similar to user 1
        matched_name, distance, confidence = db.find_match_advanced(query_embedding, recognizer)
        print(f"✓ Advanced match: name={matched_name}, distance={distance:.4f}, confidence={confidence:.2%}")
        
        # Test KNN matching
        matched_name_knn, avg_distance, votes = db.find_knn_match(query_embedding, k=2)
        print(f"✓ KNN match: name={matched_name_knn}, distance={avg_distance:.4f}, votes={votes}")
        
        # Test adaptive threshold
        threshold = db.calculate_adaptive_threshold("test_user_1")
        print(f"✓ Adaptive threshold for test_user_1: {threshold:.4f}")
        
        # Test confidence calculation
        distances = [0.5, 0.8, 1.0]
        confidence = db.get_match_confidence(distances, "test_user_1")
        print(f"✓ Confidence score: {confidence:.2%}")
        
        # Cleanup
        db.remove_user("test_user_1")
        db.remove_user("test_user_2")
        print("✓ Removed test users")
        
        print("\n[TEST] Enhanced Database Manager tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Enhanced Database Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_insightface_recognizer():
    """Test InsightFace recognizer (if available)"""
    print("\n[TEST] Testing InsightFace Recognizer...")
    
    try:
from src.advanced.insightface_recognizer import InsightFaceRecognizer
        
        # Try to initialize (may fail if not installed)
        recognizer = InsightFaceRecognizer(gpu_enabled=False)
        print("✓ InsightFaceRecognizer initialized")
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        detections = recognizer.detect_faces(test_frame)
        print(f"✓ Face detection completed: {len(detections)} faces detected")
        
        # Test embedding (if face detected or with mock)
        test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        try:
            embedding = recognizer.get_embedding(test_face)
            print(f"✓ Embedding generated: shape={embedding.shape}, size={len(embedding)}")
            assert len(embedding) == 512, "ArcFace embedding should be 512-d"
        except ValueError:
            print("✓ Embedding test skipped (no face in test image - expected)")
        
        # Test comparison
        emb1 = np.random.rand(512).astype(np.float32)
        emb2 = np.random.rand(512).astype(np.float32)
        distance = recognizer.compare_embeddings(emb1, emb2)
        print(f"✓ Embedding comparison: distance={distance:.4f}")
        
        print("\n[TEST] InsightFace Recognizer tests passed!")
        return True
        
    except (ImportError, RuntimeError) as e:
        print(f"\n⚠ InsightFace Recognizer not available (expected): {e}")
        print("This is normal if insightface is not installed.")
        return True  # Not a failure - just unavailable
    except Exception as e:
        print(f"\n✗ InsightFace Recognizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_additions():
    """Test that all new configuration parameters exist"""
    print("\n[TEST] Testing Configuration Additions...")
    
    try:
from config import config
        
        # Check new parameters
        required_params = [
            'USE_INSIGHTFACE',
            'INSIGHTFACE_MODEL',
            'GPU_ENABLED',
            'ENABLE_QUALITY_CHECKS',
            'BLUR_THRESHOLD',
            'BRIGHTNESS_RANGE',
            'MIN_CONTRAST',
            'MAX_POSE_ANGLE',
            'MIN_FACE_RESOLUTION',
            'OVERALL_QUALITY_THRESHOLD',
            'ENABLE_FACE_ALIGNMENT',
            'ALIGNED_FACE_SIZE',
            'LIVENESS_ENABLED',
            'LIVENESS_METHOD',
            'USE_KNN_MATCHING',
            'KNN_K',
            'ADAPTIVE_THRESHOLD_PER_USER',
            'MIN_MATCH_CONFIDENCE',
            'ENROLLMENT_SAMPLES',
            'ENROLLMENT_QUALITY_THRESHOLD',
            'CAPTURE_POSE_VARIATIONS'
        ]
        
        for param in required_params:
            assert hasattr(config, param), f"Missing config parameter: {param}"
            value = getattr(config, param)
            print(f"✓ {param}: {value}")
        
        print("\n[TEST] Configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all accuracy enhancement tests"""
    print("="*70)
    print("FACIAL RECOGNITION ACCURACY ENHANCEMENT - TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Configuration", test_config_additions()))
    results.append(("Face Quality Checker", test_face_quality_checker()))
    results.append(("Face Aligner", test_face_aligner()))
    results.append(("Liveness Detector", test_liveness_detector()))
    results.append(("Enhanced Database Manager", test_enhanced_database_manager()))
    results.append(("InsightFace Recognizer", test_insightface_recognizer()))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("="*70)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
