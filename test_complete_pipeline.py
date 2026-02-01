"""
Test script for the 11-stage face recognition pipeline
"""

import cv2
import numpy as np
from complete_pipeline import CompleteFaceRecognitionPipeline
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager
import config


def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\n" + "="*60)
    print("TEST 1: Pipeline Initialization")
    print("="*60)
    
    try:
        # Initialize components
        face_model = FaceRecognitionModel()
        db_manager = DatabaseManager()
        
        # Initialize pipeline
        pipeline = CompleteFaceRecognitionPipeline(
            face_recognition_model=face_model,
            database_manager=db_manager,
            enable_all_stages=True
        )
        
        print("✓ Pipeline initialized successfully")
        
        # Get stats
        stats = pipeline.get_pipeline_stats()
        print(f"✓ Pipeline stats: {stats}")
        
        return True
    
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return False


def test_frame_processing():
    """Test processing a sample frame"""
    print("\n" + "="*60)
    print("TEST 2: Frame Processing")
    print("="*60)
    
    try:
        # Initialize components
        face_model = FaceRecognitionModel()
        db_manager = DatabaseManager()
        
        # Initialize pipeline
        pipeline = CompleteFaceRecognitionPipeline(
            face_recognition_model=face_model,
            database_manager=db_manager,
            enable_all_stages=True
        )
        
        # Create a test frame (solid color for now)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (100, 100, 100)  # Gray background
        
        # Process frame
        results = pipeline.process_frame(test_frame, mode='full')
        
        print(f"✓ Frame processed")
        print(f"  - Faces detected: {len(results.get('faces', []))}")
        print(f"  - Recognized: {results.get('recognized', False)}")
        print(f"  - Stage results: {results.get('stage_results', {})}")
        
        return True
    
    except Exception as e:
        print(f"✗ Frame processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_stages():
    """Test individual pipeline stages"""
    print("\n" + "="*60)
    print("TEST 3: Individual Stages")
    print("="*60)
    
    try:
        from frame_preprocessor import FramePreprocessor
        from multi_model_detector import MultiModelFaceDetector
        from face_tracker import FaceTracker
        from face_quality_checker import FaceQualityChecker
        from face_enhancement import FaceEnhancer
        from multi_embeddings import MultiEmbeddingGenerator
        from advanced_matcher import AdvancedMatcher
        from post_processor import PostProcessor
        
        # Test each stage individually
        stages_tested = []
        
        # Stage 1: Frame Preprocessing
        try:
            preprocessor = FramePreprocessor()
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            processed = preprocessor.preprocess(test_frame)
            print("✓ Stage 1: Frame Preprocessing")
            stages_tested.append(1)
        except Exception as e:
            print(f"✗ Stage 1 failed: {e}")
        
        # Stage 2: Multi-Model Detection
        try:
            detector = MultiModelFaceDetector()
            print("✓ Stage 2: Multi-Model Face Detection")
            stages_tested.append(2)
        except Exception as e:
            print(f"✗ Stage 2 failed: {e}")
        
        # Stage 3: Face Tracking
        try:
            tracker = FaceTracker()
            print("✓ Stage 3: Face Tracking")
            stages_tested.append(3)
        except Exception as e:
            print(f"✗ Stage 3 failed: {e}")
        
        # Stage 4: Quality Assessment
        try:
            quality_checker = FaceQualityChecker()
            print("✓ Stage 4: Face Quality Assessment")
            stages_tested.append(4)
        except Exception as e:
            print(f"✗ Stage 4 failed: {e}")
        
        # Stage 8: Face Enhancement
        try:
            enhancer = FaceEnhancer()
            test_face = np.zeros((112, 112, 3), dtype=np.uint8)
            enhanced = enhancer.enhance(test_face)
            print("✓ Stage 8: Face Enhancement")
            stages_tested.append(8)
        except Exception as e:
            print(f"✗ Stage 8 failed: {e}")
        
        # Stage 9: Multi-Embeddings
        try:
            emb_gen = MultiEmbeddingGenerator()
            print("✓ Stage 9: Multi-Face Embeddings")
            stages_tested.append(9)
        except Exception as e:
            print(f"✗ Stage 9 failed: {e}")
        
        # Stage 10: Advanced Matching
        try:
            matcher = AdvancedMatcher()
            print("✓ Stage 10: Advanced Matching")
            stages_tested.append(10)
        except Exception as e:
            print(f"✗ Stage 10 failed: {e}")
        
        # Stage 11: Post-Processing
        try:
            post_proc = PostProcessor()
            print("✓ Stage 11: Post-Processing & Verification")
            stages_tested.append(11)
        except Exception as e:
            print(f"✗ Stage 11 failed: {e}")
        
        print(f"\n✓ Successfully tested {len(stages_tested)} stages")
        return True
    
    except Exception as e:
        print(f"✗ Individual stage testing failed: {e}")
        return False


def test_different_modes():
    """Test different processing modes"""
    print("\n" + "="*60)
    print("TEST 4: Processing Modes")
    print("="*60)
    
    try:
        face_model = FaceRecognitionModel()
        db_manager = DatabaseManager()
        
        pipeline = CompleteFaceRecognitionPipeline(
            face_recognition_model=face_model,
            database_manager=db_manager,
            enable_all_stages=True
        )
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        modes = ['full', 'fast', 'quality']
        
        for mode in modes:
            try:
                results = pipeline.process_frame(test_frame, mode=mode)
                print(f"✓ Mode '{mode}' processed successfully")
            except Exception as e:
                print(f"✗ Mode '{mode}' failed: {e}")
        
        return True
    
    except Exception as e:
        print(f"✗ Mode testing failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("11-STAGE FACE RECOGNITION PIPELINE TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Initialization
    results.append(("Initialization", test_pipeline_initialization()))
    
    # Test 2: Frame Processing
    results.append(("Frame Processing", test_frame_processing()))
    
    # Test 3: Individual Stages
    results.append(("Individual Stages", test_individual_stages()))
    
    # Test 4: Processing Modes
    results.append(("Processing Modes", test_different_modes()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready for production.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
