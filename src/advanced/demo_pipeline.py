"""
Demo script showing how to use the 11-stage face recognition pipeline
"""

import cv2
import numpy as np
from .complete_pipeline import CompleteFaceRecognitionPipeline
from src.core.face_recognition_model import FaceRecognitionModel
from src.core.database_manager import DatabaseManager


def demo_pipeline_basic():
    """
    Basic pipeline demonstration
    Shows minimal setup and usage
    """
    print("\n" + "="*70)
    print("DEMO 1: Basic Pipeline Usage")
    print("="*70)
    
    # Initialize components
    print("\n1. Initializing components...")
    face_model = FaceRecognitionModel()
    db_manager = DatabaseManager()
    
    # Create pipeline with all stages enabled
    print("\n2. Creating 11-stage pipeline...")
    pipeline = CompleteFaceRecognitionPipeline(
        face_recognition_model=face_model,
        database_manager=db_manager,
        enable_all_stages=True
    )
    
    # Create a test frame
    print("\n3. Processing test frame...")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (100, 120, 130)  # Gray-blue background
    
    # Process frame in 'full' mode (all stages)
    results = pipeline.process_frame(test_frame, mode='full')
    
    # Display results
    print("\n4. Results:")
    print(f"   - Recognized: {results['recognized']}")
    print(f"   - Name: {results.get('name', 'N/A')}")
    print(f"   - Confidence: {results.get('confidence', 0):.2%}")
    print(f"   - Faces detected: {len(results.get('faces', []))}")
    
    # Show pipeline stats
    stats = pipeline.get_pipeline_stats()
    print(f"\n5. Pipeline Statistics:")
    print(f"   - Stages enabled: {stats['stages_enabled']}")
    print(f"   - Active tracked faces: {stats['tracking_active_faces']}")


def demo_processing_modes():
    """
    Demonstrate different processing modes
    """
    print("\n" + "="*70)
    print("DEMO 2: Processing Modes Comparison")
    print("="*70)
    
    # Initialize
    face_model = FaceRecognitionModel()
    db_manager = DatabaseManager()
    
    pipeline = CompleteFaceRecognitionPipeline(
        face_recognition_model=face_model,
        database_manager=db_manager,
        enable_all_stages=True
    )
    
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    modes = {
        'fast': 'Optimized for speed, skips some stages',
        'full': 'Balanced accuracy and speed, all 11 stages',
        'quality': 'Maximum accuracy, ensemble detection'
    }
    
    print("\nProcessing same frame in different modes:\n")
    
    for mode, description in modes.items():
        print(f"Mode: {mode.upper()}")
        print(f"  Description: {description}")
        
        # Process
        results = pipeline.process_frame(test_frame, mode=mode)
        
        # Show stage results
        stage_results = results.get('stage_results', {})
        print(f"  Stage results:")
        print(f"    - Preprocessing: {stage_results.get('preprocessing', 'N/A')}")
        print(f"    - Detection mode: {stage_results.get('detection', {}).get('mode', 'N/A')}")
        print()


def demo_pipeline_with_webcam():
    """
    Demonstrate pipeline with webcam (simulation)
    This shows how to integrate with real camera
    """
    print("\n" + "="*70)
    print("DEMO 3: Webcam Integration (Simulated)")
    print("="*70)
    
    print("\nThis demo shows how to integrate the pipeline with a webcam.")
    print("In production, you would:")
    print()
    print("```python")
    print("# Initialize pipeline")
    print("pipeline = CompleteFaceRecognitionPipeline(...)")
    print()
    print("# Open webcam")
    print("cap = cv2.VideoCapture(0)")
    print()
    print("while True:")
    print("    ret, frame = cap.read()")
    print("    if not ret:")
    print("        break")
    print()
    print("    # Process frame")
    print("    results = pipeline.process_frame(frame, mode='full')")
    print()
    print("    # Check recognition")
    print("    if results['recognized']:")
    print("        name = results['name']")
    print("        confidence = results['confidence']")
    print("        print(f'Recognized: {name} ({confidence:.2%})')")
    print()
    print("    # Display (with bounding boxes, etc.)")
    print("    cv2.imshow('Recognition', frame)")
    print()
    print("    if cv2.waitKey(1) & 0xFF == ord('q'):")
    print("        break")
    print()
    print("cap.release()")
    print("cv2.destroyAllWindows()")
    print("```")


def demo_stage_details():
    """
    Show details about each pipeline stage
    """
    print("\n" + "="*70)
    print("DEMO 4: Pipeline Stage Details")
    print("="*70)
    
    stages = [
        ("Stage 1", "Frame Preprocessing", "frame_preprocessor.py",
         "Auto white balance, noise reduction, contrast enhancement"),
        
        ("Stage 2", "Multi-Model Face Detection", "multi_model_detector.py",
         "MTCNN + YuNet + Haar Cascade ensemble detection"),
        
        ("Stage 3", "Face Tracking", "face_tracker.py",
         "Temporal consistency, IoU tracking, embedding smoothing"),
        
        ("Stage 4", "Quality Assessment", "face_quality_checker.py",
         "Blur, brightness, contrast, resolution, pose, symmetry checks"),
        
        ("Stage 5", "Anti-Spoofing", "liveness_detector.py",
         "Blink detection, motion analysis, texture analysis"),
        
        ("Stage 6", "Face Alignment", "face_aligner.py",
         "Landmark-based alignment, normalization to template"),
        
        ("Stage 7", "Occlusion Detection", "face_occlusion_detector.py",
         "Mask detection, mouth/nose visibility, edge analysis"),
        
        ("Stage 8", "Face Enhancement", "face_enhancement.py",
         "Illumination normalization, detail enhancement, color correction"),
        
        ("Stage 9", "Multi-Embeddings", "multi_embeddings.py",
         "FaceNet + InsightFace, ensemble embeddings"),
        
        ("Stage 10", "Advanced Matching", "advanced_matcher.py",
         "Multi-metric matching, adaptive thresholds, confidence calibration"),
        
        ("Stage 11", "Post-Processing", "post_processor.py",
         "Multi-factor verification, temporal consistency, final decision"),
    ]
    
    print("\nPipeline Stage Breakdown:\n")
    
    for stage_num, stage_name, module, description in stages:
        print(f"{stage_num}: {stage_name}")
        print(f"  Module: {module}")
        print(f"  Function: {description}")
        print()


def demo_quality_filtering():
    """
    Demonstrate quality filtering
    """
    print("\n" + "="*70)
    print("DEMO 5: Quality Filtering Example")
    print("="*70)
    
    print("\nThe pipeline automatically filters out low-quality faces:")
    print()
    print("Quality checks performed:")
    print("  ✓ Blur detection (Laplacian variance > 100)")
    print("  ✓ Brightness (60-200 range)")
    print("  ✓ Contrast (std dev > 25)")
    print("  ✓ Resolution (minimum 80px)")
    print("  ✓ Pose angle (< 30 degrees)")
    print("  ✓ Eye visibility (both eyes present)")
    print("  ✓ Symmetry (natural asymmetry)")
    print("  ✓ Noise level (variance < 1000)")
    print()
    print("Overall quality score: 0-100")
    print("  - Score < 50: Rejected")
    print("  - Score 50-70: Borderline")
    print("  - Score > 70: Good quality")
    print()
    print("Example rejection reasons:")
    print("  • 'Low quality' - Overall score too low")
    print("  • 'Liveness check failed' - Possible spoofing")
    print("  • 'Occlusion detected' - Mask or obstruction")


def run_all_demos():
    """Run all demonstration scripts"""
    print("\n" + "="*70)
    print("11-STAGE FACE RECOGNITION PIPELINE - DEMONSTRATIONS")
    print("="*70)
    
    demos = [
        ("Basic Usage", demo_pipeline_basic),
        ("Processing Modes", demo_processing_modes),
        ("Webcam Integration", demo_pipeline_with_webcam),
        ("Stage Details", demo_stage_details),
        ("Quality Filtering", demo_quality_filtering),
    ]
    
    for demo_name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Demo '{demo_name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nFor more information, see:")
    print("  - PIPELINE_DOCUMENTATION.md - Complete pipeline documentation")
    print("  - test_complete_pipeline.py - Test suite")
    print("  - complete_pipeline.py - Full implementation")


if __name__ == "__main__":
    run_all_demos()
