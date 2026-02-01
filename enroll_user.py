"""
Script to enroll authorized users into the system
Captures face images and stores embeddings in the database
"""

import os
import cv2
import argparse
import numpy as np
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager
import config


def enroll_user(name, num_samples=5):
    """
    Enroll a new authorized user with enhanced quality checks
    
    Args:
        name: User name
        num_samples: Number of face samples to capture
    """
    print(f"\n[INFO] Enrolling user: {name}")
    print(f"[INFO] Will capture {num_samples} face samples")
    
    # Initialize components - Try InsightFace first, fallback to FaceNet
    if config.USE_INSIGHTFACE:
        try:
            from insightface_recognizer import InsightFaceRecognizer
            recognizer = InsightFaceRecognizer(
                model_name=config.INSIGHTFACE_MODEL,
                gpu_enabled=config.GPU_ENABLED
            )
            detector = recognizer  # InsightFace has built-in detector
            print("[INFO] Using InsightFace for enrollment")
        except (ImportError, RuntimeError) as e:
            print(f"[WARNING] InsightFace not available ({e}), using FaceNet")
            detector = FaceDetector()
            recognizer = FaceRecognitionModel()
    else:
        detector = FaceDetector()
        recognizer = FaceRecognitionModel()
    
    # Initialize quality checker if enabled
    quality_checker = None
    if config.ENABLE_QUALITY_CHECKS:
        try:
            from face_quality_checker import FaceQualityChecker
            quality_checker = FaceQualityChecker()
        except ImportError:
            print("[WARNING] Quality checker not available")
    
    # Initialize face aligner if enabled
    face_aligner = None
    if config.ENABLE_FACE_ALIGNMENT:
        try:
            from face_aligner import FaceAligner
            face_aligner = FaceAligner()
        except ImportError:
            print("[WARNING] Face aligner not available")
    
    # Enhanced database manager
    try:
        from enhanced_database_manager import EnhancedDatabaseManager
        db_manager = EnhancedDatabaseManager()
    except ImportError:
        db_manager = DatabaseManager()
    
    # Multi-angle pose instructions
    if config.CAPTURE_POSE_VARIATIONS and num_samples >= len(config.ENROLLMENT_ANGLES):
        # Use configured angles
        poses = []
        for angle in config.ENROLLMENT_ANGLES:
            if angle == 0:
                poses.append("Look straight at camera (front)")
            elif angle < 0:
                poses.append(f"Turn head slightly left (about {abs(angle)}°)")
            else:
                poses.append(f"Turn head slightly right (about {angle}°)")
        
        # Fill remaining with front-facing
        while len(poses) < num_samples:
            poses.append(f"Look straight at camera ({len(poses) - len(config.ENROLLMENT_ANGLES) + 1})")
    else:
        poses = ["Look at camera"] * num_samples
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return
    
    print("\n[INFO] Camera opened. Position your face in the frame.")
    print("[INFO] Press SPACE to capture a sample")
    print("[INFO] Press 'q' to quit enrollment")
    
    samples_captured = 0
    current_pose_index = 0
    captured_embeddings = []  # Store for variance checking
    
    while samples_captured < num_samples:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        # Detect faces
        detections = detector.detect_faces(frame)
        
        # Draw bounding boxes for detected faces
        for detection in detections:
            box = detection['box']
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display current pose instruction
        if config.CAPTURE_POSE_VARIATIONS:
            pose_text = f"Pose {samples_captured + 1}/{num_samples}: {poses[current_pose_index]}"
            cv2.putText(frame, pose_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 2)
        
        # Display quality feedback if available
        if quality_checker and len(detections) == 1:
            face = detector.extract_face(frame, detections[0]['box'])
            landmarks = detections[0].get('keypoints', None)
            
            quality_score = quality_checker.get_quality_score(face, landmarks)
            quality_checks = quality_checker.check_all(face, landmarks)
            
            # Display quality score
            score_color = (0, 255, 0) if quality_score >= config.ENROLLMENT_QUALITY_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, f"Quality: {quality_score:.1f}/100", 
                       (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
            
            # Display individual quality checks
            y_offset = 170
            for check_name, result in quality_checks.items():
                if not result['passed']:
                    text = f"✗ {check_name}"
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += 20
        
        # Display instructions
        cv2.putText(
            frame,
            f"Samples: {samples_captured}/{num_samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            "Press SPACE to capture",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow('Enrollment - ' + name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Capture sample on SPACE press
        if key == ord(' '):
            if len(detections) == 1:
                # Extract face
                face = detector.extract_face(frame, detections[0]['box'])
                landmarks = detections[0].get('keypoints', None)
                
                # Check quality if enabled
                quality_ok = True
                if quality_checker:
                    quality_score = quality_checker.get_quality_score(face, landmarks)
                    quality_ok = quality_score >= config.ENROLLMENT_QUALITY_THRESHOLD
                    
                    if not quality_ok:
                        print(f"[WARNING] Quality too low ({quality_score:.1f}/100). Please improve lighting/focus.")
                        continue
                
                # Align face if enabled
                if face_aligner and landmarks:
                    face = face_aligner.align_face(face, landmarks)
                
                # Generate embedding
                try:
                    # For InsightFace, try to use face_object for better performance
                    if hasattr(recognizer, '__class__') and 'InsightFace' in recognizer.__class__.__name__:
                        face_object = detections[0].get('face_object', None)
                        if face_object is not None:
                            embedding = recognizer.get_embedding(face_object=face_object)
                        else:
                            embedding = recognizer.get_embedding(face)
                    else:
                        embedding = recognizer.get_embedding(face)
                    
                    # Store embedding for variance checking
                    captured_embeddings.append(embedding)
                    
                    # Check intra-user variance if we have multiple samples
                    if len(captured_embeddings) >= 2:
                        variance = calculate_embedding_variance(captured_embeddings)
                        print(f"[INFO] Current embedding variance: {variance:.4f}")
                        
                        if len(captured_embeddings) >= config.ENROLLMENT_MIN_SAMPLES:
                            if variance > config.ENROLLMENT_MAX_VARIANCE:
                                print(f"[WARNING] High variance detected ({variance:.4f} > {config.ENROLLMENT_MAX_VARIANCE})")
                                print("[WARNING] Embeddings are inconsistent. Consider re-enrollment.")
                    
                    # Add to database
                    db_manager.add_user(name, embedding)
                    
                    samples_captured += 1
                    current_pose_index = min(current_pose_index + 1, len(poses) - 1)
                    print(f"[INFO] Captured sample {samples_captured}/{num_samples}")
                except Exception as e:
                    print(f"[ERROR] Failed to generate embedding: {e}")
            elif len(detections) == 0:
                print("[WARNING] No face detected. Please position your face in the frame.")
            else:
                print("[WARNING] Multiple faces detected. Please ensure only one face is in the frame.")
        
        # Quit on 'q' press
        elif key == ord('q'):
            print("[INFO] Enrollment cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if samples_captured == num_samples:
        # Final variance check
        if len(captured_embeddings) >= 2:
            final_variance = calculate_embedding_variance(captured_embeddings)
            print(f"\n[INFO] Final embedding variance: {final_variance:.4f}")
            
            if final_variance <= config.ENROLLMENT_MAX_VARIANCE:
                print(f"[SUCCESS] User '{name}' enrolled successfully with {samples_captured} high-quality samples!")
                print(f"[SUCCESS] Embedding consistency: EXCELLENT (variance: {final_variance:.4f})")
            else:
                print(f"[SUCCESS] User '{name}' enrolled with {samples_captured} samples")
                print(f"[WARNING] Embedding variance is high ({final_variance:.4f} > {config.ENROLLMENT_MAX_VARIANCE})")
                print("[WARNING] You may experience recognition issues. Consider re-enrolling.")
        else:
            print(f"\n[SUCCESS] User '{name}' enrolled successfully with {samples_captured} samples!")
    else:
        print(f"\n[INFO] Enrolled user '{name}' with {samples_captured} samples")


def calculate_embedding_variance(embeddings):
    """
    Calculate variance among embeddings
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Average pairwise distance (variance metric)
    """
    if len(embeddings) < 2:
        return 0.0
    
    embeddings_array = np.array(embeddings)
    distances = []
    
    for i in range(len(embeddings_array)):
        for j in range(i + 1, len(embeddings_array)):
            dist = np.linalg.norm(embeddings_array[i] - embeddings_array[j])
            distances.append(dist)
    
    return np.mean(distances)


def main():
    """Main function for enrollment script"""
    parser = argparse.ArgumentParser(description='Enroll authorized users')
    parser.add_argument('--name', type=str, required=True, help='Name of the user to enroll')
    parser.add_argument('--samples', type=int, default=config.ENROLLMENT_SAMPLES, 
                       help=f'Number of face samples to capture (default: {config.ENROLLMENT_SAMPLES})')
    
    args = parser.parse_args()
    
    enroll_user(args.name, args.samples)


if __name__ == '__main__':
    main()
