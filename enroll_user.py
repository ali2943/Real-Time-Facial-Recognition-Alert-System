"""
Script to enroll authorized users into the system
Captures face images and stores embeddings in the database
"""

import os
import cv2
import argparse
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager


def enroll_user(name, num_samples=5):
    """
    Enroll a new authorized user
    
    Args:
        name: User name
        num_samples: Number of face samples to capture
    """
    print(f"\n[INFO] Enrolling user: {name}")
    print(f"[INFO] Will capture {num_samples} face samples")
    
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognitionModel()
    db_manager = DatabaseManager()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return
    
    print("\n[INFO] Camera opened. Position your face in the frame.")
    print("[INFO] Press SPACE to capture a sample")
    print("[INFO] Press 'q' to quit enrollment")
    
    samples_captured = 0
    
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
                
                # Generate embedding
                try:
                    embedding = recognizer.get_embedding(face)
                    
                    # Add to database
                    db_manager.add_user(name, embedding)
                    
                    samples_captured += 1
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
        print(f"\n[SUCCESS] User '{name}' enrolled successfully with {samples_captured} samples!")
    else:
        print(f"\n[INFO] Enrolled user '{name}' with {samples_captured} samples")


def main():
    """Main function for enrollment script"""
    parser = argparse.ArgumentParser(description='Enroll authorized users')
    parser.add_argument('--name', type=str, required=True, help='Name of the user to enroll')
    parser.add_argument('--samples', type=int, default=5, help='Number of face samples to capture (default: 5)')
    
    args = parser.parse_args()
    
    enroll_user(args.name, args.samples)


if __name__ == '__main__':
    main()
