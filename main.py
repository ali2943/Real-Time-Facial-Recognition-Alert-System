"""
Real-Time Facial Recognition Alert System
Main application for live camera feed with face recognition
"""

import cv2
import time
import argparse
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager
from utils import draw_face_box, save_unknown_face, display_stats
import config


class FacialRecognitionSystem:
    """Main system for real-time facial recognition"""
    
    def __init__(self):
        """Initialize the facial recognition system"""
        print("[INFO] Initializing Real-Time Facial Recognition Alert System...")
        
        # Initialize components
        self.detector = FaceDetector()
        self.recognizer = FaceRecognitionModel()
        self.db_manager = DatabaseManager()
        
        # Check if database has users
        users = self.db_manager.get_all_users()
        if len(users) == 0:
            print("\n[WARNING] No authorized users in database!")
            print("[WARNING] Please enroll users using: python enroll_user.py --name <name>")
        else:
            print(f"\n[INFO] Authorized users: {', '.join(users)}")
        
        # Tracking for unknown faces
        self.unknown_face_counter = 0
        
        print("[INFO] System initialized successfully!\n")
    
    def process_frame(self, frame):
        """
        Process a single frame: detect faces, recognize, and draw alerts
        
        Args:
            frame: Video frame from camera
            
        Returns:
            Processed frame with annotations
        """
        # Detect faces in the frame
        detections = self.detector.detect_faces(frame)
        
        # Process each detected face
        for detection in detections:
            box = detection['box']
            confidence = detection['confidence']
            
            try:
                # Extract face
                face = self.detector.extract_face(frame, box)
                
                # Skip if face extraction failed
                if face.size == 0:
                    continue
                
                # Generate embedding
                embedding = self.recognizer.get_embedding(face)
                
                # Find match in database
                matched_name, distance = self.db_manager.find_match(embedding, self.recognizer)
                
                if matched_name:
                    # Authorized user
                    draw_face_box(frame, box, matched_name, is_authorized=True)
                    print(f"[RECOGNIZED] {matched_name} (distance: {distance:.4f})")
                else:
                    # Unauthorized user
                    draw_face_box(frame, box, "Unknown", is_authorized=False)
                    print(f"[ALERT] Unknown person detected!")
                    
                    # Save unknown face if enabled
                    save_unknown_face(face, self.unknown_face_counter)
                    self.unknown_face_counter += 1
            
            except Exception as e:
                print(f"[ERROR] Failed to process face: {e}")
                continue
        
        return frame
    
    def run(self, camera_index=None):
        """
        Run the real-time facial recognition system
        
        Args:
            camera_index: Camera device index (default from config)
        """
        if camera_index is None:
            camera_index = config.CAMERA_INDEX
        
        print(f"[INFO] Opening camera (index: {camera_index})...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FPS)
        
        print("[INFO] Camera opened successfully")
        print("[INFO] Press 'q' to quit\n")
        
        # FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Display stats
            display_stats(processed_frame, fps)
            
            # Show frame
            cv2.imshow('Real-Time Facial Recognition Alert System', processed_frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Shutting down system...")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] System shutdown complete")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Real-Time Facial Recognition Alert System'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=None,
        help='Camera device index (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Create and run system
    system = FacialRecognitionSystem()
    system.run(camera_index=args.camera)


if __name__ == '__main__':
    main()
