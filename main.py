"""
Real-Time Facial Recognition Alert System
Main application for live camera feed with face recognition
Configured as a continuous security door access control system
"""

import cv2
import time
import argparse
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager
from utils import (
    draw_face_box, save_unknown_face, display_stats,
    display_access_granted, display_access_denied, display_system_ready,
    log_access_event, display_system_status
)
import config


class FacialRecognitionSystem:
    """Main system for real-time facial recognition - Security Door Access Control"""
    
    def __init__(self):
        """Initialize the facial recognition system"""
        print("[INFO] Initializing Security Door Access Control System...")
        
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
        
        # Access control state
        self.access_state = "ready"  # "ready", "granted", "denied"
        self.access_state_until = 0  # Timestamp when to return to ready
        self.last_access_person = None
        self.last_access_time = 0  # Cooldown tracking
        self.last_event_text = ""
        
        # System uptime tracking
        self.system_start_time = time.time()
        
        print("[INFO] System initialized successfully!")
        print("[INFO] Running in continuous 24/7 mode - errors will be logged and recovered\n")
    
    def process_frame(self, frame):
        """
        Process a single frame: detect faces, recognize, and draw alerts
        
        Args:
            frame: Video frame from camera
            
        Returns:
            Processed frame with annotations
        """
        current_time = time.time()
        
        # Check if we're in an access state (granted/denied)
        if self.access_state != "ready" and current_time < self.access_state_until:
            # Still showing access message
            if self.access_state == "granted":
                display_access_granted(frame, self.last_access_person)
            elif self.access_state == "denied":
                display_access_denied(frame)
            return frame
        
        # Return to ready state
        if self.access_state != "ready" and current_time >= self.access_state_until:
            self.access_state = "ready"
        
        # Check cooldown period
        if current_time - self.last_access_time < config.ACCESS_COOLDOWN:
            display_system_ready(frame)
            return frame
        
        try:
            # Detect faces in the frame
            detections = self.detector.detect_faces(frame)
            
            # If no faces detected, show ready state
            if len(detections) == 0:
                display_system_ready(frame)
                return frame
            
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
                    if config.DEBUG_MODE:
                        print(f"[DEBUG] Face detected, generating embedding...")
                    embedding = self.recognizer.get_embedding(face)
                    
                    # Find match in database
                    if config.DEBUG_MODE:
                        print(f"[DEBUG] Searching database for match...")
                    matched_name, distance = self.db_manager.find_match(embedding, self.recognizer)
                    
                    if config.DEBUG_MODE:
                        if matched_name:
                            print(f"[DEBUG] Best match: {matched_name}, Distance: {distance:.4f}, Threshold: {config.RECOGNITION_THRESHOLD}")
                        else:
                            print(f"[DEBUG] Best match: None, Distance: {distance if distance else 'N/A'}, Threshold: {config.RECOGNITION_THRESHOLD}")
                    
                    if matched_name:
                        # ACCESS GRANTED - Authorized user
                        print(f"[SUCCESS] Access Granted: {matched_name} (distance: {distance:.4f})")
                        self.access_state = "granted"
                        self.access_state_until = current_time + config.ACCESS_GRANTED_DISPLAY_TIME
                        self.last_access_person = matched_name
                        self.last_access_time = current_time
                        self.last_event_text = f"Last: GRANTED - {matched_name}"
                        
                        # Log access event
                        log_access_event("ACCESS GRANTED", person_name=matched_name)
                        
                        # Display granted message
                        display_access_granted(frame, matched_name)
                        
                        # Only process first detected face
                        break
                    else:
                        # ACCESS DENIED - Unauthorized user
                        if distance and distance != float('inf'):
                            print(f"[FAILURE] Access Denied: Unknown Person (best distance: {distance:.4f})")
                        else:
                            print(f"[FAILURE] Access Denied: Unknown Person (no database entries)")
                        self.access_state = "denied"
                        self.access_state_until = current_time + config.ACCESS_DENIED_DISPLAY_TIME
                        self.last_access_time = current_time
                        
                        # Save unknown face if enabled
                        photo_filename = None
                        if config.SAVE_UNKNOWN_FACES:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            photo_filename = f"unknown_{timestamp}_{self.unknown_face_counter}.jpg"
                            save_unknown_face(face, self.unknown_face_counter)
                            self.unknown_face_counter += 1
                        
                        self.last_event_text = f"Last: DENIED - Unknown"
                        
                        # Log access event
                        log_access_event("ACCESS DENIED", photo_filename=photo_filename)
                        
                        # Display denied message
                        display_access_denied(frame)
                        
                        # Only process first detected face
                        break
                
                except Exception as e:
                    print(f"[ERROR] Failed to process face: {e}")
                    # Continue to next face instead of crashing
                    continue
        
        except Exception as e:
            print(f"[ERROR] Frame processing error: {e}")
            # Don't crash - just show ready state
            display_system_ready(frame)
        
        return frame
    
    def run(self, camera_index=None):
        """
        Run the real-time facial recognition system with continuous operation
        
        Args:
            camera_index: Camera device index (default from config)
        """
        if camera_index is None:
            camera_index = config.CAMERA_INDEX
        
        reconnect_attempts = 0
        
        while True:  # Continuous operation
            try:
                print(f"[INFO] Opening camera (index: {camera_index})...")
                cap = cv2.VideoCapture(camera_index)
                
                if not cap.isOpened():
                    print("[ERROR] Could not open camera")
                    if config.AUTO_RECONNECT_CAMERA and reconnect_attempts < config.MAX_RECONNECT_ATTEMPTS:
                        reconnect_attempts += 1
                        print(f"[INFO] Attempting to reconnect ({reconnect_attempts}/{config.MAX_RECONNECT_ATTEMPTS})...")
                        time.sleep(2)
                        continue
                    else:
                        print("[ERROR] Maximum reconnection attempts reached. Exiting.")
                        return
                
                # Camera opened successfully
                reconnect_attempts = 0
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.FPS)
                
                print("[INFO] Camera opened successfully")
                print("[INFO] System running in continuous mode - Press 'q' to quit")
                print("[INFO] Access events will be logged to:", config.LOG_FILE_PATH)
                print()
                
                # FPS calculation
                fps = 0
                frame_count = 0
                start_time = time.time()
                frame_skip_counter = 0
                
                while True:
                    try:
                        ret, frame = cap.read()
                        
                        if not ret:
                            print("[ERROR] Failed to read frame from camera")
                            if config.AUTO_RECONNECT_CAMERA:
                                print("[INFO] Camera disconnected. Attempting to reconnect...")
                                break  # Break inner loop to reconnect
                            else:
                                return
                        
                        # Frame skipping for performance
                        frame_skip_counter += 1
                        if frame_skip_counter % config.FRAME_SKIP != 0:
                            continue
                        
                        # Process frame with error handling
                        try:
                            processed_frame = self.process_frame(frame)
                        except Exception as e:
                            print(f"[ERROR] Frame processing failed: {e}")
                            processed_frame = frame
                            display_system_ready(processed_frame)
                        
                        # Calculate FPS
                        frame_count += 1
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 1.0:
                            fps = frame_count / elapsed_time
                            frame_count = 0
                            start_time = time.time()
                        
                        # Display system status
                        uptime = time.time() - self.system_start_time
                        display_system_status(processed_frame, fps, uptime, self.last_event_text)
                        
                        # Show frame
                        cv2.imshow('Security Door Access Control System', processed_frame)
                        
                        # Check for quit command
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\n[INFO] Shutting down system...")
                            cap.release()
                            cv2.destroyAllWindows()
                            print("[INFO] System shutdown complete")
                            return
                    
                    except Exception as e:
                        print(f"[ERROR] Unexpected error in main loop: {e}")
                        # Don't crash - continue processing
                        time.sleep(0.1)
                        continue
                
                # If we get here, we need to reconnect
                cap.release()
                cv2.destroyAllWindows()
            
            except Exception as e:
                print(f"[ERROR] Critical error: {e}")
                if config.AUTO_RECONNECT_CAMERA and reconnect_attempts < config.MAX_RECONNECT_ATTEMPTS:
                    reconnect_attempts += 1
                    print(f"[INFO] Recovering... ({reconnect_attempts}/{config.MAX_RECONNECT_ATTEMPTS})")
                    time.sleep(2)
                    continue
                else:
                    print("[ERROR] System cannot recover. Exiting.")
                    return


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
