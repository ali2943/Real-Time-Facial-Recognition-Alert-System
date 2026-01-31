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
import numpy as np


class FacialRecognitionSystem:
    """Main system for real-time facial recognition - Security Door Access Control"""
    
    def __init__(self):
        """Initialize the facial recognition system"""
        print("[INFO] Initializing Security Door Access Control System...")
        
        # Initialize components - Try InsightFace first, fallback to FaceNet
        if config.USE_INSIGHTFACE:
            try:
                from insightface_recognizer import InsightFaceRecognizer
                self.recognizer = InsightFaceRecognizer(
                    model_name=config.INSIGHTFACE_MODEL,
                    gpu_enabled=config.GPU_ENABLED
                )
                self.detector = self.recognizer  # InsightFace has built-in detector
                print("[INFO] Using InsightFace (ArcFace) for recognition")
                
                # Adjust threshold for InsightFace (uses cosine distance, different scale)
                if config.RECOGNITION_THRESHOLD > 0.8:
                    config.RECOGNITION_THRESHOLD = 0.6
                    print(f"[INFO] Adjusted threshold for InsightFace: {config.RECOGNITION_THRESHOLD}")
            except (ImportError, RuntimeError) as e:
                print(f"[WARNING] InsightFace not available ({e}), using FaceNet")
                self.detector = FaceDetector()
                self.recognizer = FaceRecognitionModel()
        else:
            self.detector = FaceDetector()
            self.recognizer = FaceRecognitionModel()
        
        # Add new components
        if config.ENABLE_QUALITY_CHECKS:
            try:
                from face_quality_checker import FaceQualityChecker
                self.quality_checker = FaceQualityChecker()
            except ImportError as e:
                print(f"[WARNING] Quality checker not available: {e}")
                self.quality_checker = None
        else:
            self.quality_checker = None
        
        if config.ENABLE_FACE_ALIGNMENT:
            try:
                from face_aligner import FaceAligner
                self.face_aligner = FaceAligner()
            except ImportError as e:
                print(f"[WARNING] Face aligner not available: {e}")
                self.face_aligner = None
        else:
            self.face_aligner = None
        
        if config.LIVENESS_ENABLED:
            try:
                from liveness_detector import LivenessDetector
                self.liveness_detector = LivenessDetector()
            except ImportError as e:
                print(f"[WARNING] Liveness detector not available: {e}")
                self.liveness_detector = None
        else:
            self.liveness_detector = None
        
        # Enhanced database manager
        try:
            from enhanced_database_manager import EnhancedDatabaseManager
            self.db_manager = EnhancedDatabaseManager()
        except ImportError:
            print("[WARNING] Enhanced database manager not available, using standard")
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
                    
                    # Get landmarks if available
                    landmarks = detection.get('keypoints', None)
                    
                    # Step 1: Quality Check
                    if self.quality_checker is not None:
                        quality_result = self.quality_checker.check_all(face, landmarks)
                        quality_score = self.quality_checker.get_quality_score(face, landmarks)
                        
                        if config.DEBUG_MODE:
                            print(f"[DEBUG] Face quality score: {quality_score:.1f}/100")
                        
                        if quality_score < config.OVERALL_QUALITY_THRESHOLD:
                            if config.DEBUG_MODE:
                                print(f"[DEBUG] Quality too low ({quality_score:.1f}), skipping face")
                            # Show quality feedback to user
                            self._display_quality_feedback(frame, quality_result, quality_score)
                            continue
                    
                    # Step 2: Liveness Detection
                    if self.liveness_detector is not None:
                        is_live, liveness_conf, reason = self.liveness_detector.is_live(
                            frame, box, landmarks
                        )
                        
                        if config.DEBUG_MODE:
                            print(f"[DEBUG] Liveness check: {is_live}, confidence: {liveness_conf:.2f}, reason: {reason}")
                        
                        if not is_live:
                            print(f"[SECURITY] Liveness check failed: {reason}")
                            self._display_spoof_warning(frame)
                            log_access_event("SPOOF ATTEMPT", reason=reason)
                            continue
                    
                    # Step 3: Face Alignment
                    if self.face_aligner is not None and landmarks:
                        face = self.face_aligner.align_face(face, landmarks)
                    
                    # Step 4: Generate Embedding
                    if config.DEBUG_MODE:
                        print(f"[DEBUG] Face detected, generating embedding...")
                    
                    # For InsightFace, try to use face_object for better performance
                    if hasattr(self.recognizer, '__class__') and 'InsightFace' in self.recognizer.__class__.__name__:
                        face_object = detection.get('face_object', None)
                        if face_object is not None:
                            embedding = self.recognizer.get_embedding(face_object=face_object)
                        else:
                            embedding = self.recognizer.get_embedding(face)
                    else:
                        embedding = self.recognizer.get_embedding(face)
                    
                    # Step 5: Enhanced Matching
                    if config.DEBUG_MODE:
                        print(f"[DEBUG] Searching database for match...")
                    
                    if config.USE_KNN_MATCHING and hasattr(self.db_manager, 'find_match_advanced'):
                        matched_name, distance, confidence = self.db_manager.find_match_advanced(
                            embedding, self.recognizer
                        )
                    else:
                        matched_name, distance = self.db_manager.find_match(
                            embedding, self.recognizer
                        )
                        confidence = 1.0 - (distance / config.RECOGNITION_THRESHOLD) if matched_name else 0.0
                    
                    if config.DEBUG_MODE:
                        if matched_name:
                            print(f"[DEBUG] Best match: {matched_name}, Distance: {distance:.4f}, Confidence: {confidence:.2%}, Threshold: {config.RECOGNITION_THRESHOLD}")
                        else:
                            dist_str = f"{distance:.4f}" if distance is not None else "N/A"
                            print(f"[DEBUG] Best match: None, Distance: {dist_str}, Threshold: {config.RECOGNITION_THRESHOLD}")
                    
                    # Step 6: Confidence Check
                    if matched_name and confidence >= config.MIN_MATCH_CONFIDENCE:
                        # ACCESS GRANTED - Authorized user
                        print(f"[SUCCESS] Access Granted: {matched_name} (distance: {distance:.4f}, confidence: {confidence:.2%})")
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
                        if distance is not None and distance != float('inf'):
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
    
    def _display_quality_feedback(self, frame, quality_result, quality_score):
        """Display quality feedback on frame"""
        y_offset = 100
        for check_name, result in quality_result.items():
            status = "✓" if result['passed'] else "✗"
            color = (0, 255, 0) if result['passed'] else (0, 0, 255)
            text = f"{status} {check_name}: {result.get('value', 'N/A')}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
            y_offset += 25
        
        # Overall score
        score_text = f"Quality Score: {quality_score:.1f}/100"
        score_color = (0, 255, 0) if quality_score >= config.OVERALL_QUALITY_THRESHOLD else (0, 165, 255)
        cv2.putText(frame, score_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, score_color, 2)
    
    def _display_spoof_warning(self, frame):
        """Display spoofing warning on frame"""
        text = "SPOOFING ATTEMPT DETECTED"
        cv2.putText(frame, text, (50, frame.shape[0] // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
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
