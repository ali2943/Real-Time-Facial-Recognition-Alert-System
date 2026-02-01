"""
Real-Time Facial Recognition Alert System
Main application for on-click face verification with 100% confidence requirement
Configured as an on-demand security door access control system
"""

import cv2
import time
import argparse
from collections import Counter
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


class FrameBuffer:
    """
    Store recent detections for temporal consistency checking
    
    Maintains a sliding window of recent face recognition results
    to ensure consistent identification across multiple frames.
    """
    
    def __init__(self, buffer_size=5):
        """
        Initialize frame buffer
        
        Args:
            buffer_size: Number of recent detections to store
        """
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add_detection(self, name, confidence):
        """
        Add a detection result to the buffer
        
        Args:
            name: Detected person's name (or None for unknown)
            confidence: Confidence score (0.0 to 1.0)
        """
        self.buffer.append((name, confidence))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def get_consensus(self):
        """
        Get consensus from recent frames using majority voting
        
        Returns:
            Tuple of (consensus_name, average_confidence)
            Returns (None, 0.0) if no consensus reached
        """
        if len(self.buffer) < 3:
            # Need at least 3 frames for consensus
            return None, 0.0
        
        # Count occurrences of each name
        names = [n for n, c in self.buffer]
        name_counts = Counter(names)
        
        # Get most common name
        most_common_name, count = name_counts.most_common(1)[0]
        
        # Calculate average confidence for that name
        confidences = [c for n, c in self.buffer if n == most_common_name]
        avg_confidence = np.mean(confidences)
        
        # Require at least MIN_CONSENSUS_RATIO consistency
        if count / len(self.buffer) >= config.MIN_CONSENSUS_RATIO:
            return most_common_name, avg_confidence
        
        return None, 0.0
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = []
    
    def is_empty(self):
        """Check if buffer is empty"""
        return len(self.buffer) == 0


class FacialRecognitionSystem:
    """Main system for on-click facial recognition - Security Door Access Control"""
    
    def __init__(self):
        """Initialize the facial recognition system"""
        print("[INFO] Initializing Security Door Access Control System...")
        
        # Store recognition threshold (may be adjusted for InsightFace)
        self.recognition_threshold = config.RECOGNITION_THRESHOLD
        
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
                # Store in instance variable instead of modifying global config
                if self.recognition_threshold > 0.7:
                    self.recognition_threshold = 0.5  # Stricter for InsightFace
                    print(f"[INFO] Adjusted threshold for InsightFace: {self.recognition_threshold}")
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
        
        # Mask/Occlusion Detection
        if config.ENABLE_MASK_DETECTION or config.ENABLE_OCCLUSION_DETECTION:
            try:
                from face_occlusion_detector import FaceOcclusionDetector
                self.mask_detector = FaceOcclusionDetector()
            except ImportError as e:
                print(f"[WARNING] Mask detector not available: {e}")
                self.mask_detector = None
        else:
            self.mask_detector = None
        
        # Eye State Detection
        if config.ENABLE_EYE_STATE_CHECK:
            try:
                from eye_state_detector import EyeStateDetector
                self.eye_detector = EyeStateDetector()
            except ImportError as e:
                print(f"[WARNING] Eye detector not available: {e}")
                self.eye_detector = None
        else:
            self.eye_detector = None
        
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
            print("\n[ERROR] No authorized users in database!")
            print("[ERROR] System cannot start without enrolled users for security reasons.")
            print("[ERROR] Please enroll at least one user using: python enroll_user.py --name <name>")
            raise RuntimeError("Cannot start system with empty database - security risk!")
        else:
            print(f"\n[INFO] Authorized users ({len(users)}): {', '.join(users)}")
        
        # Tracking for unknown faces
        self.unknown_face_counter = 0
        
        # Frame buffer for temporal consistency (if enabled)
        self.frame_buffer = None
        if config.USE_TEMPORAL_CONSISTENCY:
            self.frame_buffer = FrameBuffer(buffer_size=config.TEMPORAL_BUFFER_SIZE)
            print(f"[INFO] Temporal consistency enabled (buffer size: {config.TEMPORAL_BUFFER_SIZE})")
        
        # Last access tracking
        self.last_access_person = None
        self.last_access_time = 0
        self.last_event_text = ""
        
        # System uptime tracking
        self.system_start_time = time.time()
        
        print("[INFO] System initialized successfully!")
        print("[INFO] Running in ON-CLICK mode - Press SPACE to capture and verify\n")
    
    def process_frame(self, frame):
        """
        Process a single frame: detect faces, recognize, and draw alerts
        This is called once per button press for on-click verification
        
        IMPORTANT: This function may return early if:
        - No face is detected
        - Face extraction fails
        - Quality checks fail
        - Liveness check fails
        - Unknown person detected (for security)
        
        In all early return cases, frame state (last_access_time, last_event_text)
        is properly updated before returning.
        
        Args:
            frame: Video frame from camera
            
        Returns:
            Processed frame with annotations
        """
        current_time = time.time()
        
        try:
            # Detect faces in the frame
            detections = self.detector.detect_faces(frame)
            
            # If no faces detected
            if len(detections) == 0:
                cv2.putText(frame, "No face detected in image", 
                           (50, frame.shape[0] // 2 - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                print("[INFO] No face detected in captured image")
                return frame
            
            # Process the first detected face
            detection = detections[0]
            box = detection['box']
            confidence = detection['confidence']
            
            try:
                # Extract face
                face = self.detector.extract_face(frame, box)
                
                # Skip if face extraction failed
                if face.size == 0:
                    cv2.putText(frame, "Face extraction failed", 
                               (50, frame.shape[0] // 2 - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    return frame
                
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
                            print(f"[DEBUG] Quality too low ({quality_score:.1f}), rejecting")
                        # Show quality feedback to user
                        self._display_quality_feedback(frame, quality_result, quality_score)
                        return frame
                
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
                        return frame
                
                # NEW: Step 2a - Mask Detection
                if config.ENABLE_MASK_DETECTION and self.mask_detector is not None:
                    has_mask, mask_conf, mask_reason = self.mask_detector.detect_mask(face, landmarks)
                    
                    if has_mask:
                        if config.DEBUG_MODE:
                            print(f"[SECURITY] Mask detected: {mask_reason} (confidence: {mask_conf:.2%})")
                        
                        cv2.putText(frame, "ACCESS DENIED: Face Covered/Mask Detected", 
                                   (50, frame.shape[0] // 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(frame, f"Reason: {mask_reason}", 
                                   (50, frame.shape[0] // 2 + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        log_access_event("MASK_DETECTED", reason=mask_reason)
                        self.last_event_text = f"Last: DENIED - Mask detected"
                        self.last_access_time = current_time
                        return frame
                
                # NEW: Step 2b - Eye State Check
                if config.ENABLE_EYE_STATE_CHECK and self.eye_detector is not None:
                    eyes_open, left_ear, right_ear, eye_reason = self.eye_detector.are_eyes_open(landmarks)
                    
                    if not eyes_open:
                        if config.DEBUG_MODE:
                            print(f"[VALIDATION] Eyes not open: {eye_reason}")
                        
                        cv2.putText(frame, "ACCESS DENIED: Eyes Must Be Open", 
                                   (50, frame.shape[0] // 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(frame, f"{eye_reason}", 
                                   (50, frame.shape[0] // 2 + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        log_access_event("EYES_CLOSED", reason=eye_reason)
                        self.last_event_text = f"Last: DENIED - Eyes closed"
                        self.last_access_time = current_time
                        return frame
                    
                    # Check for eye occlusion (sunglasses)
                    eyes_occluded, occl_conf, occl_reason = self.eye_detector.detect_eye_occlusion(face, landmarks)
                    
                    if eyes_occluded:
                        if config.DEBUG_MODE:
                            print(f"[SECURITY] Eyes occluded: {occl_reason}")
                        
                        cv2.putText(frame, "ACCESS DENIED: Eyes Occluded", 
                                   (50, frame.shape[0] // 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(frame, f"{occl_reason}", 
                                   (50, frame.shape[0] // 2 + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        log_access_event("EYES_OCCLUDED", reason=occl_reason)
                        self.last_event_text = f"Last: DENIED - Eyes occluded"
                        self.last_access_time = current_time
                        return frame
                
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
                    confidence = 1.0 - (distance / self.recognition_threshold) if matched_name else 0.0
                
                if config.DEBUG_MODE:
                    if matched_name:
                        print(f"[DEBUG] Best match: {matched_name}, Distance: {distance:.4f}, Confidence: {confidence:.2%}, Threshold: {self.recognition_threshold}")
                    else:
                        dist_str = f"{distance:.4f}" if distance is not None else "N/A"
                        print(f"[DEBUG] Best match: None, Distance: {dist_str}, Threshold: {self.recognition_threshold}")
                
                # CRITICAL: Step 6 - Explicit unknown check
                if matched_name is None:
                    # Unknown person - REJECT immediately
                    if config.DEBUG_MODE:
                        dist_str = f"{distance:.4f}" if distance != float('inf') else "inf"
                        print(f"[SECURITY] Unknown person detected! Best distance: {dist_str}, Threshold: {self.recognition_threshold}")
                    
                    # Save unknown face for security review
                    photo_filename = None
                    if config.SAVE_UNKNOWN_FACES:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        photo_filename = f"unknown_{timestamp}_{self.unknown_face_counter}.jpg"
                        save_unknown_face(face, self.unknown_face_counter)
                        self.unknown_face_counter += 1
                    
                    # Display ACCESS DENIED
                    display_access_denied(frame)
                    
                    # Log access event with details
                    log_access_event("UNKNOWN", distance=distance, photo_filename=photo_filename)
                    
                    # Update last event
                    dist_str = f"{distance:.4f}" if distance != float('inf') else "N/A"
                    self.last_event_text = f"Last: DENIED - Unknown (dist: {dist_str})"
                    self.last_access_time = current_time
                    
                    return frame  # Exit immediately - no access!
                
                # Step 7: Known person - check confidence
                if config.DEBUG_MODE:
                    print(f"[DEBUG] Match: {matched_name}, Distance: {distance:.4f}, Confidence: {confidence:.2%}, Threshold: {self.recognition_threshold}")
                
                # Step 8: Confidence check
                if confidence >= config.MIN_MATCH_CONFIDENCE:
                    # ACCESS GRANTED
                    print(f"[SUCCESS] Access Granted: {matched_name} (distance: {distance:.4f}, confidence: {confidence:.2%})")
                    self.last_access_person = matched_name
                    self.last_access_time = current_time
                    self.last_event_text = f"Last: GRANTED - {matched_name} ({confidence:.0%} confidence)"
                    
                    # Display granted message
                    display_access_granted(frame, matched_name)
                    
                    # Log access event
                    log_access_event(matched_name, "GRANTED", confidence=confidence, distance=distance)
                else:
                    # LOW CONFIDENCE - REJECT
                    if config.DEBUG_MODE:
                        print(f"[SECURITY] Low confidence: {confidence:.2%} < {config.MIN_MATCH_CONFIDENCE:.2%}")
                    
                    # Display ACCESS DENIED
                    display_access_denied(frame)
                    
                    # Log access event
                    log_access_event(matched_name, "DENIED - LOW CONFIDENCE", confidence=confidence, distance=distance)
                    
                    # Update last event
                    self.last_event_text = f"Last: DENIED - Low confidence ({confidence:.2%})"
                    self.last_access_time = current_time
            
            except Exception as e:
                print(f"[ERROR] Failed to process face: {e}")
                cv2.putText(frame, f"Processing error: {str(e)}", 
                           (50, frame.shape[0] // 2 - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        except Exception as e:
            print(f"[ERROR] Frame processing error: {e}")
            cv2.putText(frame, f"Detection error: {str(e)}", 
                       (50, frame.shape[0] // 2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
        Run the facial recognition system in on-click mode
        
        Args:
            camera_index: Camera device index (default from config)
        """
        if camera_index is None:
            camera_index = config.CAMERA_INDEX
        
        reconnect_attempts = 0
        
        while True:  # Main loop for camera reconnection
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
                print("[INFO] System running in ON-CLICK mode")
                print("[INFO] Press SPACE to capture and verify")
                print("[INFO] Press 'q' to quit")
                print("[INFO] Access events will be logged to:", config.LOG_FILE_PATH)
                print()
                
                # FPS calculation
                fps = 0
                frame_count = 0
                start_time = time.time()
                
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
                        
                        # Calculate FPS
                        frame_count += 1
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 1.0:
                            fps = frame_count / elapsed_time
                            frame_count = 0
                            start_time = time.time()
                        
                        # Display system status with capture prompt
                        display_frame = frame.copy()
                        uptime = time.time() - self.system_start_time
                        display_system_status(display_frame, fps, uptime, self.last_event_text)
                        
                        # Show "Press SPACE to capture" message
                        cv2.putText(display_frame, "Press SPACE to capture and verify", 
                                   (50, display_frame.shape[0] // 2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        # Show frame
                        cv2.imshow('Security Door Access Control System', display_frame)
                        
                        # Wait for key press
                        key = cv2.waitKey(1) & 0xFF
                        
                        # Check for quit command
                        if key == ord('q'):
                            print("\n[INFO] Shutting down system...")
                            cap.release()
                            cv2.destroyAllWindows()
                            print("[INFO] System shutdown complete")
                            return
                        
                        # Check for capture command (SPACE key)
                        elif key == ord(' '):
                            print("\n[INFO] Capture triggered! Processing image...")
                            
                            # Process the captured frame
                            try:
                                processed_frame = self.process_frame(frame)
                                
                                # Display the result for a few seconds
                                display_system_status(processed_frame, fps, uptime, self.last_event_text)
                                cv2.imshow('Security Door Access Control System', processed_frame)
                                cv2.waitKey(config.ACCESS_RESULT_DISPLAY_TIME * 1000)  # Show result for configured time
                                
                                # Reset last event text after showing result
                                self.last_event_text = ""
                                
                            except Exception as e:
                                print(f"[ERROR] Frame processing failed: {e}")
                    
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
        description='On-Click Facial Recognition System with 100% Confidence Requirement'
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
