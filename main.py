"""
Enhanced Real-Time Facial Recognition System
Production-grade multi-stage pipeline with Advanced Multi-Layer Liveness Detection
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

# Core components
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager
from face_quality_checker import FaceQualityChecker
from face_aligner import FaceAligner
from face_occlusion_detector import FaceOcclusionDetector
from image_preprocessor import ImagePreprocessor

# Advanced components
from advanced_preprocessing_pipeline import AdvancedPreprocessingPipeline
from multi_sample_embedder import MultiSampleEmbedder
from adaptive_threshold_manager import AdaptiveThresholdManager
from intelligent_decision_engine import IntelligentDecisionEngine
from simple_liveness_detector import SimpleLivenessDetector
from advanced_liveness_detector import AdvancedLivenessDetector  # NEW - Multi-layer detection
from adaptive_lighting_adjuster import AdaptiveLightingAdjuster

import config


class EnhancedFaceRecognitionSystem:
    """
    Production-grade face recognition with multi-stage processing
    Now includes 6-layer advanced liveness detection
    """
    
    def __init__(self):
        print("[INFO] Initializing Enhanced Security System...")
        print("="*60)
        
        # Core components
        print("\n[CORE COMPONENTS]")
        self.detector = FaceDetector()
        self.recognizer = FaceRecognitionModel()
        self.db_manager = DatabaseManager()
        self.quality_checker = FaceQualityChecker()
        self.aligner = FaceAligner()
        self.occlusion_detector = FaceOcclusionDetector()
        self.preprocessor = ImagePreprocessor()
        
        # Advanced components
        print("\n[ADVANCED COMPONENTS]")
        
        # Adaptive lighting adjuster
        if config.ENABLE_ADAPTIVE_LIGHTING:
            self.lighting_adjuster = AdaptiveLightingAdjuster()
            self.lighting_mode = config.LIGHTING_MODE
        else:
            self.lighting_adjuster = None
            self.lighting_mode = 'none'
            print("[INFO] Adaptive lighting: DISABLED")
        
        # Advanced preprocessing pipeline
        if config.ENABLE_ADVANCED_PREPROCESSING:
            self.advanced_preprocessor = AdvancedPreprocessingPipeline()
        else:
            self.advanced_preprocessor = None
            print("[INFO] Advanced preprocessing: DISABLED")
        
        # Multi-sample embedder
        if config.ENABLE_MULTI_SAMPLE_EMBEDDING:
            self.multi_embedder = MultiSampleEmbedder(self.recognizer)
        else:
            self.multi_embedder = None
            print("[INFO] Multi-sample embedding: DISABLED")
        
        # Adaptive thresholds
        if config.USE_ADAPTIVE_THRESHOLDS:
            self.threshold_manager = AdaptiveThresholdManager(self.db_manager)
        else:
            self.threshold_manager = None
            print("[INFO] Adaptive thresholds: DISABLED")
        
        # Intelligent decision engine
        if config.USE_INTELLIGENT_DECISIONS:
            self.decision_engine = IntelligentDecisionEngine()
        else:
            self.decision_engine = None
            print("[INFO] Intelligent decisions: DISABLED")
        
        # ================================================
        # LIVENESS DETECTOR (UPDATED - Advanced Multi-Layer)
        # ================================================
        if config.USE_ADVANCED_LIVENESS:
            self.liveness_detector = AdvancedLivenessDetector()
            print("[INFO] Liveness detection: ADVANCED (6-layer)")
            print("  ✓ Texture Analysis")
            print("  ✓ Frequency Analysis")
            print("  ✓ Color Naturalness")
            print("  ✓ Sharpness Detection")
            print("  ✓ Local Variance")
            print("  ✓ Skin Tone Validation")
        elif config.ENABLE_SIMPLE_LIVENESS:
            self.liveness_detector = SimpleLivenessDetector()
            print("[INFO] Liveness detection: SIMPLE (3-layer)")
        else:
            self.liveness_detector = None
            print("[INFO] Liveness detection: DISABLED")
        
        # Stats
        self.total_attempts = 0
        self.successful_grants = 0
        self.frame_count = 0
        
        print("\n" + "="*60)
        print("[INFO] ✅ System initialized successfully!")
        print(f"[INFO] Authorized users: {len(self.db_manager.get_all_users())}")
        print(f"[INFO] Users: {', '.join(self.db_manager.get_all_users())}")
        print("="*60 + "\n")
    
    def process_frame(self, frame):
        """
        Enhanced multi-stage processing pipeline
        
        Stages:
        0. Adaptive lighting correction
        1. Face detection
        2. Quality assessment
        3. Advanced preprocessing
        4. Face alignment
        5. Multi-sample embedding
        6. Adaptive matching
        7. Advanced liveness detection (6-layer)
        8. Intelligent decision
        """
        
        self.frame_count += 1
        display_frame = frame.copy()
        
        # ============================================
        # STAGE 0: ADAPTIVE LIGHTING
        # ============================================
        if self.lighting_adjuster:
            if config.DEBUG_MODE and self.frame_count % 30 == 0:
                print("[STAGE 0] Adaptive Lighting...")
                brightness_before = self.lighting_adjuster.get_brightness_info(frame)
                print(f"  Before: Brightness {brightness_before['mean']:.1f}, "
                      f"Range [{brightness_before['min']}, {brightness_before['max']}]")
            
            # Adjust full frame for display
            if config.APPLY_LIGHTING_TO_FULL_FRAME:
                frame = self.lighting_adjuster.adjust_lighting(frame, mode=self.lighting_mode)
                display_frame = frame.copy()
            
            if config.DEBUG_MODE and self.frame_count % 30 == 0:
                brightness_after = self.lighting_adjuster.get_brightness_info(frame)
                print(f"  After:  Brightness {brightness_after['mean']:.1f}, "
                      f"Range [{brightness_after['min']}, {brightness_after['max']}]")
        
        # ============================================
        # STAGE 1: FACE DETECTION
        # ============================================
        if config.DEBUG_MODE:
            print(f"\n{'='*60}")
            print(f"[FRAME {self.frame_count}] PROCESSING")
            print(f"{'='*60}")
            print("[STAGE 1] Face Detection...")
        
        detections = self.detector.detect_faces(frame)
        
        if len(detections) == 0:
            self._draw_status(display_frame, "No face detected", (0, 0, 255))
            return display_frame
        
        # Use first detection
        det = detections[0]
        box = det['box']
        confidence = det['confidence']
        
        if config.DEBUG_MODE:
            print(f"  ✓ Face detected (confidence: {confidence:.2%})")
        
        # Draw face box
        x, y, w, h = box
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face
        face = self.detector.extract_face(frame, box)
        
        if face.size == 0:
            self._draw_status(display_frame, "Face extraction failed", (0, 0, 255))
            return display_frame
        
        # ============================================
        # FACE-SPECIFIC LIGHTING ADJUSTMENT
        # ============================================
        if self.lighting_adjuster and config.APPLY_LIGHTING_TO_FACES_ONLY:
            if config.DEBUG_MODE:
                print("[LIGHTING] Adjusting face region...")
            
            face = self.lighting_adjuster.adjust_lighting(face, mode=self.lighting_mode)
            
            if config.DEBUG_MODE:
                face_brightness = self.lighting_adjuster.get_brightness_info(face)
                print(f"  Face brightness: {face_brightness['mean']:.1f}")
        
        # ============================================
        # STAGE 2: QUALITY ASSESSMENT
        # ============================================
        if config.DEBUG_MODE:
            print("[STAGE 2] Quality Assessment...")
        
        try:
            quality_result = self.quality_checker.get_quality_score(face)
            
            # Handle different return formats
            if isinstance(quality_result, tuple) and len(quality_result) == 2:
                quality_score, quality_details = quality_result
            elif isinstance(quality_result, tuple) and len(quality_result) == 1:
                quality_score = quality_result[0]
                quality_details = {}
            else:
                quality_score = quality_result
                quality_details = {}
            
            if config.DEBUG_MODE:
                print(f"  ✓ Quality score: {quality_score:.1f}/100")
            
            if quality_score < config.OVERALL_QUALITY_THRESHOLD:
                reason = f"Quality too low: {quality_score:.0f}/100"
                self._draw_status(display_frame, reason, (0, 165, 255))
                return display_frame
                
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"  ⚠ Quality check failed: {e}")
            # Continue with default quality
            quality_score = 100.0
            quality_details = {}
        
        # ============================================
        # STAGE 3: OCCLUSION CHECK
        # ============================================
        if config.ENABLE_MASK_DETECTION:
            if config.DEBUG_MODE:
                print("[STAGE 3] Occlusion Detection...")
            
            try:
                occlusion_result = self.occlusion_detector.detect_occlusion(face)
                
                # Handle different return formats
                if isinstance(occlusion_result, dict):
                    has_mask = occlusion_result.get('has_mask', False)
                    mask_confidence = occlusion_result.get('mask_confidence', 0.0)
                else:
                    has_mask = occlusion_result
                    mask_confidence = 1.0 if has_mask else 0.0
                
                # Only block if HIGH confidence mask detected
                if has_mask and mask_confidence > config.MASK_CONFIDENCE_THRESHOLD:
                    reason = f"Mask detected ({mask_confidence:.1%})"
                    if config.DEBUG_MODE:
                        print(f"  ✗ {reason}")
                    self._draw_status(display_frame, f"ACCESS DENIED: {reason}", (0, 0, 255))
                    return display_frame
                elif has_mask:
                    if config.DEBUG_MODE:
                        print(f"  ⚠ Possible mask ({mask_confidence:.1%}), allowing (below threshold)")
                else:
                    if config.DEBUG_MODE:
                        print(f"  ✓ No mask detected")
                    
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"  ⚠ Occlusion check failed: {e}, continuing...")
        
        # ============================================
        # STAGE 4: ADVANCED PREPROCESSING
        # ============================================
        if self.advanced_preprocessor:
            if config.DEBUG_MODE:
                print(f"[STAGE 4] Advanced Preprocessing (mode: {config.PREPROCESSING_MODE})...")
            
            try:
                face = self.advanced_preprocessor.process(face, mode=config.PREPROCESSING_MODE)
                
                if config.DEBUG_MODE:
                    print(f"  ✓ Preprocessing applied")
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"  ⚠ Preprocessing failed: {e}, using original")
        
        # ============================================
        # STAGE 5: FACE ALIGNMENT
        # ============================================
        if config.DEBUG_MODE:
            print("[STAGE 5] Face Alignment...")
        
        if 'keypoints' in det and det['keypoints']:
            try:
                face = self.aligner.align_face(face, det['keypoints'])
                if config.DEBUG_MODE:
                    print(f"  ✓ Face aligned")
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"  ⚠ Alignment failed: {e}, using resize")
                face = cv2.resize(face, config.FACE_SIZE)
        else:
            face = cv2.resize(face, config.FACE_SIZE)
        
        # ============================================
        # STAGE 6: MULTI-SAMPLE EMBEDDING GENERATION
        # ============================================
        if config.DEBUG_MODE:
            print("[STAGE 6] Embedding Generation...")
        
        try:
            if self.multi_embedder:
                if config.DEBUG_MODE:
                    print(f"  Generating {config.NUM_EMBEDDING_SAMPLES} sample embeddings...")
                
                embedding = self.multi_embedder.generate_robust_embedding(
                    face, 
                    num_samples=config.NUM_EMBEDDING_SAMPLES
                )
                
                if config.DEBUG_MODE:
                    print(f"  ✓ Multi-sample embedding generated")
            else:
                embedding = self.recognizer.get_embedding(face)
                if config.DEBUG_MODE:
                    print(f"  ✓ Single embedding generated")
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"  ⚠ Embedding generation failed: {e}, using single")
            embedding = self.recognizer.get_embedding(face)
        
        # ============================================
        # STAGE 7: ADAPTIVE DATABASE MATCHING
        # ============================================
        if config.DEBUG_MODE:
            print("[STAGE 7] Database Matching...")
        
        try:
            if self.threshold_manager:
                # Use adaptive thresholds
                matched_name, distance, threshold = self._adaptive_match(embedding)
                
                if config.DEBUG_MODE:
                    if matched_name:
                        print(f"  ✓ Match: {matched_name}")
                        print(f"    Distance: {distance:.4f}")
                        print(f"    User threshold: {threshold:.4f}")
                    else:
                        print(f"  ✗ No match (best distance: {distance:.4f})")
            else:
                # Standard matching
                matched_name, distance = self.db_manager.find_match(embedding, self.recognizer)
                threshold = config.DEFAULT_RECOGNITION_THRESHOLD
                
                if config.DEBUG_MODE:
                    if matched_name:
                        print(f"  ✓ Match: {matched_name}, Distance: {distance:.4f}")
                    else:
                        print(f"  ✗ No match")
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"  ✗ Matching failed: {e}")
            matched_name = None
            distance = float('inf')
            threshold = config.DEFAULT_RECOGNITION_THRESHOLD
        
        if not matched_name:
            self._draw_status(display_frame, "Unknown Person", (0, 0, 255))
            self._log_access("UNKNOWN", "DENIED", 0.0, distance)
            return display_frame
        
        # ============================================
        # STAGE 8: ADVANCED LIVENESS DETECTION (UPDATED)
        # ============================================
        liveness_score = 1.0  # Default (assume live)
        
        if self.liveness_detector:
            if config.DEBUG_MODE:
                if config.USE_ADVANCED_LIVENESS:
                    print("[STAGE 8] Advanced Liveness Detection (6-layer)...")
                else:
                    print("[STAGE 8] Simple Liveness Detection...")
            
            try:
                is_live, liveness_score, liveness_details = self.liveness_detector.check_liveness(face)
                
                if config.DEBUG_MODE:
                    if config.USE_ADVANCED_LIVENESS and config.SHOW_LIVENESS_BREAKDOWN:
                        # Show detailed 6-layer breakdown
                        print(f"\n{self.liveness_detector.explain_decision(liveness_details)}\n")
                    else:
                        # Simple summary
                        print(f"  Liveness: {'LIVE ✅' if is_live else 'FAKE ❌'} ({liveness_score:.1%})")
                        
                        if config.USE_ADVANCED_LIVENESS:
                            # Show component scores
                            print(f"  Component Scores:")
                            for component, score in liveness_details['scores'].items():
                                status = '✅' if score > 0.35 else '⚠️'
                                print(f"    {component:12} {status} {score:.1%}")
                            
                            # Show critical checks
                            if 'critical_checks' in liveness_details:
                                print(f"  Critical Checks:")
                                for check, passed in liveness_details['critical_checks'].items():
                                    status = '✅ PASS' if passed else '❌ FAIL'
                                    print(f"    {check.capitalize():12} {status}")
                
                if not is_live:
                    # Get failure reason
                    if config.USE_ADVANCED_LIVENESS:
                        reason = liveness_details.get('decision_reason', 'Liveness check failed')
                        # Shorten reason for display
                        reason_short = reason[:50] + "..." if len(reason) > 50 else reason
                    else:
                        reason_short = f"Liveness check failed ({liveness_score:.1%})"
                    
                    self._draw_status(display_frame, f"ACCESS DENIED: {reason_short}", (0, 0, 255))
                    self._log_access(matched_name, "DENIED - LIVENESS", liveness_score, distance)
                    
                    if config.DEBUG_MODE:
                        print(f"  ❌ DENIED - Reason: {reason if config.USE_ADVANCED_LIVENESS else reason_short}")
                    
                    return display_frame
                else:
                    if config.DEBUG_MODE:
                        print(f"  ✅ Liveness check passed")
                
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"  ⚠ Liveness check failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Fallback behavior
                if config.LIVENESS_FALLBACK_ENABLED:
                    print("  ℹ️  Using fallback: assume live")
                    liveness_score = 1.0
                else:
                    # Fail secure
                    self._draw_status(display_frame, "ACCESS DENIED: Liveness error", (0, 0, 255))
                    self._log_access(matched_name, "DENIED - LIVENESS ERROR", 0.0, distance)
                    return display_frame
        
        # ============================================
        # STAGE 9: INTELLIGENT DECISION ENGINE
        # ============================================
        if config.DEBUG_MODE:
            print("[STAGE 9] Decision Making...")
        
        try:
            if self.decision_engine:
                # Use intelligent decision engine
                decision, overall_score, details = self.decision_engine.make_decision(
                    match_result=(matched_name, distance, threshold),
                    quality_score=quality_score,
                    liveness_score=liveness_score,
                    temporal_confidence=1.0
                )
                
                if config.DEBUG_MODE:
                    print(f"\n{self.decision_engine.explain_decision(details)}\n")
                
                # Execute decision
                if decision == 'GRANT':
                    self._draw_access_granted(display_frame, matched_name, overall_score)
                    self._log_access(matched_name, "GRANTED", overall_score, distance)
                    self.successful_grants += 1
                    
                    # Update adaptive threshold (online learning)
                    if self.threshold_manager:
                        self.threshold_manager.update_threshold(matched_name, distance)
                    
                elif decision == 'MFA_REQUIRED':
                    self._draw_mfa_required(display_frame, matched_name, overall_score)
                    self._log_access(matched_name, "MFA_REQUIRED", overall_score, distance)
                    
                else:  # DENY
                    reason = details.get('reason', 'Insufficient confidence')
                    self._draw_status(display_frame, f"ACCESS DENIED: {reason}", (0, 0, 255))
                    self._log_access(matched_name, f"DENIED - {reason}", overall_score, distance)
            
            else:
                # Simple confidence check
                base_confidence = 1.0 - (distance / threshold)
                
                if config.DEBUG_MODE:
                    print(f"  Confidence: {base_confidence:.2%}")
                    print(f"  Threshold: {config.MIN_MATCH_CONFIDENCE:.2%}")
                
                if base_confidence >= config.MIN_MATCH_CONFIDENCE:
                    self._draw_access_granted(display_frame, matched_name, base_confidence)
                    self._log_access(matched_name, "GRANTED", base_confidence, distance)
                    self.successful_grants += 1
                else:
                    reason = f"Low confidence: {base_confidence:.1%}"
                    self._draw_status(display_frame, f"ACCESS DENIED: {reason}", (0, 0, 255))
                    self._log_access(matched_name, f"DENIED - {reason}", base_confidence, distance)
        
        except Exception as e:
            if config.DEBUG_MODE:
                print(f"  ✗ Decision making failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Fallback simple check
            base_confidence = 1.0 - (distance / threshold)
            if base_confidence >= config.MIN_MATCH_CONFIDENCE:
                self._draw_access_granted(display_frame, matched_name, base_confidence)
                self._log_access(matched_name, "GRANTED", base_confidence, distance)
                self.successful_grants += 1
            else:
                self._draw_status(display_frame, "Error in decision engine", (0, 0, 255))
        
        self.total_attempts += 1
        
        return display_frame
    
    def _adaptive_match(self, embedding):
        """
        Match against database using adaptive thresholds
        
        Returns:
            (matched_name, distance, threshold)
        """
        best_match = None
        best_distance = float('inf')
        best_threshold = config.DEFAULT_RECOGNITION_THRESHOLD
        
        users = self.db_manager.get_all_users()
        
        for user in users:
            user_embeddings = self.db_manager.get_user_embeddings(user)
            user_threshold = self.threshold_manager.get_threshold(user)
            
            for user_emb in user_embeddings:
                dist = np.linalg.norm(embedding - user_emb)
                
                # Check if within user's threshold AND better than current best
                if dist < user_threshold and dist < best_distance:
                    best_distance = dist
                    best_match = user
                    best_threshold = user_threshold
        
        return best_match, best_distance, best_threshold
    
    def _draw_access_granted(self, frame, name, confidence):
        """Draw access granted message"""
        h, w = frame.shape[:2]
        
        # Green overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Message
        cv2.putText(frame, "ACCESS GRANTED", 
                   (w//2 - 200, h//2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.putText(frame, f"Welcome, {name}!", 
                   (w//2 - 150, h//2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.1%}", 
                   (w//2 - 120, h//2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _draw_status(self, frame, message, color):
        """Draw status message"""
        h, w = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text
        cv2.putText(frame, message, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_mfa_required(self, frame, name, confidence):
        """Draw MFA required message"""
        h, w = frame.shape[:2]
        
        # Orange overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Message
        cv2.putText(frame, "MFA REQUIRED", 
                   (w//2 - 150, h//2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        
        cv2.putText(frame, f"User: {name}", 
                   (w//2 - 100, h//2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.1%}", 
                   (w//2 - 120, h//2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    def _draw_brightness_indicator(self, frame):
        """Draw brightness indicator bar"""
        if not self.lighting_adjuster or not config.SHOW_BRIGHTNESS_INDICATOR:
            return
        
        brightness_info = self.lighting_adjuster.get_brightness_info(frame)
        brightness = brightness_info['mean']
        
        h, w = frame.shape[:2]
        
        # Brightness bar position
        bar_x = w - 50
        bar_y = 50
        bar_h = 200
        bar_w = 30
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                     (0, 0, 0), -1)
        
        # Fill based on brightness (0-255 mapped to bar height)
        fill_h = int((brightness / 255.0) * bar_h)
        fill_y = bar_y + bar_h - fill_h
        
        # Color based on brightness
        if brightness < 80:
            color = (0, 0, 255)  # Red - too dark
        elif brightness > 180:
            color = (0, 165, 255)  # Orange - too bright
        else:
            color = (0, 255, 0)  # Green - good
        
        cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_w, bar_y + bar_h), 
                     color, -1)
        
        # Target line (128)
        target_y = bar_y + bar_h - int((config.TARGET_BRIGHTNESS / 255.0) * bar_h)
        cv2.line(frame, (bar_x - 5, target_y), (bar_x + bar_w + 5, target_y), 
                (255, 255, 255), 2)
        
        # Text
        cv2.putText(frame, f"{brightness:.0f}", (bar_x - 15, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, "Brightness", (bar_x - 40, bar_y + bar_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _log_access(self, user, result, confidence, distance):
        """Log access attempt"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"[{timestamp}] {result} - User: {user}, Confidence: {confidence:.2%}, Distance: {distance:.4f}\n"
        
        try:
            with open(config.ACCESS_LOG_FILE, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"[WARNING] Logging failed: {e}")
        
        if config.DEBUG_MODE:
            print(f"[LOG] {log_entry.strip()}")
    
    def run(self):
        """Run the system with adaptive lighting and advanced liveness"""
        print("[INFO] Starting camera...")
        
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return
        
        # ================================================
        # CAMERA AUTO SETTINGS
        # ================================================
        if config.ENABLE_CAMERA_AUTO_EXPOSURE:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            print("[INFO] Camera auto exposure enabled")
        
        if config.ENABLE_CAMERA_AUTO_WB:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            print("[INFO] Camera auto white balance enabled")
        
        print("[INFO] Camera opened successfully")
        
        # Print controls
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS:")
        print("="*60)
        print("  SPACE - Process frame and verify")
        print("  Q     - Quit system")
        
        if config.ENABLE_KEYBOARD_CONTROLS and self.lighting_adjuster:
            print("\n  Lighting Controls:")
            print("  +/=   - Increase brightness")
            print("  -/_   - Decrease brightness")
            print("  A     - Auto lighting mode")
            print("  B     - Balance lighting mode")
            print("  N     - No lighting adjustment")
        
        print("  D     - Toggle debug mode")
        print("  I     - Toggle brightness indicator")
        
        if config.USE_ADVANCED_LIVENESS:
            print("  V     - Toggle liveness breakdown view")
        
        print("="*60 + "\n")
        
        # Gamma adjustment (for keyboard controls)
        current_gamma = config.DEFAULT_GAMMA
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Apply lighting to full frame if enabled
            if self.lighting_adjuster and config.APPLY_LIGHTING_TO_FULL_FRAME:
                frame = self.lighting_adjuster.adjust_lighting(frame, mode=self.lighting_mode)
            
            # Show live feed
            display_frame = frame.copy()
            
            # Add instructions
            cv2.putText(display_frame, "Press SPACE to verify", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show liveness mode
            if self.liveness_detector:
                liveness_mode = "Advanced (6-layer)" if config.USE_ADVANCED_LIVENESS else "Simple (3-layer)"
                cv2.putText(display_frame, f"Liveness: {liveness_mode}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show lighting mode
            if self.lighting_adjuster:
                cv2.putText(display_frame, f"Lighting: {self.lighting_mode}", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show stats
            if self.total_attempts > 0:
                success_rate = (self.successful_grants / self.total_attempts) * 100
                stats = f"Attempts: {self.total_attempts} | Success: {success_rate:.1f}%"
                cv2.putText(display_frame, stats, 
                           (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw brightness indicator
            if config.SHOW_BRIGHTNESS_INDICATOR:
                self._draw_brightness_indicator(display_frame)
            
            cv2.imshow('Enhanced Face Recognition System', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # ================================================
            # KEYBOARD CONTROLS
            # ================================================
            if key == ord(' '):
                # Process frame
                print("\n[INFO] 🔍 Processing frame...")
                try:
                    result_frame = self.process_frame(frame)
                    
                    # Show result for 3 seconds
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        cv2.imshow('Enhanced Face Recognition System', result_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                except Exception as e:
                    print(f"[ERROR] Processing failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif key == ord('q'):
                print("\n[INFO] Shutting down...")
                break
            
            # Lighting controls
            elif config.ENABLE_KEYBOARD_CONTROLS and self.lighting_adjuster:
                if key == ord('+') or key == ord('='):
                    current_gamma = min(config.MAX_GAMMA, current_gamma + 0.1)
                    print(f"[LIGHTING] Gamma: {current_gamma:.1f} (brighter)")
                
                elif key == ord('-') or key == ord('_'):
                    current_gamma = max(config.MIN_GAMMA, current_gamma - 0.1)
                    print(f"[LIGHTING] Gamma: {current_gamma:.1f} (darker)")
                
                elif key == ord('a'):
                    self.lighting_mode = 'auto'
                    print("[LIGHTING] Auto mode activated")
                
                elif key == ord('b'):
                    self.lighting_mode = 'balance'
                    print("[LIGHTING] Balance mode activated")
                
                elif key == ord('n'):
                    self.lighting_mode = 'none'
                    print("[LIGHTING] Adjustment disabled")
                
                elif key == ord('d'):
                    config.DEBUG_MODE = not config.DEBUG_MODE
                    print(f"[DEBUG] {'Enabled' if config.DEBUG_MODE else 'Disabled'}")
                
                elif key == ord('i'):
                    config.SHOW_BRIGHTNESS_INDICATOR = not config.SHOW_BRIGHTNESS_INDICATOR
                    print(f"[INDICATOR] {'Shown' if config.SHOW_BRIGHTNESS_INDICATOR else 'Hidden'}")
                
                elif key == ord('v') and config.USE_ADVANCED_LIVENESS:
                    config.SHOW_LIVENESS_BREAKDOWN = not config.SHOW_LIVENESS_BREAKDOWN
                    print(f"[LIVENESS] Breakdown view: {'Enabled' if config.SHOW_LIVENESS_BREAKDOWN else 'Disabled'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total access attempts: {self.total_attempts}")
        print(f"Successful grants: {self.successful_grants}")
        if self.total_attempts > 0:
            print(f"Success rate: {(self.successful_grants/self.total_attempts)*100:.1f}%")
        print("="*60)


if __name__ == "__main__":
    system = EnhancedFaceRecognitionSystem()
    system.run()