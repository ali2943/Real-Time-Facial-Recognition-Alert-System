"""
Complete Pipeline Test
Tests all stages of recognition
"""

import cv2
import numpy as np
from datetime import datetime

# Import all components
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager
from face_quality_checker import FaceQualityChecker
from adaptive_threshold_manager import AdaptiveThresholdManager
from intelligent_decision_engine import IntelligentDecisionEngine
from simple_liveness_detector import SimpleLivenessDetector
import config

class PipelineTester:
    def __init__(self):
        print("="*60)
        print("INITIALIZING PIPELINE TEST")
        print("="*60)
        
        self.detector = FaceDetector()
        self.recognizer = FaceRecognitionModel()
        self.db = DatabaseManager()
        
        # Optional components
        try:
            self.quality_checker = FaceQualityChecker()
        except:
            self.quality_checker = None
        
        if config.USE_ADAPTIVE_THRESHOLDS:
            self.threshold_manager = AdaptiveThresholdManager(self.db)
        else:
            self.threshold_manager = None
        
        if config.USE_INTELLIGENT_DECISIONS:
            self.decision_engine = IntelligentDecisionEngine()
        else:
            self.decision_engine = None
        
        if config.ENABLE_SIMPLE_LIVENESS:
            self.liveness_detector = SimpleLivenessDetector()
        else:
            self.liveness_detector = None
        
        print("\n[COMPONENTS LOADED]")
        print(f"  Quality Checker: {'✅' if self.quality_checker else '❌'}")
        print(f"  Adaptive Thresholds: {'✅' if self.threshold_manager else '❌'}")
        print(f"  Decision Engine: {'✅' if self.decision_engine else '❌'}")
        print(f"  Liveness Detector: {'✅' if self.liveness_detector else '❌'}")
        
        users = self.db.get_all_users()
        print(f"\n[DATABASE]")
        print(f"  Users: {len(users)}")
        for user in users:
            embeddings = self.db.get_user_embeddings(user)
            print(f"    {user}: {len(embeddings)} samples")
        
        print("="*60 + "\n")
    
    def test_single_frame(self, frame):
        """Test complete pipeline on single frame"""
        
        print("\n" + "="*60)
        print(f"PIPELINE TEST - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        results = {}
        
        # STAGE 1: Detection
        print("\n[STAGE 1] Face Detection")
        detections = self.detector.detect_faces(frame)
        
        if len(detections) == 0:
            print("  ❌ No face detected")
            results['stage'] = 1
            results['status'] = 'FAIL'
            results['reason'] = 'No face'
            return results
        
        print(f"  ✅ Face detected (confidence: {detections[0]['confidence']:.1%})")
        
        # Extract face
        box = detections[0]['box']
        face = self.detector.extract_face(frame, box)
        face = cv2.resize(face, (112, 112))
        
        # STAGE 2: Quality Check
        print("\n[STAGE 2] Quality Assessment")
        
        if self.quality_checker:
            try:
                # Try different method names
                if hasattr(self.quality_checker, 'check_quality'):
                    quality_result = self.quality_checker.check_quality(face)
                elif hasattr(self.quality_checker, 'assess_quality'):
                    quality_result = self.quality_checker.assess_quality(face)
                else:
                    quality_result = 100.0
                
                if isinstance(quality_result, tuple):
                    quality_score = quality_result[0]
                else:
                    quality_score = quality_result
                
                print(f"  Quality: {quality_score:.1f}/100")
                
                if quality_score < config.OVERALL_QUALITY_THRESHOLD:
                    print(f"  ❌ Below threshold ({config.OVERALL_QUALITY_THRESHOLD})")
                    results['stage'] = 2
                    results['status'] = 'FAIL'
                    results['reason'] = 'Low quality'
                    results['quality'] = quality_score
                    return results
                else:
                    print(f"  ✅ Above threshold")
            except Exception as e:
                print(f"  ⚠️  Quality check failed: {e}")
                quality_score = 100.0
        else:
            print(f"  ⚠️  No quality checker")
            quality_score = 100.0
        
        results['quality'] = quality_score
        
        # STAGE 3: Embedding
        print("\n[STAGE 3] Embedding Generation")
        embedding = self.recognizer.get_embedding(face)
        print(f"  ✅ Embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.2f}")
        
        # STAGE 4: Matching
        print("\n[STAGE 4] Database Matching")
        
        if self.threshold_manager:
            # Adaptive matching
            best_match = None
            best_distance = float('inf')
            best_threshold = config.DEFAULT_RECOGNITION_THRESHOLD
            
            for user in self.db.get_all_users():
                user_embeddings = self.db.get_user_embeddings(user)
                user_threshold = self.threshold_manager.get_threshold(user)
                
                for user_emb in user_embeddings:
                    dist = np.linalg.norm(embedding - user_emb)
                    if dist < user_threshold and dist < best_distance:
                        best_distance = dist
                        best_match = user
                        best_threshold = user_threshold
            
            matched_name = best_match
            distance = best_distance
            threshold = best_threshold
            
            print(f"  Match: {matched_name if matched_name else 'None'}")
            print(f"  Distance: {distance:.4f}")
            print(f"  Threshold: {threshold:.4f}")
        else:
            # Standard matching
            matched_name, distance = self.db.find_match(embedding, self.recognizer)
            threshold = config.DEFAULT_RECOGNITION_THRESHOLD
            
            print(f"  Match: {matched_name if matched_name else 'None'}")
            print(f"  Distance: {distance:.4f}")
            print(f"  Threshold: {threshold:.4f}")
        
        if not matched_name:
            print(f"  ❌ No match in database")
            results['stage'] = 4
            results['status'] = 'FAIL'
            results['reason'] = 'Unknown person'
            results['distance'] = distance
            return results
        
        print(f"  ✅ Matched: {matched_name}")
        results['matched_name'] = matched_name
        results['distance'] = distance
        results['threshold'] = threshold
        
        # STAGE 5: Liveness
        print("\n[STAGE 5] Liveness Detection")
        
        if self.liveness_detector:
            is_live, liveness_score, liveness_details = self.liveness_detector.check_liveness(face)
            
            print(f"  Liveness: {liveness_score:.1%}")
            print(f"  Details: {liveness_details}")
            
            if not is_live:
                print(f"  ❌ Liveness failed (threshold: {config.LIVENESS_THRESHOLD:.1%})")
                results['stage'] = 5
                results['status'] = 'FAIL'
                results['reason'] = 'Liveness check failed'
                results['liveness_score'] = liveness_score
                return results
            else:
                print(f"  ✅ Liveness passed")
        else:
            print(f"  ⚠️  Liveness detection disabled")
            liveness_score = 1.0
        
        results['liveness_score'] = liveness_score
        
        # STAGE 6: Decision
        print("\n[STAGE 6] Final Decision")
        
        if self.decision_engine:
            decision, overall_score, details = self.decision_engine.make_decision(
                match_result=(matched_name, distance, threshold),
                quality_score=quality_score,
                liveness_score=liveness_score,
                temporal_confidence=1.0
            )
            
            print(f"  Overall Score: {overall_score:.1%}")
            print(f"  Decision: {decision}")
            print(f"  Reason: {details.get('reason', 'N/A')}")
            
            # Show breakdown
            print(f"\n  Component Scores:")
            for component, score in details['component_scores'].items():
                weight = details['weights'][component]
                contribution = score * weight
                print(f"    {component.capitalize():12} {score:.1%} × {weight:.0%} = {contribution:.1%}")
            
            results['decision'] = decision
            results['overall_score'] = overall_score
            results['stage'] = 6
            results['status'] = 'SUCCESS' if decision == 'GRANT' else 'FAIL'
            results['reason'] = details.get('reason', '')
        else:
            # Simple decision
            confidence = 1.0 - (distance / threshold)
            
            print(f"  Confidence: {confidence:.1%}")
            
            if confidence >= config.MIN_MATCH_CONFIDENCE:
                decision = 'GRANT'
                print(f"  ✅ GRANT")
            else:
                decision = 'DENY'
                print(f"  ❌ DENY (below threshold {config.MIN_MATCH_CONFIDENCE:.1%})")
            
            results['decision'] = decision
            results['overall_score'] = confidence
            results['stage'] = 6
            results['status'] = 'SUCCESS' if decision == 'GRANT' else 'FAIL'
        
        print("="*60)
        
        return results
    
    def run_interactive(self):
        """Interactive pipeline testing"""
        
        print("\nSTARTING INTERACTIVE TEST")
        print("Press SPACE to test current frame")
        print("Press P for photo attack test mode")
        print("Press Q to quit\n")
        
        cap = cv2.VideoCapture(0)
        photo_test_mode = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            
            # Instructions
            if photo_test_mode:
                cv2.putText(display, "PHOTO TEST MODE - Show a photo", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Press SPACE to test pipeline", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display, "P - Photo test | Q - Quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Pipeline Test', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                results = self.test_single_frame(frame)
                
                # Show results on frame
                result_frame = frame.copy()
                
                if results['status'] == 'SUCCESS':
                    color = (0, 255, 0)
                    text = f"GRANTED - {results['matched_name']}"
                else:
                    color = (0, 0, 255)
                    text = f"DENIED - {results['reason']}"
                
                cv2.putText(result_frame, text, 
                           (50, result_frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                
                cv2.imshow('Pipeline Test', result_frame)
                cv2.waitKey(3000)
            
            elif key == ord('p'):
                photo_test_mode = not photo_test_mode
                print(f"\nPhoto test mode: {'ON' if photo_test_mode else 'OFF'}")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = PipelineTester()
    tester.run_interactive()