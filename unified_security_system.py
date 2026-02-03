"""
Unified 3-Tier Anti-Spoofing Security System
Maximum protection against photo/video attacks
"""

import cv2
import numpy as np
import time

from advanced_liveness_detector import AdvancedLivenessDetector
from motion_liveness_detector import MotionLivenessDetector
from challenge_response_liveness import ChallengeResponseLiveness
import config


class UnifiedSecuritySystem:
    """
    3-Tier security system:
    1. Advanced Liveness (passive - texture/frequency/color)
    2. Motion Detection (passive - requires natural movement)
    3. Challenge-Response (active - user must respond)
    """
    
    def __init__(self):
        print("[INFO] Initializing Unified Security System...")
        print("="*60)
        
        # Tier 1: Advanced Liveness
        if config.USE_ADVANCED_LIVENESS:
            self.advanced_liveness = AdvancedLivenessDetector()
            print("[TIER 1] Advanced Liveness: ENABLED")
        else:
            self.advanced_liveness = None
            print("[TIER 1] Advanced Liveness: DISABLED")
        
        # Tier 2: Motion Detection
        if config.ENABLE_MOTION_LIVENESS:
            self.motion_detector = MotionLivenessDetector(
                buffer_size=config.MOTION_BUFFER_SIZE
            )
            print("[TIER 2] Motion Detection: ENABLED")
        else:
            self.motion_detector = None
            print("[TIER 2] Motion Detection: DISABLED")
        
        # Tier 3: Challenge-Response
        if config.ENABLE_CHALLENGE_RESPONSE:
            self.challenge_system = ChallengeResponseLiveness()
            print("[TIER 3] Challenge-Response: ENABLED")
        else:
            self.challenge_system = None
            print("[TIER 3] Challenge-Response: DISABLED")
        
        # State
        self.tier1_passed = False
        self.tier2_passed = False
        self.tier3_passed = False
        
        self.tier1_score = 0.0
        self.tier2_score = 0.0
        self.tier3_score = 0.0
        
        print("="*60)
        print(f"Security Level: {config.SECURITY_LEVEL}")
        print(f"Require All Tiers: {config.REQUIRE_ALL_TIERS}")
        print("="*60 + "\n")
    
    def check_security(self, face_img, embedding=None, debug=False):
        """
        Run all security tiers
        
        Args:
            face_img: Face image (BGR, 112x112)
            embedding: Face embedding (optional, for motion detection)
            debug: Print debug info
            
        Returns:
            (is_secure, overall_score, details)
        """
        
        results = {
            'tier1': None,
            'tier2': None,
            'tier3': None,
            'passed': False,
            'overall_score': 0.0,
            'reason': ''
        }
        
        # ==================== TIER 1: Advanced Liveness ====================
        if self.advanced_liveness:
            if debug:
                print("[TIER 1] Running Advanced Liveness...")
            
            is_live, score, details = self.advanced_liveness.check_liveness(face_img)
            
            self.tier1_passed = is_live
            self.tier1_score = score
            
            results['tier1'] = {
                'passed': is_live,
                'score': score,
                'details': details
            }
            
            if debug:
                print(f"  Result: {'PASS ✅' if is_live else 'FAIL ❌'} ({score:.1%})")
            
            # If REQUIRE_ALL_TIERS and tier 1 fails, reject immediately
            if config.REQUIRE_ALL_TIERS and not is_live:
                results['passed'] = False
                results['overall_score'] = score
                results['reason'] = 'Advanced liveness check failed'
                return False, score, results
        
        # ==================== TIER 2: Motion Detection ====================
        if self.motion_detector:
            # Add frame to buffer
            self.motion_detector.add_frame(face_img, embedding)
            
            # Check if ready
            is_live, score, details = self.motion_detector.check_liveness()
            
            if details.get('status') == 'waiting':
                # Still collecting frames
                results['tier2'] = {
                    'status': 'waiting',
                    'frames_collected': details['frames_collected'],
                    'frames_needed': details['frames_needed']
                }
                
                if debug:
                    print(f"[TIER 2] Collecting motion data... {details['frames_collected']}/{details['frames_needed']}")
                
                # Not ready yet - cannot make final decision
                results['passed'] = False
                results['reason'] = 'Motion analysis in progress'
                return None, 0.0, results
            
            else:
                # Analysis complete
                self.tier2_passed = is_live
                self.tier2_score = score
                
                results['tier2'] = {
                    'passed': is_live,
                    'score': score,
                    'details': details
                }
                
                if debug:
                    print(f"[TIER 2] Motion Detection: {'PASS ✅' if is_live else 'FAIL ❌'} ({score:.1%})")
                
                # If REQUIRE_ALL_TIERS and tier 2 fails, reject
                if config.REQUIRE_ALL_TIERS and not is_live:
                    results['passed'] = False
                    results['overall_score'] = score
                    results['reason'] = 'No motion detected - possible photo/video'
                    return False, score, results
        
        # ==================== TIER 3: Challenge-Response ====================
        if self.challenge_system:
            # Start challenge if not active
            if self.challenge_system.current_challenge is None:
                challenge = self.challenge_system.start_challenge()
                
                if debug:
                    instruction = self.challenge_system.get_challenge_instruction()
                    print(f"[TIER 3] Challenge: {instruction}")
                
                results['tier3'] = {
                    'status': 'challenge_started',
                    'challenge': challenge,
                    'instruction': self.challenge_system.get_challenge_instruction()
                }
                
                # Cannot complete yet
                results['passed'] = False
                results['reason'] = 'Awaiting challenge response'
                return None, 0.0, results
            
            # Check response
            responded, score, details = self.challenge_system.check_response(face_img)
            
            if details.get('status') == 'baseline_set':
                # Baseline captured, continue
                results['tier3'] = {
                    'status': 'monitoring',
                    'challenge': self.challenge_system.current_challenge
                }
                return None, 0.0, results
            
            elif details.get('error') == 'Timeout':
                # Challenge timeout
                results['tier3'] = {
                    'passed': False,
                    'score': 0.0,
                    'reason': 'Challenge timeout'
                }
                
                if debug:
                    print("[TIER 3] Challenge TIMEOUT ❌")
                
                if config.REQUIRE_ALL_TIERS:
                    results['passed'] = False
                    results['reason'] = 'Challenge timeout'
                    return False, 0.0, results
            
            elif responded:
                # Challenge passed
                self.tier3_passed = True
                self.tier3_score = score
                
                results['tier3'] = {
                    'passed': True,
                    'score': score,
                    'details': details
                }
                
                if debug:
                    print(f"[TIER 3] Challenge PASSED ✅")
            
            else:
                # Still checking
                results['tier3'] = {
                    'status': 'checking',
                    'challenge': self.challenge_system.current_challenge
                }
                return None, 0.0, results
        
        # ==================== FINAL DECISION ====================
        
        if config.REQUIRE_ALL_TIERS:
            # ALL tiers must pass
            all_passed = True
            
            if self.advanced_liveness and not self.tier1_passed:
                all_passed = False
            if self.motion_detector and not self.tier2_passed:
                all_passed = False
            if self.challenge_system and not self.tier3_passed:
                all_passed = False
            
            overall_score = min(self.tier1_score, self.tier2_score, self.tier3_score)
            
            results['passed'] = all_passed
            results['overall_score'] = overall_score
            results['reason'] = 'All tiers passed' if all_passed else 'One or more tiers failed'
            
            return all_passed, overall_score, results
        
        else:
            # Weighted combination
            total_weight = 0.0
            weighted_score = 0.0
            
            if self.advanced_liveness:
                weighted_score += self.tier1_score * config.TIER_WEIGHTS['advanced_liveness']
                total_weight += config.TIER_WEIGHTS['advanced_liveness']
            
            if self.motion_detector:
                weighted_score += self.tier2_score * config.TIER_WEIGHTS['motion']
                total_weight += config.TIER_WEIGHTS['motion']
            
            if self.challenge_system:
                weighted_score += self.tier3_score * config.TIER_WEIGHTS['challenge']
                total_weight += config.TIER_WEIGHTS['challenge']
            
            if total_weight > 0:
                overall_score = weighted_score / total_weight
            else:
                overall_score = 0.0
            
            passed = overall_score >= config.OVERALL_SECURITY_THRESHOLD
            
            results['passed'] = passed
            results['overall_score'] = overall_score
            results['reason'] = f"Combined score: {overall_score:.1%}"
            
            return passed, overall_score, results
    
    def reset(self):
        """Reset all tiers"""
        if self.motion_detector:
            self.motion_detector.reset()
        if self.challenge_system:
            self.challenge_system.reset()
        
        self.tier1_passed = False
        self.tier2_passed = False
        self.tier3_passed = False
        
        self.tier1_score = 0.0
        self.tier2_score = 0.0
        self.tier3_score = 0.0
    
    def get_status_message(self, results):
        """Get user-facing status message"""
        
        # Tier 2 waiting
        if results.get('tier2') and results['tier2'].get('status') == 'waiting':
            frames = results['tier2']['frames_collected']
            needed = results['tier2']['frames_needed']
            return f"Analyzing motion... {frames}/{needed}"
        
        # Tier 3 challenge
        if results.get('tier3'):
            if results['tier3'].get('status') == 'challenge_started':
                return results['tier3']['instruction']
            elif results['tier3'].get('status') in ['monitoring', 'checking']:
                return results['tier3'].get('instruction', 'Waiting for response...')
        
        # Analysis complete
        if results['passed']:
            return "Security checks passed"
        else:
            return results.get('reason', 'Security check failed')


# Quick test
if __name__ == "__main__":
    from face_detector import FaceDetector
    
    security = UnifiedSecuritySystem()
    face_detector = FaceDetector()
    
    print("\n" + "="*60)
    print("UNIFIED SECURITY SYSTEM TEST")
    print("="*60)
    print("\nShow your face to camera")
    print("System will run all security checks")
    print("Press Q to quit")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Detect face
        detections = face_detector.detect_faces(frame)
        
        if len(detections) > 0:
            box = detections[0]['box']
            x, y, w, h = box
            
            # Extract face
            face = face_detector.extract_face(frame, box)
            face = cv2.resize(face, (112, 112))
            
            # Run security check
            is_secure, score, results = security.check_security(face, debug=True)
            
            # Get status message
            status_msg = security.get_status_message(results)
            
            # Draw on frame
            if is_secure is None:
                # In progress
                color = (0, 255, 255)  # Yellow
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display, status_msg, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            elif is_secure:
                # Passed
                color = (0, 255, 0)  # Green
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)
                cv2.putText(display, f"SECURE - {score:.1%}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                # Failed
                color = (0, 0, 255)  # Red
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)
                cv2.putText(display, f"DENIED - {status_msg}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(display, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Security Test', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()