"""
Challenge-Response Liveness Detection
Ask user to perform action (blink, smile, turn head)
Photos cannot respond!
"""

import cv2
import numpy as np
import random
import time


class ChallengeResponseLiveness:
    """
    Interactive liveness verification
    User must respond to random challenges
    """
    
    def __init__(self):
        self.challenges = [
            'blink',
            'smile',
            'turn_left',
            'turn_right',
            'nod'
        ]
        
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_timeout = 5.0  # seconds
        
        # Detection state
        self.baseline_face = None
        self.baseline_landmarks = None
        
        print("[INFO] Challenge-Response Liveness initialized")
    
    def start_challenge(self):
        """Start a new random challenge"""
        
        self.current_challenge = random.choice(self.challenges)
        self.challenge_start_time = time.time()
        
        return self.current_challenge
    
    def get_challenge_instruction(self):
        """Get user-facing instruction"""
        
        instructions = {
            'blink': 'Please BLINK your eyes',
            'smile': 'Please SMILE',
            'turn_left': 'Turn your head LEFT',
            'turn_right': 'Turn your head RIGHT',
            'nod': 'NOD your head'
        }
        
        return instructions.get(self.current_challenge, '')
    
    def check_response(self, face_img, landmarks=None):
        """
        Check if user responded to challenge
        
        Returns:
            (responded, confidence, details)
        """
        
        if self.current_challenge is None:
            return False, 0.0, {'error': 'No active challenge'}
        
        # Check timeout
        elapsed = time.time() - self.challenge_start_time
        if elapsed > self.challenge_timeout:
            return False, 0.0, {'error': 'Timeout'}
        
        # Set baseline on first check
        if self.baseline_face is None:
            self.baseline_face = face_img
            self.baseline_landmarks = landmarks
            return None, 0.0, {'status': 'baseline_set'}
        
        # Check specific challenge
        if self.current_challenge == 'blink':
            responded, conf = self._check_blink(face_img, landmarks)
        elif self.current_challenge == 'smile':
            responded, conf = self._check_smile(face_img, landmarks)
        elif self.current_challenge == 'turn_left':
            responded, conf = self._check_turn_left(face_img)
        elif self.current_challenge == 'turn_right':
            responded, conf = self._check_turn_right(face_img)
        elif self.current_challenge == 'nod':
            responded, conf = self._check_nod(face_img)
        else:
            responded, conf = False, 0.0
        
        details = {
            'challenge': self.current_challenge,
            'elapsed': elapsed,
            'timeout': self.challenge_timeout
        }
        
        return responded, conf, details
    
    def _check_blink(self, face_img, landmarks):
        """
        Detect blink
        Simple: check if eyes get darker (closed)
        """
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(self.baseline_face, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        
        # Eye regions (approximate)
        left_eye = gray[int(h*0.35):int(h*0.45), int(w*0.2):int(w*0.4)]
        baseline_left_eye = baseline_gray[int(h*0.35):int(h*0.45), int(w*0.2):int(w*0.4)]
        
        # Check if eyes got darker (closed)
        current_brightness = np.mean(left_eye)
        baseline_brightness = np.mean(baseline_left_eye)
        
        brightness_diff = baseline_brightness - current_brightness
        
        # Blink: brightness drops >15
        if brightness_diff > 15:
            return True, 1.0
        else:
            return False, brightness_diff / 15
    
    def _check_smile(self, face_img, landmarks):
        """Detect smile - mouth widens"""
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(self.baseline_face, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        
        # Mouth region
        mouth = gray[int(h*0.6):int(h*0.8), int(w*0.3):int(w*0.7)]
        baseline_mouth = baseline_gray[int(h*0.6):int(h*0.8), int(w*0.3):int(w*0.7)]
        
        # Edge detection (smile creates more edges)
        edges = cv2.Canny(mouth, 50, 150)
        baseline_edges = cv2.Canny(baseline_mouth, 50, 150)
        
        edge_increase = np.sum(edges) - np.sum(baseline_edges)
        
        # Smile: edge increase >500
        if edge_increase > 500:
            return True, 1.0
        else:
            return False, edge_increase / 500
    
    def _check_turn_left(self, face_img):
        """Detect left turn - face shifts right in frame"""
        
        # Simple: compare center of mass
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(self.baseline_face, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        _, baseline_thresh = cv2.threshold(baseline_gray, 100, 255, cv2.THRESH_BINARY)
        
        # Moments
        M = cv2.moments(thresh)
        baseline_M = cv2.moments(baseline_thresh)
        
        if M['m00'] > 0 and baseline_M['m00'] > 0:
            cx = M['m10'] / M['m00']
            baseline_cx = baseline_M['m10'] / baseline_M['m00']
            
            shift = cx - baseline_cx
            
            # Left turn: shift right (+ve)
            if shift > 5:
                return True, 1.0
            else:
                return False, shift / 5
        
        return False, 0.0
    
    def _check_turn_right(self, face_img):
        """Detect right turn"""
        
        responded, conf = self._check_turn_left(face_img)
        
        # Invert (right turn = negative shift)
        if responded:
            return False, 0.0
        else:
            # Check negative shift
            return True, 1.0 if conf < 0 else 0.0
    
    def _check_nod(self, face_img):
        """Detect nod - vertical movement"""
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(self.baseline_face, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        _, baseline_thresh = cv2.threshold(baseline_gray, 100, 255, cv2.THRESH_BINARY)
        
        M = cv2.moments(thresh)
        baseline_M = cv2.moments(baseline_thresh)
        
        if M['m00'] > 0 and baseline_M['m00'] > 0:
            cy = M['m01'] / M['m00']
            baseline_cy = baseline_M['m01'] / baseline_M['m00']
            
            vertical_shift = abs(cy - baseline_cy)
            
            if vertical_shift > 5:
                return True, 1.0
            else:
                return False, vertical_shift / 5
        
        return False, 0.0
    
    def reset(self):
        """Reset challenge"""
        self.current_challenge = None
        self.baseline_face = None
        self.baseline_landmarks = None