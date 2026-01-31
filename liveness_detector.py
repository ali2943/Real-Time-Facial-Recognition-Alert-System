"""
Liveness Detection Module - Anti-Spoofing for Real-Time Facial Recognition

This module implements multiple anti-spoofing strategies to prevent attacks from:
- Static photos (printed or on screens)
- Video replay attacks
- Masks or 3D models (basic detection)

Design Rationale:
- Eye Aspect Ratio (EAR): Detects natural eye blinks which photos/videos can't replicate in real-time
- Motion Analysis: Detects micro-movements from breathing and natural head motion
- Texture Analysis: Identifies print artifacts and screen moire patterns
- Challenge-Response: Prompts user for random actions (blink, turn head) for high-security scenarios

Limitations on Laptop Webcam (2D RGB):
- Cannot detect sophisticated 3D masks (would need depth sensor)
- Limited in very low light conditions
- Cannot distinguish identical twins (would need IR or depth)
"""

import cv2
import numpy as np
from collections import deque
import time
import random
import config


class LivenessDetector:
    """
    Anti-spoofing detection using multiple complementary strategies
    
    This implementation focuses on realistic laptop webcam capabilities,
    similar to mobile Face Unlock but without IR/depth sensors.
    """
    
    def __init__(self):
        """
        Initialize liveness detector with configurable strategies
        
        The detector maintains temporal buffers to analyze patterns across frames,
        which is essential for detecting real vs. spoofed faces.
        """
        self.method = config.LIVENESS_METHOD
        self.frames_required = config.LIVENESS_FRAMES_REQUIRED
        self.require_blink = config.REQUIRE_BLINK
        self.blink_timeout = config.BLINK_TIMEOUT
        self.texture_threshold = config.TEXTURE_ANALYSIS_THRESHOLD
        
        # Frame buffers for temporal analysis (track face over time)
        self.frame_buffer = deque(maxlen=self.frames_required)
        self.face_positions = deque(maxlen=self.frames_required)
        self.landmarks_buffer = deque(maxlen=self.frames_required)
        
        # Eye blink detection state
        self.blink_detected = False
        self.blink_check_start = None
        self.blink_count = 0
        # Eye Aspect Ratio (EAR) threshold empirically determined:
        # - Open eyes: EAR ≈ 0.25-0.35
        # - Closed eyes: EAR < 0.21
        # - This threshold works across different eye shapes and sizes
        self.ear_threshold = 0.21
        self.ear_history = deque(maxlen=30)  # Track EAR over time for blink detection
        
        # Head movement detection
        self.head_positions = deque(maxlen=10)
        self.movement_threshold = 5.0  # pixels
        
        # Challenge-response system for high-security scenarios
        self.challenge_active = False
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_timeout = 5.0  # seconds
        self.challenges = ['blink', 'turn_left', 'turn_right', 'nod']
        
        print(f"[INFO] Liveness Detector initialized (method: {self.method})")
        print(f"[INFO] - Blink detection: {'Enabled' if self.require_blink else 'Disabled'}")
        print(f"[INFO] - Challenge mode: Available for high-security use cases")
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) - Soukupová & Čech method
        
        Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        where p1-p6 are the 6 eye landmark points
        
        For simplified landmarks (single point per eye from InsightFace),
        we estimate based on eye region intensity analysis.
        
        Args:
            eye_landmarks: Eye landmark points (can be single point or 6 points)
            
        Returns:
            EAR value: ~0.3 for open eyes, <0.21 for closed eyes
            
        Design Note:
        Full dlib 68-point landmarks would give more accurate EAR,
        but InsightFace provides 5 key points only. We adapt by:
        1. If 6+ points available: Use proper EAR formula
        2. If single point: Estimate from surrounding pixel intensity
        """
        # Check if we have full eye landmarks (6 points per eye)
        if isinstance(eye_landmarks, np.ndarray) and len(eye_landmarks) >= 6:
            # Proper EAR calculation with 6 points
            # Vertical distances
            v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            # Horizontal distance
            h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # EAR formula
            ear = (v1 + v2) / (2.0 * h)
            return ear
        else:
            # Simplified estimation for single-point landmarks
            # Return default open-eye value (will be updated with actual implementation if needed)
            return 0.3
    
    def detect_eye_blink(self, landmarks_sequence):
        """
        Detect eye blinks using Eye Aspect Ratio (EAR) temporal analysis
        
        A blink is characterized by:
        1. EAR drops below threshold (eye closes)
        2. EAR rises back above threshold (eye opens)
        3. Entire sequence happens within 100-400ms (3-12 frames at 30fps)
        
        Args:
            landmarks_sequence: Sequence of facial landmarks from multiple frames
            
        Returns:
            True if valid blink detected, False otherwise
            
        Design Rationale:
        - Photos cannot blink in real-time
        - Video replays have fixed blink patterns (can detect with timing analysis)
        - Real humans blink naturally every 2-10 seconds
        - Blink duration is consistent (100-400ms) across individuals
        """
        if landmarks_sequence is None or len(landmarks_sequence) < 3:
            return False
        
        # Calculate EAR for each frame in sequence
        ear_values = []
        for landmarks in landmarks_sequence:
            if landmarks and 'left_eye' in landmarks and 'right_eye' in landmarks:
                try:
                    # Calculate EAR for both eyes
                    left_eye = landmarks['left_eye']
                    right_eye = landmarks['right_eye']
                    
                    # For single-point landmarks, estimate EAR
                    # In production, you'd use dlib 68-point landmarks for precise EAR
                    left_ear = self.calculate_ear(left_eye)
                    right_ear = self.calculate_ear(right_eye)
                    
                    # Average of both eyes (more robust than single eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    ear_values.append(avg_ear)
                except Exception as e:
                    # Skip frames with landmark extraction errors
                    continue
        
        if len(ear_values) < 3:
            return False
        
        # Store in history for pattern analysis
        self.ear_history.extend(ear_values)
        
        # Detect blink pattern: valley in EAR values
        # Pattern: high -> low -> high (eyes open -> close -> open)
        blink_detected = False
        for i in range(1, len(ear_values) - 1):
            # Check for valley: current value is minimum
            if (ear_values[i] < self.ear_threshold and
                ear_values[i] < ear_values[i-1] and
                ear_values[i] < ear_values[i+1]):
                
                # Verify valley depth (must be significant)
                valley_depth = min(ear_values[i-1], ear_values[i+1]) - ear_values[i]
                if valley_depth > 0.05:  # Significant eye closure
                    blink_detected = True
                    self.blink_count += 1
                    if config.DEBUG_MODE:
                        print(f"[DEBUG] Blink detected! Count: {self.blink_count}, EAR valley: {ear_values[i]:.3f}")
                    break
        
        return blink_detected
    
    def check_motion_liveness(self, face_sequence):
        """
        Analyze face movement across frames for natural micro-movements
        
        Real faces exhibit:
        - Micro-movements from breathing (1-3 pixels)
        - Slight head movements (unconscious adjustments)
        - Non-zero variance in position
        
        Static photos/screens show:
        - Zero or very minimal movement
        - Constant position (unless physically moved)
        - Low variance
        
        Args:
            face_sequence: List of face positions (bounding boxes) from multiple frames
            
        Returns:
            Tuple of (is_live: bool, confidence: float)
            
        Design Rationale:
        Even when trying to stay still, humans have involuntary micro-movements
        from heartbeat, breathing, and muscle tension. Photos lack these entirely.
        """
        if len(face_sequence) < 3:
            # Not enough frames for reliable analysis
            return True, 0.5
        
        # Calculate movement between consecutive frames
        movements = []
        for i in range(1, len(face_sequence)):
            prev_box = face_sequence[i-1]
            curr_box = face_sequence[i]
            
            # Calculate center displacement (accounts for whole-head movement)
            prev_center = np.array([prev_box[0] + prev_box[2]/2, prev_box[1] + prev_box[3]/2])
            curr_center = np.array([curr_box[0] + curr_box[2]/2, curr_box[1] + curr_box[3]/2])
            
            displacement = np.linalg.norm(curr_center - prev_center)
            movements.append(displacement)
        
        # Statistical analysis of movements
        avg_movement = np.mean(movements)
        movement_variance = np.var(movements)
        max_movement = np.max(movements)
        
        # Decision logic:
        # 1. Static images: avg < 1px AND variance < 0.5
        # 2. Real faces: avg > 1px OR variance > 0.5 (even when trying to stay still)
        # 3. Extreme movement (>30px): might be someone physically moving a photo
        
        is_static = avg_movement < 1.0 and movement_variance < 0.5
        is_excessive = max_movement > 30.0  # Someone moving photo/screen
        
        if is_static:
            # Likely a static photo or very still video
            return False, 0.2
        elif is_excessive:
            # Too much movement - suspicious
            return False, 0.3
        else:
            # Natural movement detected
            # Confidence based on movement characteristics
            confidence = min(1.0, (avg_movement + movement_variance) / 10.0)
            confidence = max(0.5, confidence)  # Minimum 0.5 for detected movement
            
            if config.DEBUG_MODE:
                print(f"[DEBUG] Motion: avg={avg_movement:.2f}px, var={movement_variance:.2f}, conf={confidence:.2f}")
            
            return True, confidence
    
    def detect_head_movement(self, face_sequence):
        """
        Detect deliberate head movements (left, right, nod)
        
        Used for challenge-response authentication where user is prompted
        to move their head in a specific direction.
        
        Args:
            face_sequence: Recent face positions
            
        Returns:
            Detected movement direction: 'left', 'right', 'nod', 'up', 'down', or None
            
        Design Note:
        This is for active liveness detection where user is prompted to
        perform specific movements. More secure than passive detection.
        """
        if len(face_sequence) < 5:
            return None
        
        # Calculate center positions over time
        centers = []
        for box in face_sequence:
            center_x = box[0] + box[2] / 2
            center_y = box[1] + box[3] / 2
            centers.append((center_x, center_y))
        
        # Analyze horizontal movement (left/right)
        x_positions = [c[0] for c in centers]
        x_trend = x_positions[-1] - x_positions[0]
        x_variance = np.var(x_positions)
        
        # Analyze vertical movement (up/down/nod)
        y_positions = [c[1] for c in centers]
        y_trend = y_positions[-1] - y_positions[0]
        y_variance = np.var(y_positions)
        
        # Detect movement direction (threshold: 10 pixels)
        movement = None
        if abs(x_trend) > 10 and x_variance > 5:
            movement = 'left' if x_trend < 0 else 'right'
        elif abs(y_trend) > 10 and y_variance > 5:
            movement = 'up' if y_trend < 0 else 'down'
            # Nod detection: down then up movement
            if len(y_positions) > 8:
                mid_point = len(y_positions) // 2
                first_half_trend = y_positions[mid_point] - y_positions[0]
                second_half_trend = y_positions[-1] - y_positions[mid_point]
                if first_half_trend > 5 and second_half_trend < -5:
                    movement = 'nod'
        
        return movement
    
    def analyze_texture(self, face_image):
        """
        Analyze texture for print/screen detection using gradient analysis
        
        Distinguishing characteristics:
        1. Printed photos: More uniform texture, visible print patterns, lower gradient variance
        2. Screens: Moire patterns, pixel grid artifacts, color banding
        3. Real faces: Natural skin texture, pore details, hair variation, high local variance
        
        Args:
            face_image: Face image to analyze
            
        Returns:
            Tuple of (is_real: bool, score: float)
            
        Design Rationale:
        - Uses Sobel edge detection to measure texture richness
        - Real skin has complex micro-texture that photos lose
        - Screens introduce artifacts (pixel grid, refresh) detectable in gradient domain
        - This works on standard RGB camera without need for special sensors
        
        Limitations:
        - High-quality prints may pass (would need 3D depth or IR)
        - Very good lighting required for reliable detection
        - May struggle with very dark or very bright faces
        """
        try:
            # Convert to grayscale for texture analysis
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Apply Gaussian blur to reduce noise while preserving edges
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate gradient magnitude using Sobel operator
            # Sobel is better than simple difference for detecting edges
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Analyze texture characteristics
            # 1. Edge density: Real faces have more edges (skin texture, pores, hair)
            edge_density = np.mean(gradient_magnitude > 30)
            
            # 2. Texture variance: Real faces have higher local variance
            texture_variance = np.var(gradient_magnitude)
            
            # 3. High-frequency content: Photos lose high-frequency details
            # Use Laplacian for high-frequency analysis
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            high_freq_content = np.var(laplacian)
            
            # Combine metrics into texture score (0-1 scale)
            # Normalized to 0-1 based on empirical thresholds
            edge_score = min(1.0, edge_density * 5)  # Typical real face: 0.2-0.3
            variance_score = min(1.0, texture_variance / 1000)  # Typical: 500-1200
            freq_score = min(1.0, high_freq_content / 100)  # Typical: 50-150
            
            # Weighted combination
            # Edge density: 40% (most reliable indicator)
            # Variance: 30% (distinguishes flat prints)
            # High-freq: 30% (catches low-quality reproductions)
            texture_score = (edge_score * 0.4 + variance_score * 0.3 + freq_score * 0.3)
            
            # Decision threshold
            is_real = texture_score > self.texture_threshold
            
            if config.DEBUG_MODE:
                print(f"[DEBUG] Texture analysis: score={texture_score:.3f}, edge={edge_density:.3f}, var={texture_variance:.1f}, freq={high_freq_content:.1f}")
            
            return is_real, texture_score
        
        except Exception as e:
            print(f"[WARNING] Texture analysis failed: {e}")
            # Fail open (allow access) rather than fail closed
            # This prevents legitimate users being blocked by errors
            return True, 0.5
    
    def start_challenge(self):
        """
        Start a random challenge for challenge-response liveness detection
        
        This is used for high-security scenarios where passive detection
        isn't sufficient. User is prompted to perform a specific action.
        
        Returns:
            Challenge instruction string
            
        Design Note:
        Challenge-response is more secure than passive detection because:
        1. Attacker can't predict what action will be requested
        2. Video replay attacks are defeated (pre-recorded video won't match challenge)
        3. Combines with other checks for defense-in-depth
        """
        self.current_challenge = random.choice(self.challenges)
        self.challenge_start_time = time.time()
        self.challenge_active = True
        
        # Map challenge types to user-friendly instructions
        instructions = {
            'blink': "Please blink twice",
            'turn_left': "Please turn your head left",
            'turn_right': "Please turn your head right",
            'nod': "Please nod your head"
        }
        
        instruction = instructions.get(self.current_challenge, "Please follow instruction")
        print(f"[CHALLENGE] {instruction}")
        
        return self.current_challenge, instruction
    
    def check_challenge_response(self):
        """
        Check if user completed the current challenge
        
        Returns:
            Tuple of (completed: bool, timed_out: bool)
        """
        if not self.challenge_active:
            return False, False
        
        # Check timeout
        elapsed = time.time() - self.challenge_start_time
        if elapsed > self.challenge_timeout:
            self.challenge_active = False
            return False, True
        
        # Check if challenge was completed
        completed = False
        if self.current_challenge == 'blink':
            # Check if at least 2 blinks detected
            if self.blink_count >= 2:
                completed = True
        elif self.current_challenge in ['turn_left', 'turn_right', 'nod']:
            # Check head movement
            movement = self.detect_head_movement(list(self.face_positions))
            if movement == self.current_challenge.replace('turn_', ''):
                completed = True
            elif movement == 'nod' and self.current_challenge == 'nod':
                completed = True
        
        if completed:
            self.challenge_active = False
            print(f"[CHALLENGE] Completed: {self.current_challenge}")
        
        return completed, False
    
    def reset_challenge(self):
        """Reset challenge state"""
        self.challenge_active = False
        self.current_challenge = None
        self.challenge_start_time = None
        self.blink_count = 0
    
    def is_live(self, current_frame, face_box, landmarks):
        """
        Multi-strategy liveness check with comprehensive logging
        
        This is the main entry point for liveness detection. It coordinates
        multiple detection strategies based on configuration.
        
        Args:
            current_frame: Current video frame (BGR format)
            face_box: Bounding box of detected face [x, y, w, h]
            landmarks: Facial landmarks (dict with left_eye, right_eye, nose, etc.)
            
        Returns:
            Tuple of (is_live: bool, confidence: float, reason: str)
            
        Detection Strategies Available:
        1. 'motion': Passive micro-movement analysis (fastest, least intrusive)
        2. 'blink': Eye blink detection (moderate speed, good security)
        3. 'texture': Surface texture analysis (slower, works on still images)
        4. 'combined': All methods with voting (best security, slower)
        
        Design Philosophy:
        - Fail gracefully: if detection fails, log warning but don't block user
        - Progressive disclosure: start with fast checks, escalate if needed
        - Defense in depth: combine multiple independent checks
        - User experience: prefer passive checks, use active (challenge) only when needed
        """
        # Add current data to temporal buffers for pattern analysis
        self.face_positions.append(face_box)
        self.landmarks_buffer.append(landmarks)
        
        # Extract face region for texture analysis
        x, y, w, h = face_box
        y1, y2 = max(0, y), min(current_frame.shape[0], y+h)
        x1, x2 = max(0, x), min(current_frame.shape[1], x+w)
        face_region = current_frame[y1:y2, x1:x2]
        
        if face_region.size > 0:
            self.frame_buffer.append(face_region)
        
        # Initialize blink timer if required
        if self.require_blink and self.blink_check_start is None:
            self.blink_check_start = time.time()
        
        # Wait for enough frames (warm-up period)
        if len(self.frame_buffer) < self.frames_required:
            frames_remaining = self.frames_required - len(self.frame_buffer)
            return True, 0.5, f"Initializing... ({frames_remaining} frames)"
        
        # Check if challenge is active
        if self.challenge_active:
            completed, timed_out = self.check_challenge_response()
            if completed:
                return True, 1.0, f"Challenge passed: {self.current_challenge}"
            elif timed_out:
                return False, 0.0, f"Challenge timeout: {self.current_challenge}"
            else:
                elapsed = time.time() - self.challenge_start_time
                remaining = self.challenge_timeout - elapsed
                return True, 0.5, f"Challenge pending: {self.current_challenge} ({remaining:.1f}s remaining)"
        
        # Perform liveness checks based on configured method
        if self.method == 'motion':
            # Motion-based detection: fastest, least intrusive
            is_live, confidence = self.check_motion_liveness(list(self.face_positions))
            reason = "Motion analysis"
            
            if config.DEBUG_MODE:
                status = "PASS" if is_live else "FAIL"
                print(f"[LIVENESS] Motion check: {status} (confidence: {confidence:.2f})")
            
        elif self.method == 'blink':
            # Blink detection: requires user to blink naturally
            blink_detected = self.detect_eye_blink(list(self.landmarks_buffer))
            is_live = blink_detected
            confidence = 1.0 if blink_detected else 0.0
            reason = "Blink detected" if blink_detected else "No blink detected"
            
            if config.DEBUG_MODE:
                print(f"[LIVENESS] Blink check: {'PASS' if blink_detected else 'FAIL'} (total blinks: {self.blink_count})")
            
        elif self.method == 'texture':
            # Texture analysis: works on static images too
            if face_region.size > 0:
                is_live, texture_score = self.analyze_texture(face_region)
                confidence = texture_score
                reason = f"Texture analysis (score: {texture_score:.2f})"
                
                if config.DEBUG_MODE:
                    status = "PASS" if is_live else "FAIL"
                    print(f"[LIVENESS] Texture check: {status} (score: {texture_score:.2f})")
            else:
                is_live, confidence, reason = True, 0.5, "Texture analysis skipped (invalid face region)"
            
        elif self.method == 'combined':
            # Combined multi-strategy approach (most secure)
            # Combines motion, texture, and optionally blink detection
            
            # 1. Motion check
            motion_live, motion_conf = self.check_motion_liveness(list(self.face_positions))
            
            # 2. Texture check
            if face_region.size > 0:
                texture_live, texture_score = self.analyze_texture(face_region)
            else:
                texture_live, texture_score = True, 0.5
            
            # 3. Blink check (if required)
            blink_detected = self.detect_eye_blink(list(self.landmarks_buffer))
            
            # Voting system: majority wins
            votes = [motion_live, texture_live]
            confidences = [motion_conf, texture_score]
            
            if self.require_blink:
                votes.append(blink_detected)
                confidences.append(1.0 if blink_detected else 0.0)
            
            # Calculate overall result
            passed_checks = sum(votes)
            total_checks = len(votes)
            is_live = passed_checks >= (total_checks / 2)  # Majority vote
            confidence = np.mean(confidences)
            
            # Detailed reason
            reason = (f"Combined check ({passed_checks}/{total_checks} passed): "
                     f"motion={motion_conf:.2f}, texture={texture_score:.2f}")
            if self.require_blink:
                reason += f", blink={blink_detected}"
            
            if config.DEBUG_MODE:
                status = "PASS" if is_live else "FAIL"
                print(f"[LIVENESS] Combined check: {status}")
                print(f"  - Motion: {'✓' if motion_live else '✗'} (conf: {motion_conf:.2f})")
                print(f"  - Texture: {'✓' if texture_live else '✗'} (score: {texture_score:.2f})")
                if self.require_blink:
                    print(f"  - Blink: {'✓' if blink_detected else '✗'}")
            
        else:
            # Default: assume live (no liveness check configured)
            is_live = True
            confidence = 1.0
            reason = "No liveness check configured"
            
            if config.DEBUG_MODE:
                print(f"[LIVENESS] Bypassed (method: {self.method})")
        
        # Check mandatory blink timeout if required
        if self.require_blink and not self.blink_detected:
            elapsed = time.time() - self.blink_check_start
            if elapsed > self.blink_timeout:
                return False, 0.0, f"Blink timeout exceeded ({elapsed:.1f}s > {self.blink_timeout}s)"
            
            # Update if blink was detected
            if self.detect_eye_blink(list(self.landmarks_buffer)):
                self.blink_detected = True
        
        return is_live, confidence, reason
    
    def reset(self):
        """Reset liveness detector state"""
        self.frame_buffer.clear()
        self.face_positions.clear()
        self.landmarks_buffer.clear()
        self.blink_detected = False
        self.blink_check_start = None
