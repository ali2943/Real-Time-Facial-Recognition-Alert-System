"""
Motion-Based Liveness Detection
Real faces have micro-movements, photos are static
VERY EFFECTIVE against photo attacks!
"""

import cv2
import numpy as np
from collections import deque


class MotionLivenessDetector:
    """
    Detects liveness through motion analysis
    Real faces: breathing, micro-movements, pulse
    Photos: completely static
    """
    
    def __init__(self, buffer_size=15):
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.embedding_buffer = deque(maxlen=buffer_size)
        
        self.motion_threshold = 0.45  # Threshold for liveness
        
        print("[INFO] Motion Liveness Detector initialized")
        print(f"  Buffer size: {buffer_size} frames")
        print(f"  Motion threshold: {self.motion_threshold:.1%}")
    
    def add_frame(self, face_img, embedding=None):
        """
        Add frame to buffer for motion analysis
        
        Args:
            face_img: Face image (BGR, 112x112)
            embedding: Optional face embedding
        """
        
        face = cv2.resize(face_img, (112, 112))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        self.frame_buffer.append(gray)
        
        if embedding is not None:
            self.embedding_buffer.append(embedding)
    
    def check_liveness(self):
        """
        Check liveness based on accumulated frames
        
        Returns:
            (is_live, confidence, details)
        """
        
        # Need minimum frames
        if len(self.frame_buffer) < 10:
            return None, 0.0, {
                'status': 'waiting',
                'frames_collected': len(self.frame_buffer),
                'frames_needed': 10
            }
        
        scores = {}
        
        # Check 1: Optical flow motion
        flow_score = self._check_optical_flow()
        scores['optical_flow'] = flow_score
        
        # Check 2: Frame difference motion
        diff_score = self._check_frame_differences()
        scores['frame_diff'] = diff_score
        
        # Check 3: Embedding stability (if available)
        if len(self.embedding_buffer) >= 10:
            embedding_score = self._check_embedding_stability()
            scores['embedding_stability'] = embedding_score
            
            # Combine with embeddings
            overall_score = (
                flow_score * 0.35 +
                diff_score * 0.30 +
                embedding_score * 0.35
            )
        else:
            # Without embeddings
            overall_score = (
                flow_score * 0.55 +
                diff_score * 0.45
            )
        
        is_live = overall_score > self.motion_threshold
        
        details = {
            'status': 'analyzed',
            'scores': scores,
            'overall': overall_score,
            'threshold': self.motion_threshold,
            'frames_used': len(self.frame_buffer)
        }
        
        return is_live, overall_score, details
    
    def _check_optical_flow(self):
        """
        Calculate optical flow between frames
        Real faces have consistent micro-movements
        Photos are static (no flow)
        """
        
        flows = []
        
        for i in range(len(self.frame_buffer) - 1):
            prev = self.frame_buffer[i]
            curr = self.frame_buffer[i + 1]
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_magnitude = np.mean(magnitude)
            
            flows.append(avg_magnitude)
        
        # Analyze motion patterns
        avg_flow = np.mean(flows)
        std_flow = np.std(flows)
        
        # Real faces: avg_flow 0.2-1.5, std 0.05-0.3
        # Photos: avg_flow <0.1, std <0.03
        
        # Score based on average flow
        if avg_flow > 1.5:
            flow_avg_score = 1.0
        elif avg_flow > 0.2:
            flow_avg_score = (avg_flow - 0.2) / 1.3
        else:
            flow_avg_score = 0.0
        
        # Score based on variance (real faces have varying motion)
        if std_flow > 0.3:
            flow_std_score = 1.0
        elif std_flow > 0.05:
            flow_std_score = (std_flow - 0.05) / 0.25
        else:
            flow_std_score = 0.0
        
        # Combine
        flow_score = (flow_avg_score * 0.6 + flow_std_score * 0.4)
        
        return flow_score
    
    def _check_frame_differences(self):
        """
        Analyze frame-to-frame differences
        Real faces change between frames
        Photos are identical across frames
        """
        
        differences = []
        
        for i in range(len(self.frame_buffer) - 1):
            prev = self.frame_buffer[i]
            curr = self.frame_buffer[i + 1]
            
            # Absolute difference
            diff = cv2.absdiff(curr, prev)
            
            # Mean difference
            mean_diff = np.mean(diff)
            
            differences.append(mean_diff)
        
        avg_diff = np.mean(differences)
        
        # Real faces: avg_diff 2-10
        # Photos: avg_diff <1
        
        if avg_diff > 10:
            score = 1.0
        elif avg_diff > 2:
            score = (avg_diff - 2) / 8
        else:
            score = 0.0
        
        return score
    
    def _check_embedding_stability(self):
        """
        Analyze embedding consistency
        Real faces: stable embeddings (same person)
        Photos: TOO stable (completely identical)
        """
        
        embeddings = list(self.embedding_buffer)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(embeddings) - 1):
            dist = np.linalg.norm(embeddings[i] - embeddings[i+1])
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Real faces: small but non-zero distance (0.01-0.05)
        # Photos: near-zero distance (<0.005)
        
        if avg_distance < 0.005:
            # Too stable = photo
            score = 0.0
        elif avg_distance < 0.01:
            # Borderline
            score = 0.3
        elif avg_distance < 0.08:
            # Good - real face
            score = 1.0
        else:
            # Too much variation = different person or bad detection
            score = 0.5
        
        return score
    
    def reset(self):
        """Reset buffers"""
        self.frame_buffer.clear()
        self.embedding_buffer.clear()
    
    def get_status_message(self):
        """Get user-facing status message"""
        
        frames_collected = len(self.frame_buffer)
        
        if frames_collected < 10:
            return f"Analyzing motion... {frames_collected}/10 frames"
        else:
            return "Motion analysis complete"


# Test
if __name__ == "__main__":
    from face_detector import FaceDetector
    
    detector = MotionLivenessDetector()
    face_detector = FaceDetector()
    
    print("\n" + "="*60)
    print("MOTION LIVENESS TEST")
    print("="*60)
    print("\n1. Test REAL FACE - Press 1")
    print("2. Test PHOTO - Press 2")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(0)
    
    results = []
    current_test = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Instructions
        if current_test is None:
            if len(results) == 0:
                cv2.putText(display, "REAL FACE - Press 1 to start", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif len(results) == 1:
                cv2.putText(display, "PHOTO - Press 2 to start", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Tests complete - Q to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Show status
            status = detector.get_status_message()
            cv2.putText(display, status, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Collect frames
            detections = face_detector.detect_faces(frame)
            if len(detections) > 0:
                face = face_detector.extract_face(frame, detections[0]['box'])
                detector.add_frame(face)
                
                # Check if ready
                is_live, conf, details = detector.check_liveness()
                
                if details['status'] == 'analyzed':
                    # Test complete
                    results.append({
                        'type': current_test,
                        'is_live': is_live,
                        'confidence': conf,
                        'details': details
                    })
                    
                    print(f"\n[{current_test}]")
                    print(f"  Motion detected: {'YES ✓' if is_live else 'NO ✗'}")
                    print(f"  Confidence: {conf:.1%}")
                    print(f"  Scores: {details['scores']}")
                    
                    current_test = None
                    detector.reset()
        
        cv2.imshow('Motion Test', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1') and current_test is None and len(results) == 0:
            current_test = "REAL FACE"
            detector.reset()
            print("\n[TEST] REAL FACE - Collecting frames...")
        
        elif key == ord('2') and current_test is None and len(results) == 1:
            current_test = "PHOTO"
            detector.reset()
            print("\n[TEST] PHOTO - Collecting frames...")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Results
    if len(results) == 2:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        real = results[0]
        photo = results[1]
        
        print(f"\nREAL FACE: {real['confidence']:.1%} - {'LIVE ✓' if real['is_live'] else 'STATIC ✗'}")
        print(f"PHOTO:     {photo['confidence']:.1%} - {'LIVE ✗ PROBLEM!' if photo['is_live'] else 'STATIC ✓'}")
        
        diff = real['confidence'] - photo['confidence']
        print(f"\nSeparation: {diff:.1%}")
        
        if not photo['is_live']:
            print("\n✅ Motion detection working correctly!")
        else:
            print("\n❌ Photo passed motion test - needs tuning")