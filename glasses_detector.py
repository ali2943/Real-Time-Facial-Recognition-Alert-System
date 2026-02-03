"""
Glasses/Spectacles Detection - IMPROVED VERSION
Better algorithms for accurate detection
"""

import cv2
import numpy as np


class GlassesDetector:
    """
    Improved glasses detection using multiple methods
    """
    
    def __init__(self):
        self.enabled = True
        
        # Detection thresholds (TUNED)
        self.confidence_threshold = 0.55
        
        print("[INFO] Glasses Detector initialized (Improved)")
    
    def detect_glasses(self, face_img, landmarks=None):
        """
        Detect glasses using improved methods
        """
        
        if face_img is None or face_img.size == 0:
            return False, 0.0, {'error': 'Invalid image'}
        
        # Resize to standard size
        face = cv2.resize(face_img, (112, 112))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        scores = {}
        
        # Method 1: Histogram analysis in eye region (IMPROVED)
        histogram_score = self._analyze_eye_histogram(gray)
        scores['histogram'] = histogram_score
        
        # Method 2: Variance analysis (IMPROVED)
        variance_score = self._analyze_variance_pattern(gray)
        scores['variance'] = variance_score
        
        # Method 3: Edge orientation (NEW - VERY RELIABLE)
        edge_score = self._analyze_edge_orientation(gray)
        scores['edge_orientation'] = edge_score
        
        # Method 4: Specular reflection (IMPROVED)
        reflection_score = self._detect_specular_reflection(face)
        scores['reflection'] = reflection_score
        
        # Method 5: Frequency analysis (NEW)
        frequency_score = self._analyze_frequency(gray)
        scores['frequency'] = frequency_score
        
        # Combine scores with optimized weights
        overall_confidence = (
            histogram_score * 0.20 +
            variance_score * 0.15 +
            edge_score * 0.30 +          # Most reliable
            reflection_score * 0.20 +
            frequency_score * 0.15
        )
        
        has_glasses = overall_confidence > self.confidence_threshold
        
        details = {
            'histogram_score': histogram_score,
            'variance_score': variance_score,
            'edge_orientation_score': edge_score,
            'reflection_score': reflection_score,
            'frequency_score': frequency_score,
            'overall_confidence': overall_confidence,
            'threshold': self.confidence_threshold
        }
        
        return has_glasses, overall_confidence, details
    
    def _analyze_eye_histogram(self, gray):
        """
        Analyze histogram distribution in eye region
        Glasses create distinct brightness patterns
        """
        
        h, w = gray.shape
        
        # Eye region (broader area)
        eye_region = gray[int(h*0.2):int(h*0.6), int(w*0.1):int(w*0.9)]
        
        # Calculate histogram
        hist = cv2.calcHist([eye_region], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Glasses tend to create bimodal distribution (dark frames + bright reflection)
        # Calculate histogram peaks
        peaks = []
        for i in range(5, 251):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.001:
                peaks.append((i, hist[i]))
        
        # Score based on peak characteristics
        if len(peaks) >= 2:
            # Multiple peaks suggest glasses
            peak_separation = abs(peaks[0][0] - peaks[-1][0])
            if peak_separation > 100:  # Wide separation
                score = 0.8
            elif peak_separation > 60:
                score = 0.5
            else:
                score = 0.2
        else:
            score = 0.1
        
        return score
    
    def _analyze_variance_pattern(self, gray):
        """
        Analyze local variance patterns
        Glasses create high variance regions (frame edges)
        """
        
        h, w = gray.shape
        
        # Divide into grid
        grid_h, grid_w = 8, 8
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        variances = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                var = np.var(cell)
                variances.append(var)
        
        variances = np.array(variances)
        
        # Glasses create high variance in specific regions (not uniform)
        # Calculate variance of variances
        var_of_var = np.var(variances)
        
        # Also check max variance
        max_var = np.max(variances)
        
        # Score
        # Glasses: high var_of_var (>2000) and high max_var (>600)
        # No glasses: low var_of_var (<1000) and low max_var (<400)
        
        score_1 = min(1.0, max(0.0, (var_of_var - 500) / 2000))
        score_2 = min(1.0, max(0.0, (max_var - 200) / 500))
        
        score = (score_1 + score_2) / 2
        
        return score
    
    def _analyze_edge_orientation(self, gray):
        """
        Analyze edge orientations
        Glasses have strong horizontal edges (frames)
        This is the MOST RELIABLE method
        """
        
        h, w = gray.shape
        
        # Focus on eye region
        eye_region = gray[int(h*0.25):int(h*0.55), :]
        
        # Sobel derivatives
        sobelx = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and orientation
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = np.arctan2(sobely, sobelx)
        
        # Threshold to keep only strong edges
        strong_edges = magnitude > np.percentile(magnitude, 80)
        
        if np.sum(strong_edges) == 0:
            return 0.0
        
        # Get orientations of strong edges
        strong_orientations = orientation[strong_edges]
        
        # Count horizontal edges (near 0 or ±π)
        # Horizontal: -π/4 to π/4 or 3π/4 to 5π/4
        horizontal_mask = (np.abs(strong_orientations) < np.pi/4) | \
                         (np.abs(strong_orientations) > 3*np.pi/4)
        
        horizontal_ratio = np.sum(horizontal_mask) / len(strong_orientations)
        
        # Glasses typically have 40-70% horizontal edges
        # No glasses: 20-40% (natural face contours)
        
        if horizontal_ratio > 0.55:
            score = 1.0
        elif horizontal_ratio > 0.40:
            score = (horizontal_ratio - 0.40) / 0.15
        else:
            score = 0.0
        
        return score
    
    def _detect_specular_reflection(self, face_bgr):
        """
        Detect specular reflections from glass lenses
        IMPROVED: Uses color information
        """
        
        h, w = face_bgr.shape[:2]
        
        # Eye region
        eye_region = face_bgr[int(h*0.25):int(h*0.55), :]
        
        # Convert to HSV
        hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)
        
        # Specular reflections: high V, low S
        # (bright white spots)
        specular_mask = (v_chan > 220) & (s_chan < 50)
        
        specular_pixels = np.sum(specular_mask)
        total_pixels = eye_region.shape[0] * eye_region.shape[1]
        
        specular_ratio = specular_pixels / total_pixels
        
        # Glasses: 1-5% specular
        # No glasses: <1%
        
        if specular_ratio > 0.05:
            score = 1.0
        elif specular_ratio > 0.01:
            score = (specular_ratio - 0.01) / 0.04
        else:
            score = 0.0
        
        return score
    
    def _analyze_frequency(self, gray):
        """
        Frequency domain analysis
        Glasses add high-frequency components (frame edges)
        """
        
        h, w = gray.shape
        eye_region = gray[int(h*0.25):int(h*0.55), :]
        
        # FFT
        f = np.fft.fft2(eye_region)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Analyze high-frequency energy
        rh, rw = magnitude.shape
        center_y, center_w = rh // 2, rw // 2
        
        # High-frequency region (outer)
        mask_high = np.zeros(magnitude.shape, dtype=np.uint8)
        cv2.circle(mask_high, (center_w, center_y), min(rh, rw) // 2, 1, -1)
        cv2.circle(mask_high, (center_w, center_y), min(rh, rw) // 4, 0, -1)
        
        # Low-frequency region (center)
        mask_low = np.zeros(magnitude.shape, dtype=np.uint8)
        cv2.circle(mask_low, (center_w, center_y), min(rh, rw) // 6, 1, -1)
        
        high_energy = np.sum(magnitude * mask_high)
        low_energy = np.sum(magnitude * mask_low)
        
        if low_energy > 0:
            ratio = high_energy / low_energy
        else:
            ratio = 0
        
        # Glasses have higher ratio (more high-freq content)
        # Glasses: 0.4-0.8
        # No glasses: 0.2-0.4
        
        score = min(1.0, max(0.0, (ratio - 0.2) / 0.5))
        
        return score
    
    def explain_detection(self, details):
        """
        Generate human-readable explanation
        """
        
        lines = []
        lines.append("="*60)
        lines.append("GLASSES DETECTION REPORT (IMPROVED)")
        lines.append("="*60)
        
        has_glasses = details['overall_confidence'] > details['threshold']
        
        lines.append(f"\nResult: {'GLASSES DETECTED 👓' if has_glasses else 'NO GLASSES ✓'}")
        lines.append(f"Confidence: {details['overall_confidence']:.1%}")
        lines.append(f"Threshold: {details['threshold']:.1%}")
        
        lines.append("\nCOMPONENT SCORES:")
        lines.append("-"*60)
        
        components = [
            ('Histogram Analysis', details['histogram_score'], 0.20),
            ('Variance Pattern', details['variance_score'], 0.15),
            ('Edge Orientation', details['edge_orientation_score'], 0.30),
            ('Specular Reflection', details['reflection_score'], 0.20),
            ('Frequency Analysis', details['frequency_score'], 0.15)
        ]
        
        for name, score, weight in components:
            bar_length = int(score * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            contribution = score * weight
            
            status = '✅' if score > 0.4 else ('⚠️' if score > 0.2 else '❌')
            
            lines.append(f"{name:20} {status} {score:.1%} {bar} (×{weight:.0%} = {contribution:.1%})")
        
        lines.append("="*60)
        
        return "\n".join(lines)


# Same test code as before...
if __name__ == "__main__":
    import sys
    
    detector = GlassesDetector()
    
    print("\n" + "="*60)
    print("IMPROVED GLASSES DETECTOR TEST")
    print("="*60)
    print("\nInstructions:")
    print("  1. WITHOUT glasses - Press 1")
    print("  2. WITH glasses - Press 2")
    print("  Q - Quit")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(0)
    
    from face_detector import FaceDetector
    face_detector = FaceDetector()
    
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        if len(results) == 0:
            cv2.putText(display, "WITHOUT glasses - Press 1", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(results) == 1:
            cv2.putText(display, "WITH glasses - Press 2", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Tests complete - Press Q", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Glasses Test', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1') and len(results) == 0:
            detections = face_detector.detect_faces(frame)
            if len(detections) > 0:
                face = face_detector.extract_face(frame, detections[0]['box'])
                
                has_glasses, conf, details = detector.detect_glasses(face)
                
                results.append({
                    'type': 'WITHOUT',
                    'has_glasses': has_glasses,
                    'confidence': conf,
                    'details': details
                })
                
                print("\n[WITHOUT GLASSES]")
                print(detector.explain_detection(details))
        
        elif key == ord('2') and len(results) == 1:
            detections = face_detector.detect_faces(frame)
            if len(detections) > 0:
                face = face_detector.extract_face(frame, detections[0]['box'])
                
                has_glasses, conf, details = detector.detect_glasses(face)
                
                results.append({
                    'type': 'WITH',
                    'has_glasses': has_glasses,
                    'confidence': conf,
                    'details': details
                })
                
                print("\n[WITH GLASSES]")
                print(detector.explain_detection(details))
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Compare
    if len(results) == 2:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        
        without = results[0]
        with_glasses = results[1]
        
        print(f"\nWITHOUT GLASSES:")
        print(f"  Detected: {'YES ⚠️ FALSE POSITIVE' if without['has_glasses'] else 'NO ✓ CORRECT'}")
        print(f"  Confidence: {without['confidence']:.1%}")
        
        print(f"\nWITH GLASSES:")
        print(f"  Detected: {'YES ✓ CORRECT' if with_glasses['has_glasses'] else 'NO ⚠️ FALSE NEGATIVE'}")
        print(f"  Confidence: {with_glasses['confidence']:.1%}")
        
        diff = with_glasses['confidence'] - without['confidence']
        print(f"\nDifference: {diff:.1%}")
        
        if diff > 0.30:
            print("\n✅ EXCELLENT detection capability")
        elif diff > 0.20:
            print("\n✅ GOOD detection capability")
        elif diff > 0.10:
            print("\n⚠️  MODERATE detection capability")
        else:
            print("\n❌ POOR detection - needs further tuning")
        
        print("="*60)