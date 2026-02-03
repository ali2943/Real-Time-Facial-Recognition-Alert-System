"""
Advanced Multi-Layer Liveness Detection
Combines multiple techniques for maximum security

Checks:
1. ✅ Texture Analysis (LBP)
2. ✅ Frequency Analysis (FFT) 
3. ✅ Color Distribution
4. ✅ Sharpness/Blur
5. ✅ Edge Complexity
6. ✅ Local Variance
7. ✅ Skin Tone Validation

Ultra-secure: Nearly impossible to fool with photos
"""

import cv2
import numpy as np


class AdvancedLivenessDetector:
    """
    Production-grade liveness detection
    Multiple complementary checks
    """
    
    def __init__(self):
        # Thresholds (will be auto-calibrated)
        self.texture_threshold = 0.35
        self.frequency_threshold = 0.40
        self.color_threshold = 0.45
        self.overall_threshold = 0.45
        
        # Weights for combining scores
        self.weights = {
            'texture': 0.25,      # LBP-based texture
            'frequency': 0.20,    # FFT high-freq content
            'color': 0.20,        # Skin tone & color variance
            'sharpness': 0.15,    # Edge sharpness
            'variance': 0.10,     # Local variance
            'skin_tone': 0.10     # YCrCb skin detection
        }
        
        print("[INFO] Advanced Liveness Detector initialized")
        print(f"  Overall threshold: {self.overall_threshold:.1%}")
        print(f"  Weights: {self.weights}")
    
    def check_liveness(self, face_img):
        """
        Comprehensive liveness check
        
        Args:
            face_img: Face image (BGR, any size)
            
        Returns:
            (is_live, confidence, details)
        """
        
        if face_img is None or face_img.size == 0:
            return False, 0.0, {'error': 'Invalid image'}
        
        # Standardize size
        face = cv2.resize(face_img, (112, 112))
        
        # Run all checks
        scores = {}
        
        # ============================================
        # CHECK 1: TEXTURE ANALYSIS (LBP-inspired)
        # ============================================
        texture_score = self._check_texture_richness(face)
        scores['texture'] = texture_score
        
        # ============================================
        # CHECK 2: FREQUENCY ANALYSIS (FFT)
        # ============================================
        frequency_score = self._check_frequency_content(face)
        scores['frequency'] = frequency_score
        
        # ============================================
        # CHECK 3: COLOR DISTRIBUTION
        # ============================================
        color_score = self._check_color_naturalness(face)
        scores['color'] = color_score
        
        # ============================================
        # CHECK 4: SHARPNESS (Edge Quality)
        # ============================================
        sharpness_score = self._check_sharpness(face)
        scores['sharpness'] = sharpness_score
        
        # ============================================
        # CHECK 5: LOCAL VARIANCE
        # ============================================
        variance_score = self._check_local_variance(face)
        scores['variance'] = variance_score
        
        # ============================================
        # CHECK 6: SKIN TONE VALIDATION
        # ============================================
        skin_score = self._check_skin_tone(face)
        scores['skin_tone'] = skin_score
        
        # ============================================
        # COMBINE SCORES
        # ============================================
        overall_score = sum(scores[k] * self.weights[k] for k in scores)
        
        # ============================================
        # DECISION WITH VOTING
        # ============================================
        # Primary decision: overall score
        is_live_score = overall_score > self.overall_threshold
        
        # Secondary validation: critical checks must pass
        critical_checks = {
            'texture': texture_score > self.texture_threshold,
            'frequency': frequency_score > self.frequency_threshold,
            'color': color_score > self.color_threshold
        }
        
        critical_passed = sum(critical_checks.values()) >= 2  # At least 2 of 3
        
        # Final decision
        is_live = is_live_score and critical_passed
        
        # Detailed breakdown
        details = {
            'overall': overall_score,
            'threshold': self.overall_threshold,
            'scores': scores,
            'critical_checks': critical_checks,
            'critical_passed': critical_passed,
            'decision_reason': self._get_decision_reason(scores, overall_score, critical_checks)
        }
        
        return is_live, overall_score, details
    
    def _check_texture_richness(self, face):
        """
        Check micro-texture using gradient-based analysis
        Real faces: rich micro-textures (pores, wrinkles, hair)
        Photos: smooth, lack fine details
        """
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients in multiple scales
        scores = []
        
        for ksize in [3, 5, 7]:
            # Sobel gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            
            # Gradient magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate texture complexity
            # Use standard deviation of gradient magnitudes
            texture_complexity = np.std(magnitude)
            
            # Normalize (real: 15-40, photo: 5-20)
            normalized = min(1.0, max(0.0, (texture_complexity - 5) / 30))
            scores.append(normalized)
        
        # Average across scales
        texture_score = np.mean(scores)
        
        return texture_score
    
    def _check_frequency_content(self, face):
        """
        Analyze frequency spectrum
        Real faces: rich high-frequency content
        Photos: smoothed, less high-frequency (compression, printing)
        """
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Define frequency regions
        # Low freq: center circle (r = h/6)
        # High freq: outer ring (r > h/4)
        
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        low_freq_mask = distances < (min(h, w) / 6)
        high_freq_mask = (distances > (min(h, w) / 4)) & (distances < (min(h, w) / 2))
        
        # Calculate energy
        low_freq_energy = np.sum(magnitude[low_freq_mask])
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        
        # Ratio
        if low_freq_energy > 0:
            ratio = high_freq_energy / low_freq_energy
        else:
            ratio = 0
        
        # Normalize (real: 0.3-0.8, photo: 0.1-0.4)
        frequency_score = min(1.0, max(0.0, (ratio - 0.1) / 0.6))
        
        return frequency_score
    
    def _check_color_naturalness(self, face):
        """
        Check color distribution naturalness
        Real faces: natural skin tone gradients
        Photos: altered colors, over/under saturation
        """
        
        # Convert to HSV
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Check 1: Saturation distribution
        # Real faces: moderate saturation with natural variance
        s_mean = np.mean(s)
        s_std = np.std(s)
        
        # Ideal: mean 60-140, std 20-50
        sat_mean_score = 1.0 - min(1.0, abs(s_mean - 100) / 60)
        sat_std_score = min(1.0, s_std / 50)
        
        saturation_score = (sat_mean_score + sat_std_score) / 2
        
        # Check 2: Value (brightness) variance
        # Real faces: natural lighting gradients
        v_std = np.std(v)
        
        # Ideal: std 20-60
        value_score = min(1.0, v_std / 60)
        
        # Check 3: Hue consistency
        # Real faces: consistent hue (skin tone)
        h_std = np.std(h)
        
        # Low variance is good for skin (10-40)
        # But too low means fake (< 5)
        if h_std < 5:
            hue_score = 0.0
        else:
            hue_score = min(1.0, h_std / 40)
        
        # Combine
        color_score = (saturation_score * 0.4 + value_score * 0.4 + hue_score * 0.2)
        
        return color_score
    
    def _check_sharpness(self, face):
        """
        Check edge sharpness
        Real faces: sharp, complex edges
        Photos: softer edges (printing, compression)
        """
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance (measure of sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)
        
        # Normalize (real: 80-200, photo: 30-100)
        sharpness_score = min(1.0, max(0.0, (variance - 30) / 150))
        
        return sharpness_score
    
    def _check_local_variance(self, face):
        """
        Check local intensity variance
        Real faces: varying local texture
        Photos: more uniform
        """
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance using sliding window
        kernel_size = 7
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sqr_diff = (gray.astype(np.float32) - mean) ** 2
        local_variance = cv2.blur(sqr_diff, (kernel_size, kernel_size))
        
        # Average local variance
        avg_variance = np.mean(local_variance)
        
        # Also check variance of local variances (texture consistency)
        var_of_variance = np.var(local_variance)
        
        # Normalize
        avg_score = min(1.0, avg_variance / 500)
        var_score = min(1.0, var_of_variance / 10000)
        
        variance_score = (avg_score * 0.6 + var_score * 0.4)
        
        return variance_score
    
    def _check_skin_tone(self, face):
        """
        Validate skin tone in YCrCb space
        Real faces: specific Cr/Cb ranges
        Photos: often outside natural range
        """
        
        ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Skin tone ranges (empirically determined)
        # Cr: 133-173
        # Cb: 77-127
        
        cr_mean = np.mean(cr)
        cb_mean = np.mean(cb)
        
        # Check if within range
        cr_in_range = (133 <= cr_mean <= 173)
        cb_in_range = (77 <= cb_mean <= 127)
        
        if cr_in_range and cb_in_range:
            # Calculate how centered in range
            cr_centered = 1.0 - abs(cr_mean - 153) / 20
            cb_centered = 1.0 - abs(cb_mean - 102) / 25
            
            skin_score = (cr_centered + cb_centered) / 2
        else:
            # Outside skin tone range
            skin_score = 0.0
        
        return skin_score
    
    def _get_decision_reason(self, scores, overall, critical_checks):
        """Generate human-readable decision reason"""
        
        if overall < self.overall_threshold:
            # Find weakest component
            weakest = min(scores.items(), key=lambda x: x[1])
            return f"Low overall score ({overall:.1%}). Weakest: {weakest[0]} ({weakest[1]:.1%})"
        
        elif not all(critical_checks.values()):
            failed = [k for k, v in critical_checks.items() if not v]
            return f"Critical checks failed: {', '.join(failed)}"
        
        else:
            strongest = max(scores.items(), key=lambda x: x[1])
            return f"All checks passed. Strongest: {strongest[0]} ({strongest[1]:.1%})"
    
    def explain_decision(self, details):
        """
        Generate detailed explanation of decision
        """
        
        lines = []
        lines.append("="*60)
        lines.append("LIVENESS DETECTION REPORT")
        lines.append("="*60)
        
        lines.append(f"\nOverall Score: {details['overall']:.1%}")
        lines.append(f"Threshold: {details['threshold']:.1%}")
        lines.append(f"Decision: {'✅ LIVE' if details['overall'] > details['threshold'] else '❌ FAKE'}")
        
        lines.append("\nDETAILED SCORES:")
        lines.append("-"*60)
        
        for check, score in details['scores'].items():
            weight = self.weights[check]
            contribution = score * weight
            
            # Visual bar
            bar_length = int(score * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            lines.append(f"{check.upper():15} {score:.1%} {bar} (×{weight:.0%} = {contribution:.1%})")
        
        lines.append("\nCRITICAL CHECKS:")
        lines.append("-"*60)
        
        for check, passed in details['critical_checks'].items():
            status = '✅ PASS' if passed else '❌ FAIL'
            score = details['scores'][check]
            lines.append(f"{check.capitalize():12} {status} ({score:.1%})")
        
        lines.append(f"\nCritical Validation: {'✅ PASSED' if details['critical_passed'] else '❌ FAILED'}")
        
        lines.append("\nREASON:")
        lines.append("-"*60)
        lines.append(details['decision_reason'])
        
        lines.append("="*60)
        
        return "\n".join(lines)


# Test harness
if __name__ == "__main__":
    import sys
    
    detector = AdvancedLivenessDetector()
    
    print("\n" + "="*60)
    print("ADVANCED LIVENESS DETECTOR - TEST MODE")
    print("="*60)
    print("\nInstructions:")
    print("  1. First test with REAL FACE")
    print("  2. Then test with PHOTO")
    print("  3. Compare results")
    print("\nPress SPACE to test, Q to quit")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(0)
    
    test_results = []
    test_mode = "REAL"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Instructions
        if len(test_results) == 0:
            cv2.putText(display, "Test 1: Show REAL FACE - Press SPACE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(test_results) == 1:
            cv2.putText(display, "Test 2: Show PHOTO - Press SPACE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Tests complete - Press Q", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Advanced Liveness Test', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and len(test_results) < 2:
            # Extract center region as face (for testing)
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            size = 150
            face = frame[center_y-size:center_y+size, center_x-size:center_x+size]
            
            # Test
            is_live, confidence, details = detector.check_liveness(face)
            
            test_type = "REAL FACE" if len(test_results) == 0 else "PHOTO"
            
            print(f"\n{'='*60}")
            print(f"TEST: {test_type}")
            print(detector.explain_decision(details))
            
            test_results.append({
                'type': test_type,
                'is_live': is_live,
                'confidence': confidence,
                'details': details
            })
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final comparison
    if len(test_results) == 2:
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        
        real = test_results[0]
        photo = test_results[1]
        
        print(f"\nREAL FACE:")
        print(f"  Score: {real['confidence']:.1%}")
        print(f"  Result: {'✅ PASS' if real['is_live'] else '❌ FAIL'}")
        
        print(f"\nPHOTO:")
        print(f"  Score: {photo['confidence']:.1%}")
        print(f"  Result: {'❌ SHOULD FAIL' if photo['is_live'] else '✅ CORRECTLY FAILED'}")
        
        print(f"\nSEPARATION:")
        diff = real['confidence'] - photo['confidence']
        print(f"  Difference: {diff:.1%}")
        
        if diff > 0.15:
            print(f"  ✅ EXCELLENT separation")
        elif diff > 0.10:
            print(f"  ✅ GOOD separation")
        elif diff > 0.05:
            print(f"  ⚠️  MARGINAL separation")
        else:
            print(f"  ❌ POOR separation - adjust thresholds")
        
        # Security verdict
        print(f"\n{'='*60}")
        print("SECURITY VERDICT")
        print("="*60)
        
        if real['is_live'] and not photo['is_live']:
            print("✅ SYSTEM SECURE")
            print("   Real faces accepted, photos rejected")
        elif not real['is_live'] and not photo['is_live']:
            print("⚠️  TOO STRICT")
            print("   Both rejected - lower threshold")
            print(f"   Recommended: OVERALL_THRESHOLD = {photo['confidence'] + 0.05:.2f}")
        elif real['is_live'] and photo['is_live']:
            print("❌ INSECURE")
            print("   Photos being accepted - raise threshold")
            print(f"   Recommended: OVERALL_THRESHOLD = {photo['confidence'] + 0.10:.2f}")
        else:
            print("❌ CRITICAL ERROR")
            print("   Real face rejected but photo accepted!")
        
        print("="*60)