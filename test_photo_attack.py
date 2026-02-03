"""
Test if system accepts photos (it shouldn't!)
"""

import cv2
import numpy as np
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager
from simple_liveness_detector import SimpleLivenessDetector
import config

print("="*60)
print("PHOTO ATTACK TEST")
print("="*60)

detector = FaceDetector()
recognizer = FaceRecognitionModel()
db = DatabaseManager()

# Check if liveness is enabled
if config.ENABLE_SIMPLE_LIVENESS:
    liveness_detector = SimpleLivenessDetector()
    print("\n✅ Liveness detection: ENABLED")
    print(f"   Threshold: {config.LIVENESS_THRESHOLD:.1%}")
else:
    liveness_detector = None
    print("\n❌ Liveness detection: DISABLED")
    print("   ⚠️  System WILL accept photos!")

print("\nInstructions:")
print("  1. First, show your real face")
print("  2. Then, hold a PHOTO of yourself")
print("  3. Compare results")
print("\nPress SPACE to test, Q to quit")
print("="*60 + "\n")

cap = cv2.VideoCapture(0)

test_count = 0
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    display = frame.copy()
    
    # Instructions
    if test_count == 0:
        cv2.putText(display, "Test 1: Show REAL FACE - Press SPACE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif test_count == 1:
        cv2.putText(display, "Test 2: Show PHOTO - Press SPACE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(display, "Tests complete - Press Q", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow('Photo Attack Test', display)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' ') and test_count < 2:
        # Detect face
        detections = detector.detect_faces(frame)
        
        if len(detections) == 0:
            print("❌ No face detected - try again")
            continue
        
        # Extract face
        box = detections[0]['box']
        face = detector.extract_face(frame, box)
        face = cv2.resize(face, (112, 112))
        
        # Get embedding
        embedding = recognizer.get_embedding(face)
        
        # Match
        matched_name, distance = db.find_match(embedding, recognizer)
        
        # Liveness check
        if liveness_detector:
            is_live, liveness_score, liveness_details = liveness_detector.check_liveness(face)
        else:
            is_live = True
            liveness_score = 1.0
            liveness_details = {}
        
        # Store result
        test_type = "REAL FACE" if test_count == 0 else "PHOTO"
        result = {
            'type': test_type,
            'matched': matched_name,
            'distance': distance,
            'is_live': is_live,
            'liveness_score': liveness_score,
            'liveness_details': liveness_details
        }
        results.append(result)
        
        # Print result
        print(f"\n{'='*60}")
        print(f"TEST {test_count + 1}: {test_type}")
        print(f"{'='*60}")
        print(f"Matched: {matched_name if matched_name else 'Unknown'}")
        print(f"Distance: {distance:.4f}")
        if liveness_detector:
            print(f"Liveness: {'LIVE' if is_live else 'FAKE'} ({liveness_score:.1%})")
            print(f"Details: {liveness_details}")
        else:
            print(f"Liveness: NOT CHECKED")
        
        # Decision
        if matched_name and is_live:
            print(f"Decision: ✅ GRANT")
        else:
            if not matched_name:
                print(f"Decision: ❌ DENY (Unknown)")
            else:
                print(f"Decision: ❌ DENY (Liveness Failed)")
        
        test_count += 1
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final comparison
if len(results) == 2:
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    real_result = results[0]
    photo_result = results[1]
    
    print(f"\nREAL FACE:")
    print(f"  Match: {real_result['matched']}")
    print(f"  Distance: {real_result['distance']:.4f}")
    print(f"  Liveness: {real_result['liveness_score']:.1%}")
    print(f"  Status: {'✅ GRANTED' if real_result['is_live'] else '❌ DENIED'}")
    
    print(f"\nPHOTO:")
    print(f"  Match: {photo_result['matched']}")
    print(f"  Distance: {photo_result['distance']:.4f}")
    print(f"  Liveness: {photo_result['liveness_score']:.1%}")
    print(f"  Status: {'❌ SHOULD BE DENIED' if photo_result['is_live'] else '✅ CORRECTLY DENIED'}")
    
    print("\n" + "="*60)
    print("SECURITY ASSESSMENT")
    print("="*60)
    
    if not liveness_detector:
        print("❌ CRITICAL: No liveness detection!")
        print("   System accepts photos - NOT SECURE")
        print("   Fix: Set ENABLE_SIMPLE_LIVENESS = True in config.py")
    
    elif photo_result['is_live']:
        print("❌ FAILED: Photo detected as LIVE!")
        print("   System is vulnerable to photo attacks")
        print("   Fix: Lower LIVENESS_THRESHOLD in config.py")
        print(f"   Current threshold: {config.LIVENESS_THRESHOLD:.1%}")
        print(f"   Photo scored: {photo_result['liveness_score']:.1%}")
        recommended = photo_result['liveness_score'] + 0.05
        print(f"   Recommended: {recommended:.1%}")
    
    else:
        print("✅ PASSED: Photo correctly rejected!")
        print("   Anti-spoofing is working")
        print(f"   Real face: {real_result['liveness_score']:.1%}")
        print(f"   Photo: {photo_result['liveness_score']:.1%}")
        print(f"   Difference: {(real_result['liveness_score'] - photo_result['liveness_score']):.1%}")
    
    print("="*60)

else:
    print("\nTest incomplete")