"""
Quick Benchmark - Test 10 times
"""

import cv2
import time
from main import EnhancedFaceRecognitionSystem

def benchmark():
    system = EnhancedFaceRecognitionSystem()
    cap = cv2.VideoCapture(0)
    
    results = []
    
    print("\n" + "="*60)
    print("BENCHMARK - Testing 10 captures")
    print("="*60)
    print("Position your face and wait...\n")
    
    time.sleep(2)  # Wait for camera to stabilize
    
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Test {i+1}/10...", end=" ")
        
        start_time = time.time()
        result_frame = system.process_frame(frame)
        elapsed = time.time() - start_time
        
        # Check if granted (simple heuristic - check for green in frame)
        granted = np.mean(result_frame[:,:,1]) > np.mean(result_frame[:,:,0])  # More green than blue
        
        results.append({
            'granted': granted,
            'time': elapsed
        })
        
        print(f"{'✅ GRANTED' if granted else '❌ DENIED'} ({elapsed:.2f}s)")
        
        time.sleep(0.5)
    
    cap.release()
    
    # Stats
    granted_count = sum(1 for r in results if r['granted'])
    avg_time = np.mean([r['time'] for r in results])
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total tests: {len(results)}")
    print(f"Granted: {granted_count}/{len(results)} ({granted_count/len(results)*100:.1f}%)")
    print(f"Denied: {len(results)-granted_count}/{len(results)}")
    print(f"Avg processing time: {avg_time:.2f}s")
    print("="*60)
    
    if granted_count >= 9:
        print("✅ EXCELLENT - System working perfectly!")
    elif granted_count >= 7:
        print("✅ GOOD - System working well")
    elif granted_count >= 5:
        print("⚠️  FAIR - Consider re-enrollment or tuning")
    else:
        print("❌ POOR - Re-enrollment recommended")

if __name__ == "__main__":
    import numpy as np
    benchmark() 