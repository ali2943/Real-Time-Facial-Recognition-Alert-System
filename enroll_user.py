"""
Advanced User Enrollment
Creates high-quality database samples for better recognition
"""

import cv2
import numpy as np
import argparse
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager
from face_quality_checker import FaceQualityChecker
import config

class AdvancedEnrollment:
    def __init__(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognitionModel()
        self.db_manager = DatabaseManager()
        self.quality_checker = FaceQualityChecker()
        
        # Enrollment settings
        self.target_samples = 15  # More samples = better recognition
        self.min_quality = 80     # Higher quality threshold
        self.min_uniqueness = 0.12  # Ensure diverse samples
        
        self.collected_embeddings = []
        self.collected_faces = []
    
    def enroll(self, name):
        """Enroll user with guided process"""
        
        print("="*60)
        print(f"ADVANCED ENROLLMENT: {name}")
        print("="*60)
        
        # Check if user exists
        if name in self.db_manager.get_all_users():
            response = input(f"{name} already exists. Re-enroll? (y/n): ")
            if response.lower() != 'y':
                return
            # Delete old data
            del self.db_manager.embeddings_db[name]
        
        print(f"\nTarget: {self.target_samples} high-quality samples")
        print(f"Quality threshold: {self.min_quality}/100")
        print("\nInstructions:")
        print("  1. Look straight at camera")
        print("  2. Good lighting (not too bright/dark)")
        print("  3. Neutral expression")
        print("  4. We'll guide you through different angles")
        print("\nPress SPACE to capture each sample")
        print("Press Q to finish early (min 5 samples)")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        # Guided angles
        angles = [
            "Center - Look straight",
            "Turn head slightly LEFT",
            "Turn head slightly RIGHT",
            "Tilt head slightly UP",
            "Tilt head slightly DOWN",
            "Center again",
            "Slight smile - Center",
            "Center - Neutral",
            "Any angle - Variation 1",
            "Any angle - Variation 2",
            "Any angle - Variation 3",
            "Any angle - Variation 4",
            "Any angle - Variation 5",
            "Any angle - Variation 6",
            "Any angle - Variation 7"
        ]
        
        current_angle_idx = 0
        
        while len(self.collected_embeddings) < self.target_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            
            # Current instruction
            if current_angle_idx < len(angles):
                instruction = angles[current_angle_idx]
            else:
                instruction = "Any angle - Extra samples"
            
            # Show progress
            progress = f"Samples: {len(self.collected_embeddings)}/{self.target_samples}"
            cv2.putText(display, progress, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(display, instruction, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(display, "Press SPACE to capture", (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Detect face
            detections = self.detector.detect_faces(frame)
            
            if len(detections) > 0:
                box = detections[0]['box']
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract face
                face = self.detector.extract_face(frame, box)
                
                if face.size > 0:
                    # Check quality
                    quality_result = self.quality_checker.get_quality_score(face)
                    if isinstance(quality_result, tuple):
                        quality = quality_result[0]
                    else:
                        quality = quality_result
                    
                    # Show quality
                    color = (0, 255, 0) if quality >= self.min_quality else (0, 165, 255)
                    cv2.putText(display, f"Quality: {quality:.0f}/100", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Advanced Enrollment', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and len(detections) > 0:
                # Capture sample
                face = self.detector.extract_face(frame, detections[0]['box'])
                
                if face.size > 0:
                    # Check quality
                    quality_result = self.quality_checker.get_quality_score(face)
                    if isinstance(quality_result, tuple):
                        quality = quality_result[0]
                    else:
                        quality = quality_result
                    
                    if quality < self.min_quality:
                        print(f"❌ Quality too low: {quality:.0f}/100 (need {self.min_quality})")
                        continue
                    
                    # Resize and get embedding
                    face_resized = cv2.resize(face, (112, 112))
                    embedding = self.recognizer.get_embedding(face_resized)
                    
                    # Check uniqueness
                    if self._is_unique(embedding):
                        self.collected_embeddings.append(embedding)
                        self.collected_faces.append(face)
                        current_angle_idx += 1
                        
                        print(f"✅ Sample {len(self.collected_embeddings)}/{self.target_samples} "
                              f"captured (quality: {quality:.0f})")
                    else:
                        print(f"⚠️  Sample too similar to existing - try different angle")
            
            elif key == ord('q'):
                if len(self.collected_embeddings) >= 5:
                    break
                else:
                    print("Need at least 5 samples!")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(self.collected_embeddings) == 0:
            print("\n❌ No samples collected. Enrollment failed.")
            return
        
        # Analyze collected samples
        print("\n" + "="*60)
        print("ANALYZING SAMPLES")
        print("="*60)
        
        distances = []
        for i in range(len(self.collected_embeddings)):
            for j in range(i+1, len(self.collected_embeddings)):
                dist = np.linalg.norm(self.collected_embeddings[i] - self.collected_embeddings[j])
                distances.append(dist)
        
        print(f"Collected: {len(self.collected_embeddings)} samples")
        print(f"Intra-class distance:")
        print(f"  Min:  {min(distances):.4f}")
        print(f"  Max:  {max(distances):.4f}")
        print(f"  Avg:  {np.mean(distances):.4f}")
        print(f"  Std:  {np.std(distances):.4f}")
        
        # Quality check
        avg_dist = np.mean(distances)
        if avg_dist < 0.15:
            print("✅ Excellent sample quality!")
        elif avg_dist < 0.25:
            print("✅ Good sample quality")
        elif avg_dist < 0.35:
            print("⚠️  Acceptable sample quality")
        else:
            print("⚠️  High variance - might need re-enrollment")
        
        # Save to database
        print("\nSaving to database...")
        self.db_manager.embeddings_db[name] = self.collected_embeddings
        self.db_manager.save_database()
        
        print(f"\n✅ {name} enrolled successfully with {len(self.collected_embeddings)} samples!")
        print("="*60)
    
    def _is_unique(self, new_embedding):
        """Check if embedding is unique enough"""
        
        if len(self.collected_embeddings) == 0:
            return True
        
        # Calculate distances to existing embeddings
        distances = [np.linalg.norm(new_embedding - emb) 
                    for emb in self.collected_embeddings]
        
        min_distance = min(distances)
        
        return min_distance > self.min_uniqueness

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='User name')
    args = parser.parse_args()
    
    enrollment = AdvancedEnrollment()
    enrollment.enroll(args.name)