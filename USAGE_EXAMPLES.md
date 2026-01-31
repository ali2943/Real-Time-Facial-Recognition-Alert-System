# Usage Examples: Real-Time Facial Recognition System

This document provides practical, copy-paste ready code examples for common use cases.

## Table of Contents
1. [Basic Setup](#basic-setup)
2. [User Management](#user-management)
3. [Recognition Examples](#recognition-examples)
4. [Custom Integration](#custom-integration)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting Scripts](#troubleshooting-scripts)

---

## Basic Setup

### Install and Verify

```bash
# Clone repository
git clone https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System.git
cd Real-Time-Facial-Recognition-Alert-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2; import mtcnn; from keras_facenet import FaceNet; print('✓ Installation successful')"
```

### Quick Start (5 Minutes)

```bash
# 1. Enroll yourself
python enroll_user.py --name "Your Name" --samples 10

# 2. Run the system
python main.py

# 3. Test recognition (look at camera)
# Press 'q' to quit
```

---

## User Management

### Enroll a New User

```python
# File: custom_enrollment.py
"""
Custom enrollment script with validation
"""
import cv2
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager
import config

def enroll_user_custom(name, num_samples=10):
    """
    Enroll a user with custom validation
    
    Args:
        name: User's name
        num_samples: Number of face samples to capture
    """
    detector = FaceDetector()
    recognizer = FaceRecognitionModel()
    db = DatabaseManager()
    
    cap = cv2.VideoCapture(0)
    embeddings = []
    sample_count = 0
    
    print(f"Enrolling {name}...")
    print("Press SPACE to capture, ESC to cancel")
    
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw instructions
        cv2.putText(frame, f"Captured: {sample_count}/{num_samples}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw face boxes
        for face in faces:
            box = face['box']
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Enrollment', frame)
        
        key = cv2.waitKey(1)
        
        # Capture on SPACE
        if key == ord(' ') and len(faces) > 0:
            # Extract first face
            box = faces[0]['box']
            face_img = detector.extract_face(frame, box)
            
            # Generate embedding
            embedding = recognizer.get_embedding(face_img)
            embeddings.append(embedding)
            
            sample_count += 1
            print(f"✓ Sample {sample_count} captured")
        
        # Cancel on ESC
        elif key == 27:
            print("Enrollment cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    # Save to database
    db.add_user(name, embeddings)
    print(f"✓ {name} enrolled successfully with {num_samples} samples")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    enroll_user_custom("John Doe", num_samples=15)
```

### Batch Enrollment from Images

```python
# File: batch_enroll.py
"""
Enroll users from existing images
"""
import cv2
import os
import glob
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager

def enroll_from_directory(directory_path):
    """
    Enroll users from directory structure:
    directory/
        john_doe/
            photo1.jpg
            photo2.jpg
            ...
        jane_smith/
            photo1.jpg
            ...
    
    Args:
        directory_path: Path to directory containing user folders
    """
    detector = FaceDetector()
    recognizer = FaceRecognitionModel()
    db = DatabaseManager()
    
    # Find all user folders
    user_folders = [f for f in os.listdir(directory_path) 
                   if os.path.isdir(os.path.join(directory_path, f))]
    
    print(f"Found {len(user_folders)} users to enroll")
    
    for user_name in user_folders:
        user_path = os.path.join(directory_path, user_name)
        
        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(user_path, ext)))
        
        print(f"\nEnrolling {user_name} ({len(image_files)} images)...")
        
        embeddings = []
        for img_path in image_files:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ✗ Failed to load: {img_path}")
                continue
            
            # Detect face
            faces = detector.detect_faces(img)
            if len(faces) == 0:
                print(f"  ✗ No face found: {img_path}")
                continue
            
            # Extract and get embedding
            box = faces[0]['box']
            face_img = detector.extract_face(img, box)
            embedding = recognizer.get_embedding(face_img)
            embeddings.append(embedding)
            
            print(f"  ✓ Processed: {os.path.basename(img_path)}")
        
        if len(embeddings) > 0:
            db.add_user(user_name, embeddings)
            print(f"✓ {user_name} enrolled with {len(embeddings)} samples")
        else:
            print(f"✗ {user_name} enrollment failed (no valid images)")

if __name__ == "__main__":
    enroll_from_directory("./user_photos")
```

### List and Manage Users

```python
# File: user_management.py
"""
User management utilities
"""
from database_manager import DatabaseManager
import numpy as np

def list_users_detailed():
    """Print detailed information about enrolled users"""
    db = DatabaseManager()
    users = db.get_all_users()
    
    print("=" * 60)
    print("ENROLLED USERS")
    print("=" * 60)
    
    for i, name in enumerate(users, 1):
        embeddings = db.get_user_embeddings(name)
        
        print(f"\n{i}. {name}")
        print(f"   Samples: {len(embeddings)}")
        
        # Calculate embedding statistics
        if len(embeddings) > 1:
            # Convert to numpy array for calculations
            emb_array = np.array(embeddings)
            
            # Intra-user variance (how much samples vary)
            variance = np.var(emb_array, axis=0).mean()
            print(f"   Variance: {variance:.4f}")
            
            # Average pairwise distance
            distances = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    distances.append(dist)
            
            avg_dist = np.mean(distances)
            print(f"   Avg intra-user distance: {avg_dist:.4f}")
            print(f"   Quality: {'Excellent' if avg_dist < 0.5 else 'Good' if avg_dist < 0.8 else 'Fair'}")
    
    print("\n" + "=" * 60)
    print(f"Total users: {len(users)}")
    print("=" * 60)

def remove_user_safe(name):
    """Safely remove a user with confirmation"""
    db = DatabaseManager()
    
    if name not in db.get_all_users():
        print(f"✗ User '{name}' not found")
        return False
    
    # Get user info
    embeddings = db.get_user_embeddings(name)
    print(f"User: {name}")
    print(f"Samples: {len(embeddings)}")
    
    # Confirm
    confirm = input("Remove this user? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled")
        return False
    
    # Remove
    db.remove_user(name)
    print(f"✓ Removed {name}")
    return True

if __name__ == "__main__":
    list_users_detailed()
```

---

## Recognition Examples

### Single Image Recognition

```python
# File: recognize_image.py
"""
Recognize a person from a single image
"""
import cv2
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager

def recognize_from_image(image_path):
    """
    Recognize person in an image
    
    Args:
        image_path: Path to image file
    """
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognitionModel()
    db = DatabaseManager()
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ Failed to load image: {image_path}")
        return
    
    # Detect faces
    faces = detector.detect_faces(img)
    print(f"Found {len(faces)} face(s)")
    
    # Process each face
    for i, face in enumerate(faces):
        box = face['box']
        confidence = face['confidence']
        
        print(f"\nFace {i+1}:")
        print(f"  Detection confidence: {confidence:.3f}")
        
        # Extract face
        face_img = detector.extract_face(img, box)
        
        # Get embedding
        embedding = recognizer.get_embedding(face_img)
        
        # Find match
        name, distance = db.find_match(embedding, recognizer)
        
        if name:
            print(f"  ✓ Recognized: {name}")
            print(f"  Distance: {distance:.4f}")
        else:
            print(f"  ✗ Unknown person")
            print(f"  Best distance: {distance:.4f}")
        
        # Draw on image
        x, y, w, h = box
        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        label = name if name else "Unknown"
        cv2.putText(img, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Show result
    cv2.imshow('Recognition Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_from_image("test_image.jpg")
```

### Video File Recognition

```python
# File: recognize_video.py
"""
Process video file for facial recognition
"""
import cv2
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager

def process_video(video_path, output_path=None):
    """
    Process video file and recognize faces
    
    Args:
        video_path: Input video path
        output_path: Output video path (optional)
    """
    # Initialize
    detector = FaceDetector()
    recognizer = FaceRecognitionModel()
    db = DatabaseManager()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output writer if needed
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    recognized_faces = {}
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 5th frame for speed
        if frame_count % 5 != 0:
            if out:
                out.write(frame)
            continue
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        for face in faces:
            box = face['box']
            x, y, w, h = box
            
            # Extract and recognize
            face_img = detector.extract_face(frame, box)
            embedding = recognizer.get_embedding(face_img)
            name, distance = db.find_match(embedding, recognizer)
            
            # Track recognition
            if name:
                recognized_faces[name] = recognized_faces.get(name, 0) + 1
            
            # Draw box and label
            color = (0, 255, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{name} ({distance:.2f})" if name else "Unknown"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Write output
        if out:
            out.write(frame)
        
        # Show progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    
    # Summary
    print(f"\n✓ Processed {frame_count} frames")
    if recognized_faces:
        print("\nRecognized faces:")
        for name, count in sorted(recognized_faces.items(), 
                                 key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count} times")
    else:
        print("No faces recognized")
    
    if output_path:
        print(f"✓ Output saved to: {output_path}")

if __name__ == "__main__":
    process_video("input_video.mp4", "output_recognized.mp4")
```

---

## Custom Integration

### REST API Server

```python
# File: api_server.py
"""
Simple Flask API for facial recognition
"""
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager

app = Flask(__name__)

# Initialize once (global)
detector = FaceDetector()
recognizer = FaceRecognitionModel()
db = DatabaseManager()

def decode_image(base64_string):
    """Decode base64 image to OpenCV format"""
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Recognize faces in uploaded image
    
    Request:
        {
            "image": "base64_encoded_image"
        }
    
    Response:
        {
            "faces": [
                {
                    "name": "John Doe" or null,
                    "confidence": 0.95,
                    "distance": 0.42,
                    "box": [x, y, w, h]
                }
            ]
        }
    """
    try:
        # Get image from request
        data = request.get_json()
        image_b64 = data.get('image')
        
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode image
        img = decode_image(image_b64)
        
        # Detect faces
        faces = detector.detect_faces(img)
        
        results = []
        for face in faces:
            box = face['box']
            
            # Extract and recognize
            face_img = detector.extract_face(img, box)
            embedding = recognizer.get_embedding(face_img)
            name, distance = db.find_match(embedding, recognizer)
            
            results.append({
                "name": name,
                "distance": float(distance) if distance else None,
                "confidence": face['confidence'],
                "box": box
            })
        
        return jsonify({"faces": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/enroll', methods=['POST'])
def enroll():
    """
    Enroll a new user
    
    Request:
        {
            "name": "John Doe",
            "images": ["base64_1", "base64_2", ...]
        }
    """
    try:
        data = request.get_json()
        name = data.get('name')
        images_b64 = data.get('images', [])
        
        if not name or not images_b64:
            return jsonify({"error": "Name and images required"}), 400
        
        embeddings = []
        for img_b64 in images_b64:
            img = decode_image(img_b64)
            faces = detector.detect_faces(img)
            
            if len(faces) > 0:
                box = faces[0]['box']
                face_img = detector.extract_face(img, box)
                embedding = recognizer.get_embedding(face_img)
                embeddings.append(embedding)
        
        if len(embeddings) == 0:
            return jsonify({"error": "No valid faces found"}), 400
        
        db.add_user(name, embeddings)
        
        return jsonify({
            "success": True,
            "name": name,
            "samples": len(embeddings)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## Advanced Features

### Challenge-Response Liveness

```python
# File: challenge_liveness.py
"""
Interactive liveness detection with challenges
"""
import cv2
import time
from liveness_detector import LivenessDetector
from face_detector import FaceDetector

def test_liveness_with_challenge():
    """Test liveness with user challenges"""
    detector = FaceDetector()
    liveness = LivenessDetector()
    
    cap = cv2.VideoCapture(0)
    
    # Start challenge
    challenge, instruction = liveness.start_challenge()
    print(f"Challenge: {instruction}")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect face
        faces = detector.detect_faces(frame)
        
        if len(faces) > 0:
            box = faces[0]['box']
            landmarks = faces[0].get('keypoints')
            
            # Check liveness
            is_live, conf, reason = liveness.is_live(frame, box, landmarks)
            
            # Display instructions
            cv2.putText(frame, instruction, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            elapsed = time.time() - start_time
            remaining = liveness.challenge_timeout - elapsed
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Check completion
            completed, timed_out = liveness.check_challenge_response()
            
            if completed:
                cv2.putText(frame, "✓ CHALLENGE PASSED!", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow('Liveness Challenge', frame)
                cv2.waitKey(3000)
                break
            
            elif timed_out:
                cv2.putText(frame, "✗ TIMEOUT", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow('Liveness Challenge', frame)
                cv2.waitKey(3000)
                break
        
        cv2.imshow('Liveness Challenge', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    liveness.reset_challenge()

if __name__ == "__main__":
    test_liveness_with_challenge()
```

---

## Troubleshooting Scripts

### Test Camera

```python
# File: test_camera.py
"""Test camera functionality"""
import cv2

def test_camera(camera_index=0):
    """Test if camera is working"""
    print(f"Testing camera {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {camera_index}")
        print("Try different index: 0, 1, 2, ...")
        return False
    
    print(f"✓ Camera {camera_index} opened")
    
    # Get properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Resolution: {int(width)}x{int(height)}")
    print(f"FPS: {int(fps)}")
    print("\nPress 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Failed to read frame")
            break
        
        cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Camera test complete")
    return True

if __name__ == "__main__":
    test_camera()
```

### Calibrate Thresholds

```python
# File: calibrate_threshold.py
"""
Interactive threshold calibration tool
"""
import cv2
import numpy as np
from face_detector import FaceDetector
from face_recognition_model import FaceRecognitionModel
from database_manager import DatabaseManager

def calibrate_threshold(test_user):
    """
    Help calibrate recognition threshold for a specific user
    
    Args:
        test_user: Name of enrolled user to test
    """
    detector = FaceDetector()
    recognizer = FaceRecognitionModel()
    db = DatabaseManager()
    
    # Get user embeddings
    user_embeddings = db.get_user_embeddings(test_user)
    if not user_embeddings:
        print(f"✗ User '{test_user}' not found")
        return
    
    print(f"Testing threshold for: {test_user}")
    print(f"User has {len(user_embeddings)} enrolled samples")
    print("\nLook at camera and move around...")
    print("Press 'q' to finish\n")
    
    cap = cv2.VideoCapture(0)
    distances = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        faces = detector.detect_faces(frame)
        
        for face in faces:
            box = face['box']
            face_img = detector.extract_face(frame, box)
            embedding = recognizer.get_embedding(face_img)
            
            # Calculate distances to all user embeddings
            user_distances = [
                recognizer.compare_embeddings(embedding, user_emb)
                for user_emb in user_embeddings
            ]
            
            min_dist = min(user_distances)
            distances.append(min_dist)
            
            # Display distance
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Dist: {min_dist:.3f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show statistics
        if len(distances) > 0:
            mean_dist = np.mean(distances)
            max_dist = np.max(distances)
            
            cv2.putText(frame, f"Mean: {mean_dist:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Max: {max_dist:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Threshold Calibration', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Analysis
    if len(distances) > 0:
        print("\n" + "="*50)
        print("CALIBRATION RESULTS")
        print("="*50)
        print(f"Samples collected: {len(distances)}")
        print(f"Mean distance: {np.mean(distances):.4f}")
        print(f"Max distance: {np.max(distances):.4f}")
        print(f"Std deviation: {np.std(distances):.4f}")
        print("\nRECOMMENDED THRESHOLD:")
        
        recommended = np.mean(distances) + 2 * np.std(distances)
        print(f"  {recommended:.2f}")
        print("\nSet in config.py:")
        print(f"  RECOGNITION_THRESHOLD = {recommended:.2f}")
        print("="*50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python calibrate_threshold.py <username>")
    else:
        calibrate_threshold(sys.argv[1])
```

---

*For more examples and documentation, see the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) and [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md).*
