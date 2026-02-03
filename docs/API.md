# API Documentation

Complete API reference for the Real-Time Facial Recognition Alert System.

---

## Table of Contents

- [Overview](#overview)
- [Core API](#core-api)
- [Quality API](#quality-api)
- [Security API](#security-api)
- [Advanced API](#advanced-api)
- [Utilities API](#utilities-api)
- [Usage Examples](#usage-examples)

---

## Overview

This document provides detailed API documentation for all major components of the system. Each class and method is documented with parameters, return values, and usage examples.

---

## Core API

### FaceDetector

Detects faces in images using MTCNN.

```python
from src.core.face_detector import FaceDetector
```

#### `__init__(self, min_face_size=40, thresholds=[0.6, 0.7, 0.7])`

Initialize face detector.

**Parameters:**
- `min_face_size` (int): Minimum detectable face size in pixels. Default: 40
- `thresholds` (list): MTCNN confidence thresholds for each stage. Default: [0.6, 0.7, 0.7]

**Example:**
```python
detector = FaceDetector(min_face_size=60, thresholds=[0.7, 0.8, 0.8])
```

#### `detect_faces(self, frame)`

Detect faces in a frame.

**Parameters:**
- `frame` (np.ndarray): Input image as BGR numpy array

**Returns:**
- `list`: List of face dictionaries with keys:
  - `box`: [x, y, width, height]
  - `confidence`: float (0.0-1.0)
  - `keypoints`: dict with eye, nose, mouth positions

**Example:**
```python
faces = detector.detect_faces(frame)
for face in faces:
    x, y, w, h = face['box']
    confidence = face['confidence']
```

---

### FaceRecognitionModel

Generates face embeddings using FaceNet or InsightFace.

```python
from src.core.face_recognition_model import FaceRecognitionModel
```

#### `__init__(self, model_name='facenet')`

Initialize face recognition model.

**Parameters:**
- `model_name` (str): Model to use ('facenet' or 'insightface'). Default: 'facenet'

**Example:**
```python
recognizer = FaceRecognitionModel(model_name='insightface')
```

#### `get_embedding(self, face_image)`

Generate embedding from face image.

**Parameters:**
- `face_image` (np.ndarray): Aligned face image, RGB format

**Returns:**
- `np.ndarray`: Face embedding vector (128 or 512 dimensions)

**Example:**
```python
embedding = recognizer.get_embedding(face_image)
print(f"Embedding shape: {embedding.shape}")  # (128,) or (512,)
```

#### `calculate_distance(self, embedding1, embedding2, metric='cosine')`

Calculate distance between two embeddings.

**Parameters:**
- `embedding1` (np.ndarray): First embedding
- `embedding2` (np.ndarray): Second embedding
- `metric` (str): Distance metric ('cosine' or 'euclidean'). Default: 'cosine'

**Returns:**
- `float`: Distance value (lower = more similar)

**Example:**
```python
distance = recognizer.calculate_distance(emb1, emb2, metric='cosine')
if distance < 0.6:
    print("Same person")
```

---

### DatabaseManager

Manages user embeddings database.

```python
from src.core.database_manager import DatabaseManager
```

#### `__init__(self, db_path='database/embeddings.pkl')`

Initialize database manager.

**Parameters:**
- `db_path` (str): Path to database file. Default: 'database/embeddings.pkl'

**Example:**
```python
db = DatabaseManager(db_path='custom_database.pkl')
```

#### `add_user(self, name, embeddings, metadata=None)`

Add a user to the database.

**Parameters:**
- `name` (str): User name
- `embeddings` (list): List of embedding arrays
- `metadata` (dict, optional): Additional user information

**Returns:**
- `bool`: True if successful

**Example:**
```python
embeddings = [emb1, emb2, emb3]
metadata = {'enrolled_date': '2024-01-15', 'department': 'Engineering'}
db.add_user('John Doe', embeddings, metadata)
```

#### `find_match(self, embedding, threshold=0.6)`

Find matching user for an embedding.

**Parameters:**
- `embedding` (np.ndarray): Face embedding to match
- `threshold` (float): Maximum distance for a match. Default: 0.6

**Returns:**
- `tuple`: (match_dict, confidence) or (None, 0.0) if no match
  - `match_dict`: {'name': str, 'distance': float, 'metadata': dict}
  - `confidence`: float (0.0-1.0)

**Example:**
```python
match, confidence = db.find_match(embedding, threshold=0.6)
if match:
    print(f"Matched: {match['name']} with {confidence*100:.1f}% confidence")
else:
    print("No match found")
```

#### `remove_user(self, name)`

Remove a user from database.

**Parameters:**
- `name` (str): User name to remove

**Returns:**
- `bool`: True if removed, False if not found

**Example:**
```python
if db.remove_user('John Doe'):
    print("User removed")
```

#### `list_users(self)`

Get list of all enrolled users.

**Returns:**
- `list`: List of user dictionaries with 'name', 'sample_count', 'metadata'

**Example:**
```python
users = db.list_users()
for user in users:
    print(f"{user['name']}: {user['sample_count']} samples")
```

#### `save(self)`

Save database to disk.

**Returns:**
- `bool`: True if successful

**Example:**
```python
db.save()
```

#### `load(self)`

Load database from disk.

**Returns:**
- `bool`: True if successful

**Example:**
```python
db.load()
```

---

### FaceAligner

Aligns and normalizes face images.

```python
from src.core.face_aligner import FaceAligner
```

#### `__init__(self, desired_size=160)`

Initialize face aligner.

**Parameters:**
- `desired_size` (int): Output face size. Default: 160

**Example:**
```python
aligner = FaceAligner(desired_size=160)
```

#### `align_face(self, frame, face_box, keypoints)`

Align face in frame.

**Parameters:**
- `frame` (np.ndarray): Input frame
- `face_box` (list): [x, y, width, height]
- `keypoints` (dict): Facial landmarks from detector

**Returns:**
- `np.ndarray`: Aligned face image

**Example:**
```python
aligned_face = aligner.align_face(frame, face['box'], face['keypoints'])
```

---

## Quality API

### FaceQualityChecker

Validates face image quality.

```python
from src.quality.face_quality_checker import FaceQualityChecker
```

#### `__init__(self)`

Initialize quality checker.

**Example:**
```python
quality_checker = FaceQualityChecker()
```

#### `check_quality(self, face_image)`

Check face image quality.

**Parameters:**
- `face_image` (np.ndarray): Face image to check

**Returns:**
- `dict`: Quality metrics with keys:
  - `blur_score`: float (higher = sharper)
  - `brightness`: float (0-255)
  - `contrast`: float
  - `is_acceptable`: bool
  - `reasons`: list of str (failure reasons)

**Example:**
```python
quality = quality_checker.check_quality(face_image)
if quality['is_acceptable']:
    print(f"Quality OK: blur={quality['blur_score']:.1f}")
else:
    print(f"Quality issues: {quality['reasons']}")
```

#### `check_blur(self, image, threshold=100)`

Check if image is blurry.

**Parameters:**
- `image` (np.ndarray): Image to check
- `threshold` (float): Blur threshold. Default: 100

**Returns:**
- `tuple`: (blur_score, is_acceptable)

**Example:**
```python
blur_score, acceptable = quality_checker.check_blur(image, threshold=100)
```

#### `check_brightness(self, image, min_val=40, max_val=220)`

Check if brightness is in acceptable range.

**Parameters:**
- `image` (np.ndarray): Image to check
- `min_val` (float): Minimum brightness. Default: 40
- `max_val` (float): Maximum brightness. Default: 220

**Returns:**
- `tuple`: (brightness, is_acceptable)

**Example:**
```python
brightness, acceptable = quality_checker.check_brightness(image)
```

---

### ImagePreprocessor

Preprocesses images for recognition.

```python
from src.quality.image_preprocessor import ImagePreprocessor
```

#### `__init__(self)`

Initialize preprocessor.

**Example:**
```python
preprocessor = ImagePreprocessor()
```

#### `preprocess(self, image)`

Preprocess image.

**Parameters:**
- `image` (np.ndarray): Input image

**Returns:**
- `np.ndarray`: Preprocessed image

**Example:**
```python
processed = preprocessor.preprocess(image)
```

#### `adjust_lighting(self, image)`

Adjust image lighting.

**Parameters:**
- `image` (np.ndarray): Input image

**Returns:**
- `np.ndarray`: Lighting-adjusted image

**Example:**
```python
adjusted = preprocessor.adjust_lighting(image)
```

---

## Security API

### LivenessDetector

Anti-spoofing liveness detection.

```python
from src.security.liveness_detector import LivenessDetector
```

#### `__init__(self, method='combined')`

Initialize liveness detector.

**Parameters:**
- `method` (str): Detection method ('texture', 'motion', 'blink', 'combined'). Default: 'combined'

**Example:**
```python
liveness = LivenessDetector(method='combined')
```

#### `detect(self, frame, face_region)`

Detect if face is live.

**Parameters:**
- `frame` (np.ndarray): Full frame
- `face_region` (dict): Face detection result

**Returns:**
- `dict`: Liveness result with keys:
  - `is_live`: bool
  - `confidence`: float (0.0-1.0)
  - `scores`: dict of individual component scores
  - `method`: str (method used)

**Example:**
```python
result = liveness.detect(frame, face)
if result['is_live']:
    print(f"Live face detected: {result['confidence']*100:.1f}% confident")
else:
    print(f"Spoof detected: {result['scores']}")
```

#### `analyze_texture(self, face_image)`

Analyze face texture for spoofing.

**Parameters:**
- `face_image` (np.ndarray): Face image

**Returns:**
- `float`: Texture score (0.0-1.0, higher = more likely live)

**Example:**
```python
texture_score = liveness.analyze_texture(face_image)
```

#### `analyze_frequency(self, face_image)`

Analyze frequency domain for artifacts.

**Parameters:**
- `face_image` (np.ndarray): Face image

**Returns:**
- `float`: Frequency score (0.0-1.0)

**Example:**
```python
freq_score = liveness.analyze_frequency(face_image)
```

#### `detect_blink(self, frame_sequence)`

Detect eye blinks in frame sequence.

**Parameters:**
- `frame_sequence` (list): List of consecutive frames

**Returns:**
- `bool`: True if blink detected

**Example:**
```python
frames = [frame1, frame2, frame3, ...]
blink_detected = liveness.detect_blink(frames)
```

---

### FaceOcclusionDetector

Detects face occlusions (masks, sunglasses, etc.).

```python
from src.security.face_occlusion_detector import FaceOcclusionDetector
```

#### `__init__(self)`

Initialize occlusion detector.

**Example:**
```python
occlusion_detector = FaceOcclusionDetector()
```

#### `detect_occlusion(self, face_image)`

Detect occlusions in face image.

**Parameters:**
- `face_image` (np.ndarray): Face image

**Returns:**
- `dict`: Occlusion result with keys:
  - `has_occlusion`: bool
  - `occlusion_score`: float (0.0-1.0)
  - `occluded_regions`: list of str
  - `details`: dict

**Example:**
```python
result = occlusion_detector.detect_occlusion(face_image)
if result['has_occlusion']:
    print(f"Occluded regions: {result['occluded_regions']}")
```

---

### EyeStateDetector

Detects eye state (open/closed).

```python
from src.security.eye_state_detector import EyeStateDetector
```

#### `__init__(self)`

Initialize eye state detector.

**Example:**
```python
eye_detector = EyeStateDetector()
```

#### `detect_eyes(self, face_image, landmarks)`

Detect eye state.

**Parameters:**
- `face_image` (np.ndarray): Face image
- `landmarks` (dict): Facial landmarks

**Returns:**
- `dict`: Eye state with keys:
  - `left_eye_open`: bool
  - `right_eye_open`: bool
  - `ear_left`: float (Eye Aspect Ratio)
  - `ear_right`: float

**Example:**
```python
eye_state = eye_detector.detect_eyes(face_image, landmarks)
if eye_state['left_eye_open'] and eye_state['right_eye_open']:
    print("Both eyes open")
```

---

## Advanced API

### CompleteFaceRecognitionPipeline

Complete end-to-end processing pipeline.

```python
from src.advanced.complete_pipeline import CompleteFaceRecognitionPipeline
```

#### `__init__(self, config_dict=None)`

Initialize complete pipeline.

**Parameters:**
- `config_dict` (dict, optional): Custom configuration

**Example:**
```python
pipeline = CompleteFaceRecognitionPipeline()
```

#### `process_frame(self, frame)`

Process a single frame end-to-end.

**Parameters:**
- `frame` (np.ndarray): Input frame

**Returns:**
- `list`: List of recognition results

**Example:**
```python
results = pipeline.process_frame(frame)
for result in results:
    print(f"{result['name']}: {result['confidence']}")
```

---

### MultiEmbeddingGenerator

Generates and fuses multiple embeddings.

```python
from src.advanced.multi_embeddings import MultiEmbeddingGenerator
```

#### `__init__(self)`

Initialize multi-embedding generator.

**Example:**
```python
multi_emb = MultiEmbeddingGenerator()
```

#### `generate_multi_embeddings(self, face_images)`

Generate embeddings from multiple face images.

**Parameters:**
- `face_images` (list): List of face images

**Returns:**
- `np.ndarray`: Fused embedding

**Example:**
```python
embeddings = multi_emb.generate_multi_embeddings([img1, img2, img3])
```

---

## Utilities API

### Logging Functions

```python
from src.utils.utils import log_access_event, setup_logging
```

#### `log_access_event(name, granted, confidence=None)`

Log an access event.

**Parameters:**
- `name` (str): User name or "Unknown"
- `granted` (bool): Whether access was granted
- `confidence` (float, optional): Recognition confidence

**Example:**
```python
log_access_event("John Doe", granted=True, confidence=0.94)
log_access_event("Unknown", granted=False)
```

#### `setup_logging(log_file='logs/access_log.txt')`

Setup logging configuration.

**Parameters:**
- `log_file` (str): Path to log file

**Example:**
```python
setup_logging('logs/custom_log.txt')
```

### Display Functions

```python
from src.utils.utils import draw_face_box, display_message
```

#### `draw_face_box(frame, box, color=(0, 255, 0), thickness=2)`

Draw bounding box on frame.

**Parameters:**
- `frame` (np.ndarray): Frame to draw on
- `box` (list): [x, y, width, height]
- `color` (tuple): BGR color. Default: green
- `thickness` (int): Line thickness. Default: 2

**Example:**
```python
draw_face_box(frame, [100, 100, 200, 200], color=(0, 255, 0))
```

#### `display_message(frame, message, position='center', color=(255, 255, 255))`

Display text message on frame.

**Parameters:**
- `frame` (np.ndarray): Frame to draw on
- `message` (str): Text to display
- `position` (str or tuple): Position ('center', 'top', 'bottom') or (x, y)
- `color` (tuple): BGR color

**Example:**
```python
display_message(frame, "ACCESS GRANTED", position='center', color=(0, 255, 0))
```

---

## Usage Examples

### Complete Recognition Example

```python
from src.core.face_detector import FaceDetector
from src.core.face_recognition_model import FaceRecognitionModel
from src.core.database_manager import DatabaseManager
from src.security.liveness_detector import LivenessDetector
from src.quality.face_quality_checker import FaceQualityChecker
import cv2

# Initialize components
detector = FaceDetector()
recognizer = FaceRecognitionModel()
database = DatabaseManager()
liveness = LivenessDetector()
quality_checker = FaceQualityChecker()

# Capture frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Detect faces
faces = detector.detect_faces(frame)

for face in faces:
    # Extract face region
    x, y, w, h = face['box']
    face_img = frame[y:y+h, x:x+w]
    
    # Check quality
    quality = quality_checker.check_quality(face_img)
    if not quality['is_acceptable']:
        print(f"Quality issues: {quality['reasons']}")
        continue
    
    # Check liveness
    liveness_result = liveness.detect(frame, face)
    if not liveness_result['is_live']:
        print("Spoof detected!")
        continue
    
    # Generate embedding
    embedding = recognizer.get_embedding(face_img)
    
    # Find match
    match, confidence = database.find_match(embedding)
    
    if match:
        print(f"ACCESS GRANTED: {match['name']} ({confidence*100:.1f}%)")
    else:
        print("ACCESS DENIED: Unknown person")

cap.release()
```

### Enrollment Example

```python
from src.core.face_detector import FaceDetector
from src.core.face_recognition_model import FaceRecognitionModel
from src.core.database_manager import DatabaseManager
import cv2

# Initialize
detector = FaceDetector()
recognizer = FaceRecognitionModel()
database = DatabaseManager()

# Capture samples
cap = cv2.VideoCapture(0)
embeddings = []
samples_needed = 10

while len(embeddings) < samples_needed:
    ret, frame = cap.read()
    faces = detector.detect_faces(frame)
    
    if len(faces) == 1:
        x, y, w, h = faces[0]['box']
        face_img = frame[y:y+h, x:x+w]
        
        embedding = recognizer.get_embedding(face_img)
        embeddings.append(embedding)
        
        print(f"Captured {len(embeddings)}/{samples_needed}")
        cv2.waitKey(500)  # Wait between captures

# Enroll user
database.add_user("John Doe", embeddings)
database.save()

cap.release()
print("Enrollment complete!")
```

### Batch Processing Example

```python
from src.core.face_detector import FaceDetector
from src.core.face_recognition_model import FaceRecognitionModel
import cv2

detector = FaceDetector()
recognizer = FaceRecognitionModel()

# Process video file
video = cv2.VideoCapture('input_video.mp4')
results = []

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        face_img = frame[y:y+h, x:x+w]
        embedding = recognizer.get_embedding(face_img)
        results.append({
            'frame_num': video.get(cv2.CAP_PROP_POS_FRAMES),
            'embedding': embedding,
            'box': face['box']
        })

video.release()
print(f"Processed {len(results)} faces")
```

---

## Error Handling

All API functions may raise the following exceptions:

- `ValueError`: Invalid parameters
- `FileNotFoundError`: Missing database or model files
- `RuntimeError`: Processing errors
- `ImportError`: Missing dependencies

**Example:**
```python
try:
    detector = FaceDetector()
    faces = detector.detect_faces(frame)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Detection error: {e}")
```

---

## See Also

- [README.md](../README.md) - Project overview
- [INSTALLATION.md](INSTALLATION.md) - Installation guide
- [USAGE.md](USAGE.md) - Usage guide
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
