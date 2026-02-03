"""
Face Detection Module using MTCNN (Multi-task Cascaded Convolutional Networks)

DESIGN RATIONALE:
----------------
MTCNN is chosen over alternatives (Haar Cascade, HOG, SSD) because:
1. Deep learning-based: More accurate than traditional methods
2. Multi-task: Simultaneously detects faces AND facial landmarks
3. Robust: Works across varying lighting, angles, and occlusions
4. Lightweight: Fast enough for real-time on CPU
5. Cascade architecture: Filters candidates progressively for efficiency

MTCNN Architecture (3 stages):
1. P-Net (Proposal Network): Quick scan for face candidates
2. R-Net (Refine Network): Filters false positives
3. O-Net (Output Network): Final detection + 5 facial landmarks

Advantages over alternatives:
- Haar Cascade: Too many false positives, not rotation-invariant
- HOG: Struggles with occlusions and lighting
- YOLO/SSD: Overkill for face detection, slower
- dlib: Good but slower than MTCNN

LANDMARKS PROVIDED:
------------------
MTCNN returns 5 key facial landmarks per detected face:
- left_eye: Center of left eye
- right_eye: Center of right eye
- nose: Tip of nose
- mouth_left: Left corner of mouth
- mouth_right: Right corner of mouth

These landmarks enable:
- Face alignment (correct rotation)
- Pose estimation (check if face is frontal)
- Liveness detection (track eye blinks, head movement)
- Quality checks (verify eyes visible, face not too rotated)

LIMITATIONS:
-----------
- Works best on frontal faces (±30 degrees)
- Requires minimum face size (default 20px, configurable)
- May miss faces with heavy occlusions (masks, sunglasses)
- Sensitive to extreme lighting (very dark or overexposed)

For higher accuracy detection, consider:
- InsightFace RetinaFace (more accurate but heavier)
- MediaPipe Face Detection (faster but less accurate)
"""

import cv2
import numpy as np
from mtcnn import MTCNN
from config import config


class FaceDetector:
    """
    Face detector using MTCNN for robust real-time face detection
    
    This class wraps MTCNN with additional error handling and filtering
    to ensure reliable operation in production environments.
    """
    
    def __init__(self):
        """
        Initialize MTCNN detector
        
        MTCNN loads three neural networks (P-Net, R-Net, O-Net).
        Model weights are included in the mtcnn package.
        First initialization may take 1-2 seconds.
        """
        try:
            self.detector = MTCNN()
            print("[INFO] MTCNN Face Detector initialized successfully")
            print("[INFO] - Detection model: 3-stage cascade (P-Net -> R-Net -> O-Net)")
            print("[INFO] - Provides: Bounding boxes + 5 facial landmarks")
            print(f"[INFO] - Confidence threshold: {config.FACE_DETECTION_CONFIDENCE}")
            print(f"[INFO] - Minimum face size: {config.MIN_FACE_SIZE}px")
        except Exception as e:
            print(f"[ERROR] Failed to initialize MTCNN: {e}")
            raise
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame with comprehensive error handling
        
        PROCESS FLOW:
        1. Validate input frame (not None, not empty)
        2. Convert BGR (OpenCV) to RGB (MTCNN requirement)
        3. Run MTCNN detection
        4. Filter results by confidence threshold
        5. Filter results by minimum face size
        6. Return valid detections with landmarks
        
        Args:
            frame: BGR image from OpenCV (numpy array, shape HxWx3)
            
        Returns:
            List of dictionaries, each containing:
            - box: [x, y, width, height] - bounding box coordinates
            - confidence: float (0-1) - detection confidence score
            - keypoints: dict with 5 facial landmarks:
                - left_eye: (x, y)
                - right_eye: (x, y)
                - nose: (x, y)
                - mouth_left: (x, y)
                - mouth_right: (x, y)
                
        ERROR HANDLING:
        - Invalid frame: Returns empty list []
        - MTCNN ValueError (common on empty detections): Returns []
        - Other exceptions: Logged and returns []
        
        FILTERING LOGIC:
        Face detection can return many false positives. We filter by:
        1. Confidence: Only keep detections above threshold (default 0.7)
           - Below 0.5: Likely false positive
           - 0.5-0.7: Uncertain, may be partial face or poor angle
           - Above 0.7: High confidence, likely real face
        
        2. Size: Only keep faces above minimum size (default 60px)
           - Too small: Not enough resolution for embedding extraction
           - Just right: Sufficient detail for recognition
           - Very large: Likely close to camera, good quality
        
        DEBUG OUTPUT:
        When DEBUG_MODE is enabled, logs:
        - Number of raw detections from MTCNN
        - Number after confidence filtering
        - Number after size filtering
        - Individual detection confidence scores
        """
        try:
            # 1. Validate input frame
            if frame is None or frame.size == 0:
                if config.DEBUG_MODE:
                    print("[DEBUG] detect_faces: Invalid frame (None or empty)")
                return []
            
            # 2. Convert color space (MTCNN expects RGB, OpenCV uses BGR)
            # Why RGB: MTCNN was trained on RGB images
            # Skipping this would result in poor detection accuracy
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 3. Run MTCNN detection
            # May raise ValueError if no faces detected - this is normal, not an error
            try:
                detections = self.detector.detect_faces(rgb_frame)
            except ValueError as e:
                # MTCNN sometimes raises ValueError on empty output
                # This is expected behavior when no faces are in frame
                if config.DEBUG_MODE:
                    print("[DEBUG] MTCNN returned no detections (ValueError - normal)")
                return []
            
            # 4. Handle None or empty results
            if detections is None:
                if config.DEBUG_MODE:
                    print("[DEBUG] MTCNN returned None")
                return []
            
            if config.DEBUG_MODE and len(detections) > 0:
                print(f"[DEBUG] MTCNN raw detections: {len(detections)}")
            
            # 5. Filter detections by confidence and size
            valid_detections = []
            for detection in detections:
                confidence = detection['confidence']
                box = detection['box']
                
                # Check confidence threshold
                if confidence < config.FACE_DETECTION_CONFIDENCE:
                    if config.DEBUG_MODE:
                        print(f"[DEBUG] Rejected low confidence detection: {confidence:.3f} < {config.FACE_DETECTION_CONFIDENCE}")
                    continue
                
                # Check minimum face size (both width and height)
                width, height = box[2], box[3]
                if width < config.MIN_FACE_SIZE or height < config.MIN_FACE_SIZE:
                    if config.DEBUG_MODE:
                        print(f"[DEBUG] Rejected small face: {width}x{height} < {config.MIN_FACE_SIZE}px")
                    continue
                
                # Valid detection - add to results
                if config.DEBUG_MODE:
                    print(f"[DEBUG] Valid face detected: conf={confidence:.3f}, size={width}x{height}px")
                
                valid_detections.append(detection)
            
            if config.DEBUG_MODE:
                print(f"[DEBUG] Final valid detections: {len(valid_detections)}")
            
            return valid_detections
        
        except Exception as e:
            # Catch any other unexpected errors
            # Important: Don't crash the entire system on detection failure
            # Log the error and return empty list to allow system to continue
            print(f"[ERROR] Face detection failed with unexpected error: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return []
    
    def extract_face(self, frame, box, margin=20):
        """
        Extract face region from frame with safety margin
        
        PURPOSE:
        After detecting a face, we need to extract just that region for:
        1. Face alignment (rotation correction)
        2. Embedding extraction (feed to FaceNet/InsightFace)
        3. Quality checks (blur, brightness, etc.)
        
        MARGIN RATIONALE:
        Adding margin around detected box is important because:
        - Detection box is often tight around face
        - Face alignment may need surrounding pixels
        - Some facial features (ears, hairline) may be just outside box
        - Better to include extra background than crop face
        
        Default 20px margin is empirically determined:
        - Too small (<10px): May crop important features
        - Just right (20px): Includes full face with context
        - Too large (>40px): Includes too much background, wastes computation
        
        Args:
            frame: Original video frame (HxW x3 BGR image)
            box: Face bounding box [x, y, width, height]
            margin: Pixels to add around detected box (default: 20)
            
        Returns:
            Extracted face image (numpy array)
            Returns empty array if extraction fails
            
        COORDINATE SAFETY:
        We clip coordinates to frame boundaries to prevent:
        - Negative indices (face at frame edge)
        - Out-of-bounds access (face partially out of frame)
        - Crashes from invalid array slicing
        
        Examples:
        - Face at left edge: x=0, margin would give x1=-20 -> clip to 0
        - Face at bottom: y+h=720, margin would give y2=740 -> clip to 720
        """
        x, y, w, h = box
        
        # Calculate coordinates with margin
        # Add margin on all sides for context
        x1 = x - margin
        y1 = y - margin
        x2 = x + w + margin
        y2 = y + h + margin
        
        # Clip to frame boundaries (prevent out-of-bounds access)
        # max(0, ...) prevents negative coordinates
        # min(frame.shape[...], ...) prevents exceeding frame dimensions
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)  # frame.shape[1] is width
        y2 = min(frame.shape[0], y2)  # frame.shape[0] is height
        
        # Extract face region using numpy array slicing
        # Format: frame[y1:y2, x1:x2] (rows, cols)
        # Note: y comes before x because images are stored as [rows, cols, channels]
        face = frame[y1:y2, x1:x2]
        
        # Validate extraction
        if face.size == 0:
            if config.DEBUG_MODE:
                print(f"[DEBUG] Face extraction failed: empty region (box={box}, margin={margin})")
        elif config.DEBUG_MODE:
            print(f"[DEBUG] Face extracted: size={face.shape[1]}x{face.shape[0]}px")
        
        return face
