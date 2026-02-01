"""
Face Recognition Module using FaceNet Embeddings

OVERVIEW:
--------
FaceNet is a deep convolutional neural network that maps face images to a
compact Euclidean space where distances directly correspond to face similarity.

KEY CONCEPTS:
------------
1. **Embedding:** 128-dimensional vector representing a face
   - Each dimension captures specific facial features
   - Similar faces have similar embeddings (small Euclidean distance)
   - Different faces have different embeddings (large Euclidean distance)

2. **Triplet Loss Training:**
   FaceNet was trained using triplet loss:
   - Anchor: Reference face
   - Positive: Same person, different photo
   - Negative: Different person
   - Goal: distance(anchor, positive) < distance(anchor, negative)

3. **Why FaceNet?**
   - Well-established (Google, 2015)
   - Good accuracy (99.63% on LFW benchmark)
   - Reasonable model size (~90MB)
   - Fast inference on CPU (~40ms per face)
   - Keras implementation available

MODEL DETAILS:
-------------
- Architecture: Inception-ResNet V1
- Input Size: 160x160 RGB
- Output: 128-dimensional embedding
- Training Dataset: 200M+ face images
- Distance Metric: Euclidean (L2 norm)

TYPICAL DISTANCE RANGES:
-----------------------
- Same person, same conditions: 0.3 - 0.6
- Same person, different conditions: 0.6 - 1.0
- Different people, similar appearance: 1.0 - 1.3
- Different people, distinct appearance: 1.3 - 2.0

THRESHOLD GUIDELINES:
--------------------
- Very strict (high security): 0.8
- Balanced (recommended): 1.0
- Lenient (convenience): 1.2

LIMITATIONS:
-----------
- Requires good face alignment (eyes horizontal)
- Sensitive to extreme lighting changes
- Struggles with heavy occlusions (masks, sunglasses)
- Cannot distinguish identical twins
- Lower accuracy than newer models (InsightFace/ArcFace)

UPGRADE PATH:
------------
For better accuracy, consider upgrading to InsightFace:
- Higher accuracy (99.83% vs 99.63%)
- 512-d embeddings (more discriminative)
- Better with challenging poses/lighting
- See insightface_recognizer.py
"""

import cv2
import numpy as np
from keras_facenet import FaceNet
import config


class FaceRecognitionModel:
    """
    Face recognition using FaceNet embeddings
    
    This class provides a high-level interface for face recognition:
    1. Load pre-trained FaceNet model
    2. Preprocess face images
    3. Generate embeddings
    4. Compare embeddings for matching
    """
    
    def __init__(self):
        """
        Initialize FaceNet model
        
        Loads the pre-trained Inception-ResNet V1 model with FaceNet weights.
        Model is loaded once and reused for all subsequent predictions.
        
        First load may take 2-3 seconds (model download if not cached).
        Subsequent loads are instant (cached in ~/.keras/models/).
        
        Raises:
            Exception: If model fails to load (missing dependencies, corrupted cache)
        """
        print("[INFO] Loading FaceNet model...")
        print("[INFO] - Model: Inception-ResNet V1")
        print("[INFO] - Embedding dimension: 128")
        print("[INFO] - Input size: 160x160 RGB")
        
        try:
            self.model = FaceNet()
            self.input_size = (160, 160)  # FaceNet expects 160x160 input
            print("[INFO] FaceNet model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load FaceNet model: {e}")
            print("[ERROR] Try reinstalling: pip install keras-facenet tensorflow")
            raise
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for FaceNet model
        
        FaceNet has specific input requirements that must be met:
        1. Size: Exactly 160x160 pixels
        2. Color: RGB format (not BGR)
        3. Range: Pixel values normalized to [0, 1]
        4. Shape: 4D tensor (batch_size, height, width, channels)
        
        Args:
            face_img: Face image (RGB format, any size)
            
        Returns:
            Preprocessed face tensor ready for model.embeddings()
            Shape: (1, 160, 160, 3), dtype: float32, range: [0, 1]
            
        PREPROCESSING STEPS:
        -------------------
        1. **Resize to 160x160:**
           - FaceNet's input layer expects this exact size
           - Uses bilinear interpolation (smooth, no aliasing)
           - Aspect ratio NOT preserved (face may be slightly stretched)
           - Face alignment should be done BEFORE this step
        
        2. **Normalize to [0, 1]:**
           - Original: uint8 [0, 255]
           - Normalized: float32 [0, 1]
           - Why: Neural networks work better with small values
           - Formula: pixel_float = pixel_int / 255.0
        
        3. **Add Batch Dimension:**
           - Model expects batches: (batch_size, H, W, C)
           - Single image needs: (1, H, W, C)
           - np.expand_dims adds dimension at axis=0
        
        DESIGN NOTE:
        -----------
        We normalize to [0, 1] instead of [-1, 1] or standardization
        because FaceNet was trained this way. Using different normalization
        would degrade accuracy significantly.
        """
        # Step 1: Resize to FaceNet's required input size
        # Uses cv2.INTER_LINEAR (default) for smooth resizing
        face_resized = cv2.resize(face_img, self.input_size)
        
        # Step 2: Convert to float32 and normalize to [0, 1]
        # astype('float32') prevents overflow and is required by TensorFlow
        # Division by 255.0 (float) ensures floating-point division
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Step 3: Add batch dimension
        # Changes shape from (160, 160, 3) to (1, 160, 160, 3)
        # Required because model expects batches even for single image
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        if config.DEBUG_MODE:
            print(f"[DEBUG] Preprocessed face: shape={face_batch.shape}, "
                  f"dtype={face_batch.dtype}, range=[{face_batch.min():.3f}, {face_batch.max():.3f}]")
        
        return face_batch
    
    def get_embedding(self, face_img):
        """
        Generate 128-dimensional face embedding using FaceNet
        
        This is the core function that transforms a face image into a
        numerical representation that can be compared with other faces.
        
        Args:
            face_img: Face image in BGR format (OpenCV default)
                     Can be any size (will be resized to 160x160)
                     Should be well-aligned and cropped to face region
            
        Returns:
            embedding: 128-dimensional numpy array (float32)
                      Shape: (128,)
                      Range: typically [-5, 5] but unbounded
                      L2 norm: typically 1-10 (not normalized)
        
        PROCESS FLOW:
        ------------
        1. Convert BGR (OpenCV) → RGB (model requirement)
        2. Resize to 160x160 and normalize
        3. Pass through neural network
        4. Extract 128-d embedding from final layer
        5. Return embedding vector
        
        EMBEDDING PROPERTIES:
        -------------------
        - Dimensionality: 128 (fixed)
        - Each dimension represents learned facial features
        - Similar faces → similar embeddings (close in 128-d space)
        - Different faces → different embeddings (far apart)
        
        WHAT THE EMBEDDING CAPTURES:
        --------------------------
        The 128 dimensions encode:
        - Face shape (oval, round, square)
        - Eye spacing and size
        - Nose shape and size
        - Mouth shape
        - Facial proportions
        - Skin texture (to some degree)
        - Age-related features
        - Gender-related features
        
        NOT captured (by design):
        - Exact lighting conditions
        - Specific facial expression (mostly)
        - Hair style (mostly)
        - Makeup (mostly)
        - Accessories (glasses handled separately)
        
        EMBEDDING QUALITY DEPENDS ON:
        ----------------------------
        - Face alignment (critical - use face_aligner.py)
        - Image quality (resolution, blur, lighting)
        - Face pose (frontal best, profile worst)
        - Occlusions (masks, hands, etc.)
        
        ERROR HANDLING:
        --------------
        If embedding generation fails (invalid input, model error):
        - Exception is raised (not caught here)
        - Caller should handle exception
        - Common causes: empty image, corrupted image data
        
        PERFORMANCE:
        -----------
        - CPU: ~40ms per embedding
        - GPU: ~5ms per embedding
        - Bottleneck is neural network inference
        
        EXAMPLE USAGE:
        -------------
        ```python
        recognizer = FaceRecognitionModel()
        
        # Load image (OpenCV imread returns BGR format)
        face_img = cv2.imread('face.jpg')
        # get_embedding will automatically convert BGR to RGB
        
        embedding = recognizer.get_embedding(face_img)
        print(embedding.shape)  # (128,)
        ```
        """
        # Step 1: Convert color space
        # OpenCV uses BGR by default, but FaceNet expects RGB
        # This conversion is critical - BGR input would give wrong results
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        if config.DEBUG_MODE:
            print(f"[DEBUG] Generating embedding for face: shape={face_rgb.shape}")
        
        # Step 2: Preprocess image
        # Resizes to 160x160, normalizes to [0,1], adds batch dimension
        face_processed = self.preprocess_face(face_rgb)
        
        # Step 3: Generate embedding using FaceNet model
        # model.embeddings() returns shape (batch_size, 128)
        # We use batch_size=1, so result is (1, 128)
        embedding = self.model.embeddings(face_processed)
        
        # Step 4: Extract single embedding from batch
        # embedding[0] converts (1, 128) → (128,)
        embedding_vector = embedding[0]
        
        if config.DEBUG_MODE:
            print(f"[DEBUG] Embedding generated: shape={embedding_vector.shape}, "
                  f"norm={np.linalg.norm(embedding_vector):.2f}")
        
        return embedding_vector
    
    def compare_embeddings(self, embedding1, embedding2):
        """
        Compare two face embeddings using Euclidean distance (L2 norm)
        
        Calculates the straight-line distance between two points in 128-dimensional
        space. This is the fundamental similarity metric for FaceNet.
        
        Args:
            embedding1: First face embedding (128-d numpy array)
            embedding2: Second face embedding (128-d numpy array)
            
        Returns:
            distance: Euclidean distance between embeddings (float)
                     Range: 0 to ~2 (theoretically unbounded, practically ≤2)
                     0 = identical faces (or same image)
                     <0.6 = very likely same person
                     0.6-1.0 = probably same person
                     1.0-1.3 = uncertain
                     >1.3 = likely different people
        
        MATHEMATICAL FORMULA:
        --------------------
        distance = ||embedding1 - embedding2||₂
                 = sqrt(Σ(e1ᵢ - e2ᵢ)²) for i = 0 to 127
        
        Where:
        - ||·||₂ is the L2 norm (Euclidean norm)
        - e1ᵢ is the i-th dimension of embedding1
        - e2ᵢ is the i-th dimension of embedding2
        
        WHY EUCLIDEAN DISTANCE?
        ----------------------
        1. FaceNet was trained with triplet loss using Euclidean distance
        2. Embeddings are positioned in space such that Euclidean distance
           directly corresponds to face similarity
        3. Simple, fast to compute (no need for complex similarity metrics)
        4. Geometrically intuitive (physical distance in space)
        
        ALTERNATIVES NOT USED:
        ---------------------
        - **Cosine Similarity:** Better for high-dimensional sparse data
          FaceNet doesn't need this (embeddings are dense, direction matters less than distance)
        
        - **Manhattan Distance (L1):** Faster but less accurate
          FaceNet trained specifically for L2, not L1
        
        - **Chi-Square Distance:** Used for histograms, not applicable here
        
        INTERPRETATION GUIDE:
        --------------------
        Based on empirical testing:
        
        Distance | Interpretation        | Action
        ---------|----------------------|------------------
        0.0-0.3  | Same image/very close | Definitely match
        0.3-0.6  | Strong match          | High confidence match
        0.6-0.8  | Probable match        | Match if threshold ≥ 0.8
        0.8-1.0  | Weak match            | Match if threshold ≥ 1.0
        1.0-1.2  | Unlikely match        | Borderline, depends on threshold
        1.2-1.5  | Probably different    | Likely different person
        >1.5     | Definitely different  | Different person
        
        PERFORMANCE:
        -----------
        - Computation time: ~0.01ms (very fast)
        - Bottleneck is embedding generation, not comparison
        - Can compare millions of embeddings per second
        
        EXAMPLE USAGE:
        -------------
        ```python
        recognizer = FaceRecognitionModel()
        
        # Get embeddings for two faces
        emb1 = recognizer.get_embedding(face1)
        emb2 = recognizer.get_embedding(face2)
        
        # Compare them
        distance = recognizer.compare_embeddings(emb1, emb2)
        
        if distance < 0.6:
            print("Very likely the same person")
        elif distance < 1.0:
            print("Probably the same person")
        else:
            print("Likely different people")
        ```
        """
        # Calculate Euclidean distance (L2 norm of difference vector)
        # np.linalg.norm computes: sqrt(sum((e1 - e2)^2))
        distance = np.linalg.norm(embedding1 - embedding2)
        
        if config.DEBUG_MODE:
            print(f"[DEBUG] Embedding comparison: distance={distance:.4f}")
        
        return distance
    
    def is_match(self, embedding1, embedding2):
        """
        Determine if two embeddings represent the same person
        
        Simple threshold-based matching: if distance is below threshold,
        embeddings are considered a match (same person).
        
        Args:
            embedding1: First face embedding (128-d numpy array)
            embedding2: Second face embedding (128-d numpy array)
            
        Returns:
            True if embeddings match (same person), False otherwise
        
        DECISION LOGIC:
        --------------
        distance = euclidean_distance(embedding1, embedding2)
        match = distance < RECOGNITION_THRESHOLD
        
        THRESHOLD FROM CONFIG:
        ---------------------
        config.RECOGNITION_THRESHOLD (default: 1.0 for FaceNet)
        
        - Lower threshold = stricter matching = fewer false accepts, more false rejects
        - Higher threshold = looser matching = more false accepts, fewer false rejects
        
        WHEN TO USE:
        -----------
        This is a simple 1-to-1 comparison. Use when:
        - Verifying a claimed identity (1:1 matching)
        - Comparing two specific images
        - Quick yes/no decision needed
        
        DON'T USE FOR:
        -------------
        - Searching database of many users (use database_manager.find_match instead)
        - Need confidence score (use compare_embeddings directly)
        - Need to tune threshold dynamically (check distance manually)
        
        EXAMPLE:
        -------
        ```python
        recognizer = FaceRecognitionModel()
        
        # Enroll: Store John's embedding
        john_face = cv2.imread('john.jpg')
        john_embedding = recognizer.get_embedding(john_face)
        
        # Verify: Is this new face John?
        new_face = cv2.imread('unknown.jpg')
        new_embedding = recognizer.get_embedding(new_face)
        
        if recognizer.is_match(john_embedding, new_embedding):
            print("Welcome, John!")
        else:
            print("Sorry, you're not John")
        ```
        
        LIMITATIONS:
        -----------
        - Binary decision (yes/no) without confidence level
        - Single threshold may not work for all users (see adaptive thresholds)
        - Doesn't account for multiple enrollment samples per person
        
        FOR PRODUCTION:
        --------------
        Consider using enhanced_database_manager with:
        - Multiple enrollment samples per person
        - Adaptive per-user thresholds
        - K-nearest neighbors matching
        - Confidence scoring
        """
        # Compare embeddings and check against threshold
        distance = self.compare_embeddings(embedding1, embedding2)
        is_match = distance < config.RECOGNITION_THRESHOLD
        
        if config.DEBUG_MODE:
            match_str = "MATCH" if is_match else "NO MATCH"
            print(f"[DEBUG] Matching decision: {match_str} "
                  f"(distance={distance:.4f}, threshold={config.RECOGNITION_THRESHOLD})")
        
        return is_match
