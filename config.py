"""
Configuration Settings for Real-Time Facial Recognition Alert System

This file contains all tunable parameters for the system. Each setting is documented
with its purpose, recommended values, and trade-offs.

QUICK START PRESETS:
-------------------
Maximum Accuracy (Best for high-security applications):
    USE_INSIGHTFACE = True
    ENABLE_QUALITY_CHECKS = True
    ENABLE_FACE_ALIGNMENT = True
    LIVENESS_ENABLED = True
    LIVENESS_METHOD = 'combined'
    MIN_MATCH_CONFIDENCE = 0.85

Balanced Performance (Recommended for most use cases):
    USE_INSIGHTFACE = True
    ENABLE_QUALITY_CHECKS = True
    ENABLE_FACE_ALIGNMENT = True
    LIVENESS_ENABLED = False
    MIN_MATCH_CONFIDENCE = 0.75

Maximum Speed (Fast but less secure):
    USE_INSIGHTFACE = False
    ENABLE_QUALITY_CHECKS = True
    ENABLE_FACE_ALIGNMENT = False
    LIVENESS_ENABLED = False
    MIN_MATCH_CONFIDENCE = 0.70

THRESHOLD TUNING GUIDE:
----------------------
Recognition Threshold:
- FaceNet: Use 0.8-1.2 (Euclidean distance)
- InsightFace/ArcFace: Use 0.4-0.6 (Cosine distance)
- Lower = stricter (fewer false accepts, more false rejects)
- Higher = looser (more false accepts, fewer false rejects)

Quality Thresholds:
- Blur: 100 is standard, increase to 150 for very sharp images only
- Brightness: (40, 220) covers most lighting conditions
- Contrast: 30 is minimum, increase to 50 for high-quality enrollment

Liveness Thresholds:
- Texture: 0.7 balances security and usability
- Motion variance: Automatically tuned based on camera/environment

DESIGN PHILOSOPHY:
-----------------
1. Security First: Default to stricter settings, allow relaxation if needed
2. Fail Gracefully: If a component fails, log it but don't block legitimate users
3. Progressive Enhancement: Core features work, advanced features add security/accuracy
4. Transparency: All decisions logged in DEBUG_MODE for troubleshooting

HARDWARE CONSIDERATIONS:
-----------------------
- Laptop webcam (720p/1080p): Use default settings
- Lower quality webcam (480p): Decrease MIN_FACE_SIZE to 40, BLUR_THRESHOLD to 80
- GPU available: Set GPU_ENABLED = True for 2-3x speedup
- Low-end CPU: Set FRAME_SKIP = 2, disable LIVENESS_ENABLED
"""

# ============================================
# FACE DETECTION SETTINGS
# ============================================
# Deep learning-based detection using MTCNN or RetinaFace (InsightFace)
# Trade-off: Higher confidence = fewer false detections but may miss faces
# Recommended: 0.7 for balanced detection, 0.5 for better recall, 0.9 for precision

FACE_DETECTION_CONFIDENCE = 0.7  # Minimum confidence for face detection (0-1)
# Why 0.7: Balances detection rate with quality. Lower values (0.5) detect more faces
# but include low-quality detections. Higher values (0.9) are very strict.

MIN_FACE_SIZE = 60  # Minimum face size in pixels
# Why 60: Ensures sufficient resolution for accurate embedding extraction.
# FaceNet/ArcFace work best with 112x112 or larger. 60px is minimum for decent quality.
# Decrease to 40 if users will be far from camera, increase to 80 for high-security.

# ============================================
# FACE RECOGNITION SETTINGS
# ============================================
# Embedding-based recognition using FaceNet or InsightFace (ArcFace)
# These settings are CRITICAL for accuracy

RECOGNITION_THRESHOLD = 0.7  # Stricter matching (lower = stricter)
# For FaceNet: 0.7 is strict, only strong matches accepted
# Values 0.8-1.0 allow weak matches → security risk!
# Lower threshold = fewer false accepts = better security
# Why 0.7 for FaceNet: Euclidean distance in 128-d space
# - 0.6-0.7: Strict (high security, recommended for access control)
# - 0.8-1.0: Balanced (general use, higher false accept risk)
# - 1.0-1.2: Loose (convenience over security, significant false accept rate)
# NOTE: InsightFace uses cosine distance, threshold auto-adjusted to 0.4-0.5

EMBEDDING_SIZE = 128  # Size of face embeddings (128 for FaceNet, 512 for InsightFace)
# Don't change unless you change the recognition model

DEBUG_MODE = True  # Print detailed debug information
# Enable for development/troubleshooting. Shows:
# - Face detection count per frame
# - Matching distances and confidence scores
# - Quality check results
# - Liveness detection details
# Set to False in production for cleaner logs

SHOW_DISTANCE_ON_SCREEN = True  # Display distance values on video feed
# Useful for calibrating thresholds. Shows embedding distance on bounding box.

# ============================================
# ACCURACY ENHANCEMENT SETTINGS
# ============================================

# --- Model Selection ---
# InsightFace (ArcFace) provides state-of-the-art accuracy, comparable to mobile Face ID
# Falls back to FaceNet if InsightFace unavailable

USE_INSIGHTFACE = True  # Use InsightFace (ArcFace) if available, fallback to FaceNet
# Why InsightFace: 
# - State-of-the-art accuracy (98%+ on LFW benchmark)
# - 512-dimensional embeddings (vs 128 for FaceNet) = better discrimination
# - Trained on larger datasets (millions of identities)
# - RetinaFace detector more accurate than MTCNN
# Trade-off: Slightly slower than FaceNet, requires more memory

INSIGHTFACE_MODEL = 'buffalo_l'  # Options: 'buffalo_l', 'buffalo_s', 'antelopev2'
# - buffalo_l: Best accuracy, larger model (~500MB), slower
# - buffalo_s: Good accuracy, smaller model (~200MB), faster
# - antelopev2: Lightweight, fastest, good for edge devices
# Recommended: buffalo_l for desktop, buffalo_s for laptop

GPU_ENABLED = False  # Enable GPU acceleration if available
# Set to True only if you have CUDA-compatible GPU installed
# Provides 2-3x speedup for InsightFace
# FaceNet doesn't benefit as much from GPU

# --- Face Quality Checks ---
# Ensures only high-quality faces are used for recognition and enrollment
# Poor quality faces (blurry, dark, off-angle) lead to false rejects

ENABLE_QUALITY_CHECKS = True  # Enable comprehensive quality assessment
# Why enabled: Prevents garbage-in-garbage-out problem
# Low-quality faces create poor embeddings leading to match failures
# Better to reject low quality and prompt user to adjust than accept and fail later

BLUR_THRESHOLD = 110.0  # Laplacian variance (higher = sharper)
# Detects motion blur and out-of-focus faces
# - 100: Standard threshold, rejects obviously blurry faces
# - 150: Stricter, only accepts very sharp faces
# - 80: More lenient for lower quality cameras
# Trade-off: Higher = better embeddings but more user frustration

BRIGHTNESS_RANGE = (40, 220)  # Acceptable brightness range (0-255 scale)
# Rejects faces that are too dark or overexposed
# - (40, 220): Standard range, handles most lighting
# - (50, 200): Stricter, better for controlled lighting
# - (30, 230): More lenient, for varying environments
# Face recognition fails in extreme lighting due to lost features

MIN_CONTRAST = 35  # Minimum contrast (standard deviation)
# Ensures sufficient variation in pixel intensities
# Low contrast = flat image = poor feature extraction
# Typical real face: 40-80, photo under glass: <30

MAX_POSE_ANGLE = 25  # Maximum head rotation in degrees
# Limits head pose variation (yaw, pitch, roll)
# - 30°: Standard, allows natural head movements
# - 20°: Stricter, requires near-frontal faces
# - 45°: Lenient, allows more head turn
# Large pose angles reduce landmark accuracy and embedding quality

MIN_FACE_RESOLUTION = 112  # Minimum face size in pixels
# After detection, face is resized to 112x112 for embedding extraction
# Source face must be at least this size to avoid upscaling artifacts
# 112 is standard for both FaceNet and InsightFace

OVERALL_QUALITY_THRESHOLD = 80  # Overall quality score (0-100)
# Weighted combination of all quality checks
# - 80: Strict, higher quality standards (current setting)
# - 75: Balanced, accepts good quality faces
# - 85: Very strict, only pristine faces
# - 65: Lenient, accepts borderline cases
# See face_quality_checker.py for weight distribution

# --- Face Alignment ---
# Normalizes face orientation before embedding extraction
# Critical for consistent embeddings across different head poses

ENABLE_FACE_ALIGNMENT = True  # Enable face alignment using landmarks
# Why enabled: Alignment dramatically improves recognition accuracy
# - Corrects head rotation (makes eyes horizontal)
# - Centers face in frame
# - Normalizes scale
# Without alignment, same person at different angles = different embeddings

ALIGNED_FACE_SIZE = (112, 112)  # Standard aligned face size
# Both FaceNet and InsightFace expect 112x112 input
# Don't change unless you retrain the model

# --- Liveness Detection (Anti-Spoofing) ---
# Prevents spoofing attacks from photos, videos, and masks
# CRITICAL for security but impacts performance

LIVENESS_ENABLED = False  # Set to True to enable anti-spoofing
# Why disabled by default: Performance impact (~30% slower)
# Enable for:
# - High-security applications (financial, healthcare)
# - Unattended kiosks
# - Systems vulnerable to photo/video attacks
# Not needed for:
# - Low-security demos
# - Supervised access control
# - Performance-critical applications

LIVENESS_METHOD = 'motion'  # Options: 'motion', 'blink', 'texture', 'combined'
# Design rationale for each method:
#
# 'motion': Detects natural micro-movements (breathing, slight head motion)
#   - Fastest (passive, no user action required)
#   - Defeats static photos and frozen video
#   - May be fooled by high-quality video replay
#   - Best for: General use, good balance of security and UX
#
# 'blink': Requires user to blink naturally
#   - Medium speed (requires 2-3 seconds of observation)
#   - Defeats photos and most videos (unless attacker blinks on cue)
#   - Requires good lighting for eye tracking
#   - Best for: Moderate security, acceptable UX impact
#
# 'texture': Analyzes surface texture for print/screen artifacts
#   - Works on single frame (fastest for decision)
#   - Detects print patterns, screen moire, lack of skin texture
#   - May struggle with high-quality prints in good lighting
#   - Best for: Quick checks, works with still images
#
# 'combined': Uses all three methods with voting
#   - Most secure (defeats all simple attacks)
#   - Slowest (~2-3 seconds)
#   - Best UX (passive + active detection)
#   - Best for: Maximum security applications

LIVENESS_FRAMES_REQUIRED = 5  # Frames to analyze for temporal patterns
# More frames = more reliable but slower
# 5 frames @ 30fps = ~170ms of data
# - 3: Fast but less reliable
# - 5: Balanced (recommended)
# - 10: More reliable, noticeable delay

REQUIRE_BLINK = False  # Require explicit blink detection
# If True, user MUST blink within timeout period
# Adds security but impacts UX (user must actively blink)
# Recommended: False for passive mode, True for high-security

BLINK_TIMEOUT = 3  # Seconds to wait for required blink
# Only applies if REQUIRE_BLINK = True
# - 3s: Standard, gives user time to notice prompt
# - 5s: More lenient
# - 2s: Faster but may frustrate users

TEXTURE_ANALYSIS_THRESHOLD = 0.7  # Texture score threshold (0-1)
# Lower = stricter (rejects more as fake)
# Higher = more lenient (may accept some prints)
# - 0.8: Lenient, few false rejects
# - 0.7: Balanced (recommended)
# - 0.6: Strict, may reject some real faces in poor lighting

# --- Advanced Recognition Features ---
# Enhanced matching strategies for better accuracy

# NOTE: When USE_INSIGHTFACE is True, RECOGNITION_THRESHOLD should be lower (0.4-0.6 vs 0.8-1.2 for FaceNet)
# The threshold is automatically adjusted when InsightFace is loaded

USE_KNN_MATCHING = True  # Use k-nearest neighbors instead of single match
# KNN compares against multiple enrollment samples and uses majority vote
# More robust than single-sample matching

KNN_K = 3  # Number of neighbors to consider
# K=3 means compare to 3 closest embeddings and use majority vote
# - K=1: Single match (baseline)
# - K=3: Good balance (recommended)
# - K=5+: Slower, marginal improvement

ADAPTIVE_THRESHOLD_PER_USER = True  # Use personalized thresholds per user
# Each user gets custom threshold based on their enrollment variance
# Users with consistent faces → tighter threshold
# Users with varying faces → looser threshold

MIN_MATCH_CONFIDENCE = 0.85  # Minimum confidence for positive match (0-1)
# Even if distance is below threshold, confidence must be high enough
# - 1.0: 100% confidence (maximum security, only exact matches)
# - 0.85: Very strict (high security, realistic and secure)
# - 0.75: Balanced (recommended for general use)
# - 0.65: Lenient (convenience)
# 85% provides high security while remaining usable
# Adjust to 0.90 for higher security or 0.80 for more leniency

# Force strict validation
REJECT_UNKNOWN_FACES = True  # Always reject if not in database (cannot be disabled)
REQUIRE_DATABASE_MATCH = True  # Must match someone in database

# Liveness Detection Movement Thresholds
MAX_MOVEMENT_THRESHOLD = 30.0  # Maximum pixels of natural head movement
# Used in motion-based liveness detection
# Too high: Won't catch someone moving a photo
# Too low: Natural movements trigger false positive
# 30px at typical webcam distance is empirically determined balance

# Enhanced Enrollment
ENROLLMENT_SAMPLES = 10  # Increased from 5 for better coverage
ENROLLMENT_QUALITY_THRESHOLD = 80  # Higher quality required for enrollment
CAPTURE_POSE_VARIATIONS = True  # Guide user through different poses
ENROLLMENT_ANGLES = [-15, -10, 0, 10, 15]  # Pose variations in degrees (left to right progression)
ENROLLMENT_MIN_SAMPLES = 5  # Minimum samples required for enrollment
ENROLLMENT_MAX_VARIANCE = 0.3  # Maximum allowed variance between embeddings

# Camera Settings
CAMERA_INDEX = 0  # Default camera (0 = primary webcam)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Display Settings
BBOX_COLOR_LEGIT = (0, 255, 0)  # Green for authorized users
BBOX_COLOR_UNKNOWN = (0, 0, 255)  # Red for unauthorized users
BBOX_THICKNESS = 2
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_OFFSET_Y = -10

# Alert Settings
ALERT_MESSAGE_LEGIT = "Legit Person"
ALERT_MESSAGE_UNKNOWN = "Alert: Unknown Person"
SAVE_UNKNOWN_FACES = True
UNKNOWN_FACES_DIR = "unknown_faces"

# Database Settings
DATABASE_DIR = "database"
EMBEDDINGS_FILE = "embeddings.pkl"

# Model Settings
FACENET_MODEL_PATH = None  # Will use Keras FaceNet implementation

# Security Door Access Control Settings
ACCESS_GRANTED_DISPLAY_TIME = 2  # Seconds to show access granted message
ACCESS_DENIED_DISPLAY_TIME = 3  # Seconds to show access denied message
ACCESS_RESULT_DISPLAY_TIME = 3  # Seconds to show verification result in on-click mode
ACCESS_COOLDOWN = 3  # Seconds between access attempts to prevent spam
ENABLE_AUDIO_FEEDBACK = False  # Play sounds for granted/denied (not implemented yet)
LOG_FILE_PATH = "access_log.txt"  # Path to access log file
AUTO_RECONNECT_CAMERA = True  # Automatically reconnect if camera fails
MAX_RECONNECT_ATTEMPTS = 5  # Maximum camera reconnection attempts
FRAME_SKIP = 1  # Process every Nth frame (1 = process all frames)

# Access Control Display Settings
ACCESS_GRANTED_TEXT = "ACCESS GRANTED"
ACCESS_DENIED_TEXT = "ACCESS DENIED"
SYSTEM_READY_TEXT = "System Ready"
ACCESS_TEXT_FONT_SCALE = 2.0  # Larger text for access messages
ACCESS_TEXT_THICKNESS = 3  # Thicker text for access messages
ACCESS_GRANTED_COLOR = (0, 255, 0)  # Green
ACCESS_DENIED_COLOR = (0, 0, 255)  # Red
SYSTEM_READY_COLOR = (255, 255, 255)  # White

# ============================================
# ADVANCED PREPROCESSING SETTINGS
# ============================================
USE_ADVANCED_PREPROCESSING = True  # Enable CLAHE, gamma correction, bilateral filtering
PREPROCESSING_MODE = 'full'  # Options: 'full', 'light', 'none'

# ============================================
# TEMPORAL CONSISTENCY SETTINGS
# ============================================
USE_TEMPORAL_CONSISTENCY = False  # Enable multi-frame verification (not suitable for on-click mode)
TEMPORAL_BUFFER_SIZE = 5  # Number of frames to consider
MIN_CONSENSUS_RATIO = 0.6  # Minimum 60% of frames must agree

# ============================================
# EMBEDDING NORMALIZATION SETTINGS
# ============================================
NORMALIZE_EMBEDDINGS = True  # Enable L2 normalization of embeddings
USE_EMBEDDING_WHITENING = False  # Advanced whitening (requires training data)

# ============================================
# QUALITY GATING SETTINGS
# ============================================
REQUIRE_MULTI_FRAME_QUALITY = False  # Quality check across multiple frames (not for on-click)
MIN_QUALITY_FRAMES = 3  # Consecutive quality frames required

# ============================================
# FACE VALIDATION SETTINGS
# ============================================

# Mask/Occlusion Detection
ENABLE_MASK_DETECTION = True  # Detect and reject masked faces
ENABLE_OCCLUSION_DETECTION = True  # Detect any face covering
MASK_DETECTION_CONFIDENCE = 0.7  # Minimum confidence for mask detection

# Eye State Validation
ENABLE_EYE_STATE_CHECK = True  # Ensure eyes are open
REQUIRE_BOTH_EYES_OPEN = True  # Both eyes must be open
EYE_ASPECT_RATIO_THRESHOLD = 0.21  # Below this = eyes closed

# Complete Face Visibility
REQUIRE_FULL_FACE_VISIBLE = True  # All facial features must be visible
MIN_FACIAL_FEATURES_VISIBLE = 4  # Minimum features (eyes, nose, mouth, etc.)
