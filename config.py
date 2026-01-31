"""
Configuration settings for the Real-Time Facial Recognition Alert System
"""

# Face Detection Settings
FACE_DETECTION_CONFIDENCE = 0.7  # Minimum confidence for face detection (0-1) - Balance between detection and quality
MIN_FACE_SIZE = 60  # Minimum face size in pixels - Increased from 20 to get better quality faces

# Face Recognition Settings - CRITICAL FOR ACCURACY
RECOGNITION_THRESHOLD = 1.0  # Maximum distance for a match (lower = stricter) - Increased from 0.6 to 1.0 for FaceNet
# FaceNet typically uses 0.8-1.2 range - This threshold will be calibrated based on real-world testing
# NOTE: Higher threshold (1.0) allows for better recognition of authorized users under varying conditions
# (lighting, angles, facial expressions) while still maintaining security with the improved face quality
# from higher MIN_FACE_SIZE (60px) and FACE_DETECTION_CONFIDENCE (0.7) settings
EMBEDDING_SIZE = 128  # Size of face embeddings
DEBUG_MODE = True  # Print detailed debug information (distance values, matching process) - ENABLED FOR TESTING
SHOW_DISTANCE_ON_SCREEN = True  # Display distance on bounding box - ENABLED FOR DEBUGGING

# ============================================
# ACCURACY ENHANCEMENT SETTINGS
# ============================================

# Model Selection
USE_INSIGHTFACE = True  # Use InsightFace (ArcFace) if available, fallback to FaceNet
INSIGHTFACE_MODEL = 'buffalo_l'  # Options: 'buffalo_l', 'buffalo_s', 'antelopev2'
GPU_ENABLED = False  # Enable GPU acceleration if available (set True only if GPU/CUDA is installed)

# Face Quality Checks
ENABLE_QUALITY_CHECKS = True
BLUR_THRESHOLD = 100.0  # Laplacian variance (higher = sharper)
BRIGHTNESS_RANGE = (40, 220)  # Acceptable brightness range
MIN_CONTRAST = 30  # Minimum contrast (std dev)
MAX_POSE_ANGLE = 30  # Maximum head rotation in degrees
MIN_FACE_RESOLUTION = 112  # Minimum face size in pixels
OVERALL_QUALITY_THRESHOLD = 75  # Overall quality score (0-100)

# Face Alignment
ENABLE_FACE_ALIGNMENT = True
ALIGNED_FACE_SIZE = (112, 112)  # Standard aligned face size

# Liveness Detection
LIVENESS_ENABLED = False  # Set to True to enable anti-spoofing (may reduce performance)
LIVENESS_METHOD = 'motion'  # Options: 'motion', 'blink', 'texture', 'combined'
LIVENESS_FRAMES_REQUIRED = 5  # Frames to analyze for motion
REQUIRE_BLINK = False  # Require blink detection (slower but more secure)
BLINK_TIMEOUT = 3  # Seconds to wait for blink
TEXTURE_ANALYSIS_THRESHOLD = 0.7

# Recognition Improvements
# NOTE: When USE_INSIGHTFACE is True, RECOGNITION_THRESHOLD should be lower (0.4-0.6 vs 0.8-1.2 for FaceNet)
# The threshold is automatically adjusted when InsightFace is loaded
USE_KNN_MATCHING = True  # Use k-nearest neighbors instead of single match
KNN_K = 3  # Number of neighbors to consider
ADAPTIVE_THRESHOLD_PER_USER = True  # Use personalized thresholds
MIN_MATCH_CONFIDENCE = 0.75  # Minimum confidence for positive match

# Enhanced Enrollment
ENROLLMENT_SAMPLES = 10  # Increased from 5 for better coverage
ENROLLMENT_QUALITY_THRESHOLD = 80  # Higher quality required for enrollment
CAPTURE_POSE_VARIATIONS = True  # Guide user through different poses

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
