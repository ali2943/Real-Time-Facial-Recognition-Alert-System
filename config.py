"""
Configuration Settings for Real-Time Facial Recognition Alert System
PRODUCTION-GRADE VERSION - Enhanced Multi-Stage Pipeline

LATEST UPDATES:
--------------
✅ Adaptive thresholds per user
✅ Multi-sample embedding fusion
✅ Intelligent decision engine
✅ Advanced preprocessing pipeline
✅ Smart liveness detection
✅ Lenient mask detection
✅ Balanced security and usability
✅ Adaptive lighting adjustment (NEW)

QUICK START PRESETS:
-------------------
Maximum Security (High-security applications):
    ENABLE_SIMPLE_LIVENESS = True
    DECISION_SCORE_THRESHOLD = 0.75
    MIN_MATCH_CONFIDENCE = 0.40
    MASK_CONFIDENCE_THRESHOLD = 0.85
    ENABLE_ADAPTIVE_LIGHTING = True

Production Recommended (Best balance):
    ENABLE_SIMPLE_LIVENESS = True
    DECISION_SCORE_THRESHOLD = 0.60
    MIN_MATCH_CONFIDENCE = 0.35
    MASK_CONFIDENCE_THRESHOLD = 0.90
    ENABLE_ADAPTIVE_LIGHTING = True

Development/Testing (Fast and lenient):
    ENABLE_SIMPLE_LIVENESS = False
    DECISION_SCORE_THRESHOLD = 0.55
    MIN_MATCH_CONFIDENCE = 0.30
    MASK_CONFIDENCE_THRESHOLD = 0.95
    ENABLE_ADAPTIVE_LIGHTING = True
"""

# ============================================
# FACE DETECTION SETTINGS
# ============================================

FACE_DETECTION_CONFIDENCE = 0.7  # Minimum confidence for face detection (0-1)
MIN_FACE_SIZE = 60  # Minimum face size in pixels

# ============================================
# FACE RECOGNITION SETTINGS (UPDATED)
# ============================================

# Base threshold (automatically adjusted by adaptive threshold manager)
RECOGNITION_THRESHOLD = 0.6  # Base threshold for matching
DEFAULT_RECOGNITION_THRESHOLD = 0.6  # Fallback if adaptive disabled

# Embedding settings
EMBEDDING_SIZE = 512  # 512 for keras-facenet, 128 for old FaceNet

# Debug settings
DEBUG_MODE = True  # Print detailed processing information
SHOW_DISTANCE_ON_SCREEN = True  # Display distance values on video feed

# ============================================
# MODEL SELECTION
# ============================================

USE_INSIGHTFACE = False  # Use InsightFace (currently not required)
INSIGHTFACE_MODEL = 'buffalo_l'
GPU_ENABLED = False  # Enable GPU acceleration

# ============================================
# QUALITY CHECKS (BALANCED)
# ============================================

ENABLE_QUALITY_CHECKS = True
BLUR_THRESHOLD = 100.0  # Lowered from 110 for more leniency
BRIGHTNESS_RANGE = (35, 225)  # Wider range than (40, 220)
MIN_CONTRAST = 30  # Lowered from 35
MAX_POSE_ANGLE = 30  # Increased from 25
MIN_FACE_RESOLUTION = 112
OVERALL_QUALITY_THRESHOLD = 60  # Lowered from 80 for more leniency

# ============================================
# FACE ALIGNMENT
# ============================================

ENABLE_FACE_ALIGNMENT = True
ALIGNED_FACE_SIZE = (112, 112)
FACE_SIZE = (112, 112)  # Standard face size for processing

# ============================================
# MASK DETECTION (LENIENT - FIXED)
# ============================================

ENABLE_MASK_DETECTION = False  # DISABLED by default (was causing too many false positives)

# If you enable it, use these lenient settings:
MASK_CONFIDENCE_THRESHOLD = 0.90  # Very high - only block obvious masks (was 0.75)
ENABLE_OCCLUSION_DETECTION = True
OCCLUSION_CONFIDENCE_THRESHOLD = 0.85  # High threshold
MASK_DETECTION_CONFIDENCE = 0.90  # Same as MASK_CONFIDENCE_THRESHOLD
REQUIRE_MULTIPLE_MASK_INDICATORS = True  # Need multiple signs of mask
USE_SOFT_VALIDATION = True  # Don't hard-fail on single check
VALIDATION_REQUIRED_PASSES = 1  # Only need 1 pass (lenient)

# Eye state validation (disabled for leniency)
ENABLE_EYE_STATE_CHECK = False
REQUIRE_BOTH_EYES_OPEN = False
EYE_ASPECT_RATIO_THRESHOLD = 0.15  # Very lenient
ALLOW_EYE_CHECK_SKIP = True

# Face visibility
REQUIRE_FULL_FACE_VISIBLE = False  # Disabled for leniency
MIN_FACIAL_FEATURES_VISIBLE = 3  # Lowered from 4

# ============================================
# ADVANCED PREPROCESSING (ENABLED)
# ============================================

ENABLE_ADVANCED_PREPROCESSING = False
PREPROCESSING_MODE = 'balanced'  # 'light', 'balanced', 'aggressive'
USE_ADVANCED_PREPROCESSING = True  # Alias for compatibility

# ============================================
# ADAPTIVE LIGHTING ADJUSTMENT (NEW)
# ============================================

# Enable automatic lighting correction
ENABLE_ADAPTIVE_LIGHTING = False 

# Lighting adjustment mode
# 'auto': Automatically detect and correct lighting issues
# 'brighten': Always brighten (for dark environments)
# 'darken': Always darken (for bright environments)  
# 'balance': Balance uneven lighting
# 'none': No lighting adjustment
LIGHTING_MODE = 'auto'

# Target brightness (0-255 scale)
# 128 is ideal middle brightness
TARGET_BRIGHTNESS = 128
BRIGHTNESS_TOLERANCE = 30  # ±30 around target

# Gamma correction settings
# Used for brightening/darkening
MIN_GAMMA = 0.5   # Maximum darkening
MAX_GAMMA = 2.5   # Maximum brightening
DEFAULT_GAMMA = 1.0  # No change

# CLAHE (Contrast Limited Adaptive Histogram Equalization) settings
# Enhances local contrast
CLAHE_CLIP_LIMIT = 2.0  # Higher = more aggressive contrast (1.0-4.0)
CLAHE_TILE_SIZE = (8, 8)  # Grid size for local enhancement

# Face-specific lighting
APPLY_LIGHTING_TO_FACES_ONLY = True  # Adjust only detected faces (recommended)
APPLY_LIGHTING_TO_FULL_FRAME = False  # Adjust entire camera feed

# Lighting feedback
SHOW_BRIGHTNESS_INDICATOR = True  # Show brightness bar on screen
SHOW_LIGHTING_INFO = True  # Print brightness stats in debug mode

# Debug lighting
DEBUG_LIGHTING = False  # Show before/after comparison
SAVE_LIGHTING_EXAMPLES = False  # Save adjusted images for tuning

# Lighting adjustment frequency
LIGHTING_UPDATE_INTERVAL = 1  # Update every N frames (1 = every frame)

# Auto camera settings (hardware level)
ENABLE_CAMERA_AUTO_EXPOSURE = True
ENABLE_CAMERA_AUTO_WB = True  # Auto white balance
CAMERA_BRIGHTNESS = 128  # Manual brightness (0-255) if auto disabled
CAMERA_CONTRAST = 128    # Manual contrast (0-255)
CAMERA_SATURATION = 128  # Manual saturation (0-255)

# ============================================
# MULTI-SAMPLE EMBEDDING (ENABLED)
# ============================================

ENABLE_MULTI_SAMPLE_EMBEDDING = True
NUM_EMBEDDING_SAMPLES = 3  # 3 samples averaged for robustness
# Increases recognition accuracy by 10-15%
# Trade-off: ~200ms slower per recognition

# ============================================
# ADAPTIVE THRESHOLDS (ENABLED)
# ============================================

USE_ADAPTIVE_THRESHOLDS = True
ADAPTIVE_THRESHOLD_PER_USER = True  # Per-user custom thresholds
# Each user gets threshold based on enrollment variance
# More consistent faces = stricter threshold
# More varied faces = looser threshold

# ============================================
# INTELLIGENT DECISION ENGINE (ENABLED)
# ============================================

USE_INTELLIGENT_DECISIONS = True
DECISION_SCORE_THRESHOLD = 0.45  # Lowered from 0.60 - Overall score threshold (0-1)

# Component weights in decision (must sum to 1.0):
DECISION_WEIGHTS = {
    'distance': 0.50,    # Face match distance (most important)
    'quality': 0.15,     # Image quality
    'liveness': 0.25,    # Anti-spoofing
    'temporal': 0.10     # Consistency over time
}

# Decision boundaries (ADJUSTED):
# >= 0.75: High confidence → GRANT
# >= 0.55: Medium confidence → GRANT (if liveness & quality OK) [LOWERED from 0.60]
# >= 0.40: Low confidence → DENY (or MFA if enabled) [LOWERED from 0.45]
# < 0.40: Very low → DENY

# ============================================
# LIVENESS DETECTION (SMART & LENIENT)
# ============================================

# ============================================
# MAXIMUM SECURITY MODE (MAIN DOOR)
# ============================================

# Advanced Liveness - ULTRA STRICT
USE_ADVANCED_LIVENESS = True
ADVANCED_LIVENESS_THRESHOLD = 0.65  # RAISED from 0.60

# Critical check thresholds - ALL RAISED
TEXTURE_THRESHOLD = 0.50  # From 0.45
FREQUENCY_THRESHOLD = 0.55  # From 0.50
COLOR_THRESHOLD = 0.60  # From 0.55
SHARPNESS_THRESHOLD = 0.50  # From 0.45

# Component weights - Prioritize most reliable
LIVENESS_WEIGHTS = {
    'texture': 0.35,      # INCREASED (most reliable)
    'frequency': 0.30,    # INCREASED (very reliable)
    'color': 0.05,        # DECREASED (can be fooled)
    'sharpness': 0.20,    # Same
    'variance': 0.10,     # Same
    'skin_tone': 0.00     # DISABLED (unreliable for photos)
}

# Decision settings - STRICTER
REQUIRE_CRITICAL_CHECKS = True
CRITICAL_CHECKS_MINIMUM = 4  # NEW - Require 4 out of 5 checks to pass

# Fail secure - NO fallback
LIVENESS_FALLBACK_ENABLED = False  # If check fails, DENY access

# Hard limits - Instant rejection
LIVENESS_HARD_LIMITS = {
    'texture': 0.35,      # Raised
    'frequency': 0.40,    # Raised
    'sharpness': 0.40     # Raised
}
# ============================================
# MAIN DOOR SECURITY - MAXIMUM PROTECTION
# ============================================

# ==================== TIER 1: Advanced Liveness ====================
USE_ADVANCED_LIVENESS = True
ADVANCED_LIVENESS_THRESHOLD = 0.65  # STRICT

# Critical thresholds
TEXTURE_THRESHOLD = 0.50
FREQUENCY_THRESHOLD = 0.55
COLOR_THRESHOLD = 0.60
SHARPNESS_THRESHOLD = 0.50

# Component weights
LIVENESS_WEIGHTS = {
    'texture': 0.35,
    'frequency': 0.30,
    'color': 0.05,
    'sharpness': 0.20,
    'variance': 0.10,
    'skin_tone': 0.00
}

# Strict requirements
REQUIRE_CRITICAL_CHECKS = True
CRITICAL_CHECKS_MINIMUM = 4  # Must pass 4 out of 5

# Fail secure
LIVENESS_FALLBACK_ENABLED = False

# Hard limits
LIVENESS_HARD_LIMITS = {
    'texture': 0.35,
    'frequency': 0.40,
    'sharpness': 0.40
}

# ==================== TIER 2: Motion Detection ====================
ENABLE_MOTION_LIVENESS = True  # NEW - Detects static photos

# Motion settings
MOTION_BUFFER_SIZE = 15  # Frames to analyze
MOTION_THRESHOLD = 0.45  # Threshold for detecting motion
MOTION_MINIMUM_FRAMES = 10  # Min frames before deciding

# Action when no motion detected
MOTION_FAILURE_ACTION = 'DENY'  # 'DENY' or 'WARN'

# ==================== TIER 3: Challenge-Response ====================
ENABLE_CHALLENGE_RESPONSE = False  # Set True for max security (slower but unbeatable)

# Challenge settings
CHALLENGE_TIMEOUT = 5.0  # Seconds to respond
CHALLENGE_TYPES = ['blink', 'smile']  # Available challenges
REQUIRE_CHALLENGE_SUCCESS = True  # Must pass challenge

# ==================== COMBINED DECISION ====================

# Security level preset
SECURITY_LEVEL = 'MAXIMUM'  # 'MAXIMUM', 'HIGH', 'MEDIUM', 'LOW'

# Multi-tier decision
REQUIRE_ALL_TIERS = True  # All tiers must pass (recommended for main door)

# Individual tier weights (if not requiring all)
TIER_WEIGHTS = {
    'advanced_liveness': 0.50,
    'motion': 0.35,
    'challenge': 0.15
}

# Overall threshold (if using weighted combination)
OVERALL_SECURITY_THRESHOLD = 0.75

# ==================== ANTI-SPOOFING MESSAGES ====================
PHOTO_DETECTED_MESSAGE = "Photo/Screen detected - Access denied"
NO_MOTION_MESSAGE = "No movement detected - Access denied"
CHALLENGE_FAILED_MESSAGE = "Challenge failed - Access denied"
ANALYZING_MESSAGE = "Analyzing... Please stay still"

# ============================================
# GLASSES DETECTION
# ============================================

# Enable glasses detection
ENABLE_GLASSES_DETECTION = True

# Action when glasses detected
GLASSES_ACTION = 'WARN'  # 'WARN', 'DENY', or 'IGNORE'

# Detection thresholds
GLASSES_CONFIDENCE_THRESHOLD = 0.60
GLASSES_EDGE_THRESHOLD = 0.15
GLASSES_REFLECTION_THRESHOLD = 200

# Messages
GLASSES_WARNING_MESSAGE = "Please remove glasses for better recognition"
GLASSES_DENY_MESSAGE = "Glasses detected - Access denied"

# ============================================
# MOTION LIVENESS DETECTION
# ============================================

# Enable motion-based liveness
ENABLE_MOTION_LIVENESS = True

# Motion settings
MOTION_BUFFER_SIZE = 15
MOTION_THRESHOLD = 0.45
MOTION_MINIMUM_FRAMES = 10

# Action when no motion
MOTION_FAILURE_ACTION = 'DENY'  # 'DENY' or 'WARN'

# ============================================
# CHALLENGE-RESPONSE LIVENESS
# ============================================

# Enable challenge-response (interactive)
ENABLE_CHALLENGE_RESPONSE = False  # Set True for maximum security (slower)

# Challenge settings
CHALLENGE_TIMEOUT = 5.0
CHALLENGE_TYPES = ['blink', 'smile']
REQUIRE_CHALLENGE_SUCCESS = True

# ============================================
# UNIFIED SECURITY SYSTEM
# ============================================

# Security level preset
SECURITY_LEVEL = 'MAXIMUM'  # 'MAXIMUM', 'HIGH', 'MEDIUM', 'LOW'

# Multi-tier decision
REQUIRE_ALL_TIERS = True  # All active tiers must pass

# Individual tier weights (if REQUIRE_ALL_TIERS = False)
TIER_WEIGHTS = {
    'advanced_liveness': 0.50,
    'motion': 0.35,
    'challenge': 0.15
}

# Overall threshold (weighted mode)
OVERALL_SECURITY_THRESHOLD = 0.75

# ============================================
# ANTI-SPOOFING MESSAGES
# ============================================

PHOTO_DETECTED_MESSAGE = "Photo/Screen detected - Access denied"
NO_MOTION_MESSAGE = "No movement detected - Access denied"
CHALLENGE_FAILED_MESSAGE = "Challenge failed - Access denied"
ANALYZING_MESSAGE = "Analyzing... Please stay still"

# ============================================
# CRITICAL CHECKS
# ============================================

# Number of critical checks that must pass
CRITICAL_CHECKS_MINIMUM = 4  # Out of 5 checks


# ============================================
# ADVANCED LIVENESS DETECTION (UPDATE THIS SECTION)
# ============================================

# Use advanced multi-layer liveness detector
USE_ADVANCED_LIVENESS = True

# Advanced liveness thresholds
ADVANCED_LIVENESS_THRESHOLD = 0.65

# Individual check thresholds (critical checks)
TEXTURE_THRESHOLD = 0.50
FREQUENCY_THRESHOLD = 0.55
COLOR_THRESHOLD = 0.60
SHARPNESS_THRESHOLD = 0.50

# Component weights (must sum to 1.0)
LIVENESS_WEIGHTS = {
    'texture': 0.35,
    'frequency': 0.30,
    'color': 0.05,
    'sharpness': 0.20,
    'variance': 0.10,
    'skin_tone': 0.00
}

# Decision settings
REQUIRE_CRITICAL_CHECKS = True
SHOW_LIVENESS_BREAKDOWN = True  # ← ADD THIS LINE
LIVENESS_FALLBACK_ENABLED = False

# Hard limits (instant fail if below)
LIVENESS_HARD_LIMITS = {
    'texture': 0.35,
    'frequency': 0.40,
    'sharpness': 0.40
}

# Critical checks minimum
CRITICAL_CHECKS_MINIMUM = 4
# ============================================
# MATCHING STRATEGY (UPDATED)
# ============================================

# Confidence thresholds
MIN_MATCH_CONFIDENCE = 0.25  # Lowered from 0.85 (more realistic)
# This works with intelligent decision engine
# Old value (0.85) was causing too many false rejections

# KNN matching
USE_KNN_MATCHING = True
KNN_K = 3  # Compare to 3 nearest neighbors

# Strict security
REJECT_UNKNOWN_FACES = True
REQUIRE_DATABASE_MATCH = True

# ============================================
# ENROLLMENT SETTINGS (ENHANCED)
# ============================================

ENROLLMENT_SAMPLES = 10  # Number of samples to capture
ENROLLMENT_QUALITY_THRESHOLD = 70  # Lowered from 80
CAPTURE_POSE_VARIATIONS = True
ENROLLMENT_ANGLES = [-15, -10, 0, 10, 15]
ENROLLMENT_MIN_SAMPLES = 5
ENROLLMENT_MAX_VARIANCE = 0.35  # Increased from 0.3 for more tolerance

# ============================================
# CAMERA SETTINGS
# ============================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# ============================================
# DISPLAY SETTINGS
# ============================================

BBOX_COLOR_LEGIT = (0, 255, 0)  # Green
BBOX_COLOR_UNKNOWN = (0, 0, 255)  # Red
BBOX_THICKNESS = 2
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_OFFSET_Y = -10

# ============================================
# ACCESS CONTROL SETTINGS
# ============================================

ACCESS_GRANTED_DISPLAY_TIME = 2
ACCESS_DENIED_DISPLAY_TIME = 3
ACCESS_RESULT_DISPLAY_TIME = 3
ACCESS_COOLDOWN = 3
ENABLE_AUDIO_FEEDBACK = False

# Access control messages
ACCESS_GRANTED_TEXT = "ACCESS GRANTED"
ACCESS_DENIED_TEXT = "ACCESS DENIED"
SYSTEM_READY_TEXT = "System Ready"
ACCESS_TEXT_FONT_SCALE = 2.0
ACCESS_TEXT_THICKNESS = 3
ACCESS_GRANTED_COLOR = (0, 255, 0)
ACCESS_DENIED_COLOR = (0, 0, 255)
SYSTEM_READY_COLOR = (255, 255, 255)

# ============================================
# ALERT SETTINGS
# ============================================

ALERT_MESSAGE_LEGIT = "Authorized User"
ALERT_MESSAGE_UNKNOWN = "Alert: Unknown Person"
SAVE_UNKNOWN_FACES = True
UNKNOWN_FACES_DIR = "unknown_faces"

# ============================================
# FILE PATHS
# ============================================

DATABASE_DIR = "database"
EMBEDDINGS_FILE = "embeddings.pkl"
LOG_FILE_PATH = "access_log.txt"
ACCESS_LOG_FILE = "access_log.txt"  # Alias for compatibility

# ============================================
# PERFORMANCE SETTINGS
# ============================================

AUTO_RECONNECT_CAMERA = True
MAX_RECONNECT_ATTEMPTS = 5
FRAME_SKIP = 1  # Process every frame

# ============================================
# TEMPORAL CONSISTENCY (DISABLED FOR ON-CLICK)
# ============================================

USE_TEMPORAL_CONSISTENCY = False  # Not suitable for on-click mode
TEMPORAL_BUFFER_SIZE = 5
MIN_CONSENSUS_RATIO = 0.6

# ============================================
# EMBEDDING PROCESSING
# ============================================

NORMALIZE_EMBEDDINGS = True  # L2 normalization
USE_EMBEDDING_WHITENING = False

# ============================================
# IMAGE PREPROCESSING
# ============================================

ENABLE_IMAGE_PREPROCESSING = True

# ============================================
# MULTI-FACTOR AUTHENTICATION (OPTIONAL)
# ============================================

ENABLE_MFA_FOR_LOW_CONFIDENCE = False  # Require PIN for borderline matches
MFA_THRESHOLD = 0.50  # Require MFA if score between 0.40-0.55

# ============================================
# SECURITY LEVELS (PRESETS)
# ============================================

# You can uncomment one of these to apply preset:

# # PRESET 1: Maximum Security
# def apply_maximum_security():
#     global DECISION_SCORE_THRESHOLD, MIN_MATCH_CONFIDENCE, ENABLE_SIMPLE_LIVENESS
#     global MASK_CONFIDENCE_THRESHOLD, OVERALL_QUALITY_THRESHOLD, ENABLE_ADAPTIVE_LIGHTING
#     DECISION_SCORE_THRESHOLD = 0.75
#     MIN_MATCH_CONFIDENCE = 0.45
#     ENABLE_SIMPLE_LIVENESS = True
#     MASK_CONFIDENCE_THRESHOLD = 0.85
#     OVERALL_QUALITY_THRESHOLD = 80
#     ENABLE_ADAPTIVE_LIGHTING = True

# # PRESET 2: Balanced (Current - Recommended)
# # Already configured above

# # PRESET 3: Lenient (Fast, user-friendly)
# def apply_lenient_mode():
#     global DECISION_SCORE_THRESHOLD, MIN_MATCH_CONFIDENCE, ENABLE_SIMPLE_LIVENESS
#     global MASK_CONFIDENCE_THRESHOLD, ENABLE_MASK_DETECTION, ENABLE_ADAPTIVE_LIGHTING
#     DECISION_SCORE_THRESHOLD = 0.50
#     MIN_MATCH_CONFIDENCE = 0.25
#     ENABLE_SIMPLE_LIVENESS = False
#     ENABLE_MASK_DETECTION = False
#     ENABLE_ADAPTIVE_LIGHTING = True

# ============================================
# CONFIDENCE BOOSTING
# ============================================

ENABLE_CONFIDENCE_BOOSTING = True
# Boosts confidence for high-quality matches with good liveness scores

# ============================================
# LOGGING & DEBUG
# ============================================

DEBUG_MASK_DETECTION = False  # Extra debug info for mask detection
LOG_ALL_ATTEMPTS = True  # Log both granted and denied
LOG_EMBEDDINGS = False  # Log embedding vectors (verbose)

# ============================================
# VALIDATION RULES
# ============================================

# Minimum thresholds (safety limits)
MIN_ALLOWED_DECISION_THRESHOLD = 0.40  # Lowered from 0.45
MIN_ALLOWED_MATCH_CONFIDENCE = 0.25
MAX_ALLOWED_RECOGNITION_THRESHOLD = 0.9

# ============================================
# PERFORMANCE MONITORING
# ============================================

ENABLE_PERFORMANCE_MONITORING = True  # Track processing times
PERFORMANCE_LOG_INTERVAL = 100  # Log every N frames

# ============================================
# BACKWARDS COMPATIBILITY
# ============================================

# Old config names mapped to new ones
FACENET_MODEL_PATH = None
USE_ADVANCED_FEATURES = True

# ============================================
# KEYBOARD CONTROLS (NEW)
# ============================================

# Enable keyboard shortcuts for real-time adjustments
ENABLE_KEYBOARD_CONTROLS = True

# Keyboard shortcuts:
# SPACE - Process frame
# Q - Quit
# + / = - Increase brightness
# - / _ - Decrease brightness
# A - Auto lighting mode
# B - Balance lighting mode
# N - No lighting adjustment
# L - Toggle liveness detection
# D - Toggle debug mode
# I - Show/hide brightness indicator

# ============================================
# SYSTEM INFO
# ============================================

CONFIG_VERSION = "2.1.0"  # Updated version
SYSTEM_NAME = "Enhanced Multi-Stage Face Recognition System with Adaptive Lighting"
LAST_UPDATED = "2026-02-02"

# ============================================
# VALIDATION (AUTO-RUN)
# ============================================

def validate_config():
    """Validate configuration on load"""
    issues = []
    
    # Check threshold ranges
    if not (0.0 <= DECISION_SCORE_THRESHOLD <= 1.0):
        issues.append(f"DECISION_SCORE_THRESHOLD must be 0-1, got {DECISION_SCORE_THRESHOLD}")
    
    if not (0.0 <= MIN_MATCH_CONFIDENCE <= 1.0):
        issues.append(f"MIN_MATCH_CONFIDENCE must be 0-1, got {MIN_MATCH_CONFIDENCE}")
    
    if DECISION_SCORE_THRESHOLD < MIN_ALLOWED_DECISION_THRESHOLD:
        issues.append(f"DECISION_SCORE_THRESHOLD too low (< {MIN_ALLOWED_DECISION_THRESHOLD})")
    
    # Check feature compatibility
    if USE_TEMPORAL_CONSISTENCY and FRAME_SKIP > 1:
        issues.append("USE_TEMPORAL_CONSISTENCY requires FRAME_SKIP = 1")
    
    # Check weights sum
    if USE_INTELLIGENT_DECISIONS:
        weight_sum = sum(DECISION_WEIGHTS.values())
        if not (0.99 <= weight_sum <= 1.01):
            issues.append(f"DECISION_WEIGHTS must sum to 1.0, got {weight_sum}")
    
    # Check lighting settings
    if ENABLE_ADAPTIVE_LIGHTING:
        if not (0 <= TARGET_BRIGHTNESS <= 255):
            issues.append(f"TARGET_BRIGHTNESS must be 0-255, got {TARGET_BRIGHTNESS}")
        
        if not (0.1 <= MIN_GAMMA <= 5.0):
            issues.append(f"MIN_GAMMA must be 0.1-5.0, got {MIN_GAMMA}")
        
        if not (0.1 <= MAX_GAMMA <= 5.0):
            issues.append(f"MAX_GAMMA must be 0.1-5.0, got {MAX_GAMMA}")
        
        if MIN_GAMMA >= MAX_GAMMA:
            issues.append(f"MIN_GAMMA must be < MAX_GAMMA")
    
    if issues:
        print("⚠️  CONFIG VALIDATION WARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration validated successfully")
    
    return len(issues) == 0

# Auto-validate on import
if __name__ != "__main__":
    validate_config()