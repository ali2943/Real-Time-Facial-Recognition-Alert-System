"""
Configuration settings for the Real-Time Facial Recognition Alert System
"""

# Face Detection Settings
FACE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for face detection (0-1)
MIN_FACE_SIZE = 20  # Minimum face size in pixels

# Face Recognition Settings
RECOGNITION_THRESHOLD = 0.6  # Maximum distance for a match (lower = stricter)
EMBEDDING_SIZE = 128  # Size of face embeddings
DEBUG_MODE = True  # Print detailed debug information (distance values, matching process)

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
