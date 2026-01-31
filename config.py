"""
Configuration settings for the Real-Time Facial Recognition Alert System
"""

# Face Detection Settings
FACE_DETECTION_CONFIDENCE = 0.9  # Minimum confidence for face detection (0-1)
MIN_FACE_SIZE = 40  # Minimum face size in pixels

# Face Recognition Settings
RECOGNITION_THRESHOLD = 0.6  # Maximum distance for a match (lower = stricter)
EMBEDDING_SIZE = 128  # Size of face embeddings

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
