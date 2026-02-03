# Configuration Reference

Complete configuration guide for the Real-Time Facial Recognition Alert System.

---

## Table of Contents

- [Configuration File](#configuration-file)
- [Security Presets](#security-presets)
- [Core Parameters](#core-parameters)
- [Face Detection Settings](#face-detection-settings)
- [Recognition Settings](#recognition-settings)
- [Liveness Detection](#liveness-detection)
- [Quality Thresholds](#quality-thresholds)
- [Performance Tuning](#performance-tuning)
- [Advanced Settings](#advanced-settings)

---

## Configuration File

Configuration is stored in `config/config.py`. All settings are Python variables that can be modified directly.

### Location
```
config/config.py
```

### Loading Configuration

```python
from config import config

# Access settings
threshold = config.RECOGNITION_THRESHOLD
camera = config.CAMERA_INDEX
```

---

## Security Presets

Choose from three predefined security levels:

### Maximum Security

**Best for**: High-security areas, sensitive locations, critical infrastructure

```python
# In config/config.py
USE_INSIGHTFACE = True
ENABLE_QUALITY_CHECKS = True
ENABLE_FACE_ALIGNMENT = True
LIVENESS_ENABLED = True
LIVENESS_METHOD = 'combined'
MIN_MATCH_CONFIDENCE = 0.85
REQUIRE_BLINK = True
LIVENESS_FRAMES_REQUIRED = 30
```

**Characteristics:**
- ✅ Maximum security
- ✅ Best anti-spoofing
- ⚠️ May have higher false reject rate
- ⚠️ Slower processing

### Balanced Performance (Recommended)

**Best for**: Office buildings, residential access, general use

```python
USE_INSIGHTFACE = True
ENABLE_QUALITY_CHECKS = True
ENABLE_FACE_ALIGNMENT = True
LIVENESS_ENABLED = False  # or 'texture' for basic
MIN_MATCH_CONFIDENCE = 0.75
REQUIRE_BLINK = False
```

**Characteristics:**
- ✅ Good balance of security and usability
- ✅ Reasonable speed
- ✅ Low false reject rate
- ✅ Good user experience

### Maximum Speed

**Best for**: Low-security areas, testing, demonstrations

```python
USE_INSIGHTFACE = False
ENABLE_QUALITY_CHECKS = True
ENABLE_FACE_ALIGNMENT = False
LIVENESS_ENABLED = False
MIN_MATCH_CONFIDENCE = 0.70
DETECTION_CONFIDENCE = 0.8
```

**Characteristics:**
- ✅ Fastest processing
- ✅ Lowest resource usage
- ⚠️ Lower security
- ⚠️ Vulnerable to photo attacks

---

## Core Parameters

### Camera Settings

```python
# Camera device index (0 = default camera)
CAMERA_INDEX = 0

# Frame resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Processing frame rate (process every Nth frame)
PROCESS_EVERY_N_FRAMES = 1

# Camera reconnection
AUTO_RECONNECT_CAMERA = True
CAMERA_RECONNECT_DELAY = 5  # seconds
```

**Usage:**
- Set `CAMERA_INDEX = 1` for external webcam
- Use `"rtsp://..."` for IP cameras
- Lower resolution for better performance
- Increase `PROCESS_EVERY_N_FRAMES` to 2-3 for slow systems

### Display Settings

```python
# Display window name
WINDOW_NAME = "Face Recognition System"

# Display sizes
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Font settings
FONT_SCALE = 1.0
FONT_THICKNESS = 2

# Colors (BGR format)
ACCESS_GRANTED_COLOR = (0, 255, 0)  # Green
ACCESS_DENIED_COLOR = (0, 0, 255)   # Red
SYSTEM_READY_COLOR = (255, 255, 255)  # White
```

### Access Control

```python
# Cooldown between access attempts (seconds)
ACCESS_COOLDOWN = 3

# Display duration for messages (seconds)
ACCESS_GRANTED_DISPLAY_TIME = 3
ACCESS_DENIED_DISPLAY_TIME = 2

# Access messages
ACCESS_GRANTED_TEXT = "ACCESS GRANTED"
ACCESS_DENIED_TEXT = "ACCESS DENIED"
SYSTEM_READY_TEXT = "SYSTEM READY"
```

---

## Face Detection Settings

### MTCNN Configuration

```python
# Minimum face size to detect (pixels)
MIN_FACE_SIZE = 40

# Detection confidence threshold (0.0-1.0)
DETECTION_CONFIDENCE = 0.9

# Face detection scale factors
SCALE_FACTOR = 0.709
```

**Tuning Guide:**

| Setting | Lower Value | Higher Value |
|---------|-------------|--------------|
| `MIN_FACE_SIZE` | Detect smaller faces | Faster, ignore distant faces |
| `DETECTION_CONFIDENCE` | More detections (false positives) | Fewer false detections |
| `SCALE_FACTOR` | Slower, more accurate | Faster, may miss faces |

**Recommended Values:**
- **Close-range**: `MIN_FACE_SIZE = 60`, `DETECTION_CONFIDENCE = 0.95`
- **Medium-range**: `MIN_FACE_SIZE = 40`, `DETECTION_CONFIDENCE = 0.9`
- **Long-range**: `MIN_FACE_SIZE = 20`, `DETECTION_CONFIDENCE = 0.85`

---

## Recognition Settings

### Model Selection

```python
# Use InsightFace (more accurate) or FaceNet
USE_INSIGHTFACE = True

# Model paths
FACENET_MODEL = 'facenet'
INSIGHTFACE_MODEL = 'buffalo_l'
```

**Model Comparison:**

| Feature | FaceNet | InsightFace |
|---------|---------|-------------|
| Accuracy | Good | Excellent |
| Speed | Fast | Moderate |
| GPU Usage | Low | Moderate |
| Memory | ~100MB | ~500MB |

### Recognition Thresholds

```python
# Face matching threshold
RECOGNITION_THRESHOLD = 0.6  # For InsightFace/ArcFace (cosine)
# or
RECOGNITION_THRESHOLD = 1.0  # For FaceNet (Euclidean)

# Minimum confidence for match
MIN_MATCH_CONFIDENCE = 0.75
```

**Threshold Tuning:**

For **InsightFace/ArcFace** (cosine similarity):
- `0.4` - Very strict (many false rejects)
- `0.5` - Strict (recommended for high security)
- `0.6` - Balanced (recommended default)
- `0.7` - Lenient (some false accepts)
- `0.8` - Very lenient (many false accepts)

For **FaceNet** (Euclidean distance):
- `0.8` - Very strict
- `1.0` - Strict (recommended for high security)
- `1.2` - Balanced (recommended default)
- `1.4` - Lenient
- `1.6` - Very lenient

---

## Liveness Detection

### Enable/Disable

```python
# Enable liveness detection
LIVENESS_ENABLED = True

# Method: 'texture', 'motion', 'blink', 'combined'
LIVENESS_METHOD = 'combined'
```

**Methods:**

- **`texture`** - Fast, good for photos
- **`motion`** - Requires multiple frames
- **`blink`** - Most accurate, requires user interaction
- **`combined`** - All methods (slowest, most secure)

### Liveness Thresholds

```python
# Overall liveness threshold (0.0-1.0)
LIVENESS_THRESHOLD = 0.7

# Individual component thresholds
TEXTURE_ANALYSIS_THRESHOLD = 0.7
FREQUENCY_THRESHOLD = 0.6
COLOR_NATURALNESS_THRESHOLD = 0.65
SHARPNESS_THRESHOLD = 0.7
VARIANCE_THRESHOLD = 0.5
```

**Tuning:**

| Threshold | Lower (0.5-0.6) | Medium (0.7) | Higher (0.8-0.9) |
|-----------|-----------------|--------------|-------------------|
| Security | Less secure | Balanced | Most secure |
| Usability | Easy to pass | Moderate | Strict |
| False Rejects | Few | Some | Many |

### Blink Detection

```python
# Require blink for access
REQUIRE_BLINK = True

# Blink timeout (seconds)
BLINK_TIMEOUT = 5

# Number of frames to analyze
LIVENESS_FRAMES_REQUIRED = 15
```

---

## Quality Thresholds

### Image Quality Checks

```python
# Enable quality validation
ENABLE_QUALITY_CHECKS = True

# Blur threshold (higher = sharper required)
QUALITY_BLUR_THRESHOLD = 100

# Brightness range (0-255)
QUALITY_BRIGHTNESS_MIN = 40
QUALITY_BRIGHTNESS_MAX = 220

# Minimum contrast
QUALITY_MIN_CONTRAST = 30

# Minimum face size after detection
QUALITY_MIN_FACE_SIZE = 80
```

**Quality Tuning:**

| Parameter | Low | Medium | High |
|-----------|-----|--------|------|
| Blur Threshold | 50 | 100 | 150 |
| Brightness Min | 30 | 40 | 50 |
| Brightness Max | 230 | 220 | 200 |
| Contrast | 20 | 30 | 50 |

### Face Alignment

```python
# Enable face alignment (normalization)
ENABLE_FACE_ALIGNMENT = True

# Aligned face size
ALIGNED_FACE_SIZE = 160
```

---

## Performance Tuning

### Processing Options

```python
# Multi-threading
USE_MULTITHREADING = True
NUM_WORKER_THREADS = 2

# Frame buffering
FRAME_BUFFER_SIZE = 10

# Skip frames if queue is full
SKIP_FRAMES_IF_BUSY = True
```

### Memory Management

```python
# Maximum embeddings cache size
MAX_EMBEDDINGS_CACHE = 100

# Clear cache interval (frames)
CACHE_CLEAR_INTERVAL = 1000

# Database update frequency
DATABASE_SAVE_INTERVAL = 10  # seconds
```

### GPU Settings

```python
# Use GPU acceleration
USE_GPU = True

# GPU memory growth (for TensorFlow)
GPU_MEMORY_GROWTH = True

# GPU memory fraction (0.0-1.0)
GPU_MEMORY_FRACTION = 0.5
```

---

## Advanced Settings

### Database Settings

```python
# Database file path
DATABASE_PATH = "database/embeddings.pkl"

# Backup settings
AUTO_BACKUP_DATABASE = True
BACKUP_INTERVAL = 3600  # seconds (1 hour)
MAX_BACKUPS = 10
```

### Logging Configuration

```python
# Log file path
LOG_FILE = "logs/access_log.txt"

# Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# Log rotation
LOG_ROTATION = True
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
```

### Unknown Face Handling

```python
# Save images of unknown faces
SAVE_UNKNOWN_FACES = True

# Unknown faces directory
UNKNOWN_FACES_DIR = "unknown_faces/"

# Maximum unknown faces to save per day
MAX_UNKNOWN_FACES_PER_DAY = 100
```

### Anti-Spoofing Advanced

```python
# Occlusion detection
ENABLE_OCCLUSION_DETECTION = True
OCCLUSION_THRESHOLD = 0.3

# Eye state detection
ENABLE_EYE_DETECTION = True
EYE_ASPECT_RATIO_THRESHOLD = 0.2

# Depth analysis (if depth camera available)
ENABLE_DEPTH_ANALYSIS = False
DEPTH_THRESHOLD = 0.1
```

---

## Configuration Examples

### Example 1: Office Building

```python
# config/config.py
# Office building with moderate security

CAMERA_INDEX = 0
USE_INSIGHTFACE = True
ENABLE_QUALITY_CHECKS = True
ENABLE_FACE_ALIGNMENT = True

LIVENESS_ENABLED = False  # Quick access for employees
MIN_MATCH_CONFIDENCE = 0.75
RECOGNITION_THRESHOLD = 0.6

ACCESS_COOLDOWN = 2
SAVE_UNKNOWN_FACES = True
```

### Example 2: Data Center (High Security)

```python
# config/config.py
# Data center with maximum security

CAMERA_INDEX = 0
USE_INSIGHTFACE = True
ENABLE_QUALITY_CHECKS = True
ENABLE_FACE_ALIGNMENT = True

LIVENESS_ENABLED = True
LIVENESS_METHOD = 'combined'
REQUIRE_BLINK = True
MIN_MATCH_CONFIDENCE = 0.85
RECOGNITION_THRESHOLD = 0.5

ACCESS_COOLDOWN = 5
SAVE_UNKNOWN_FACES = True
ENABLE_OCCLUSION_DETECTION = True
```

### Example 3: Testing/Demo

```python
# config/config.py
# Testing with fast performance

CAMERA_INDEX = 0
USE_INSIGHTFACE = False  # Faster FaceNet
ENABLE_QUALITY_CHECKS = True
ENABLE_FACE_ALIGNMENT = False

LIVENESS_ENABLED = False
MIN_MATCH_CONFIDENCE = 0.70
RECOGNITION_THRESHOLD = 0.7

PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
FRAME_WIDTH = 480  # Lower resolution
FRAME_HEIGHT = 360
```

---

## Troubleshooting Configuration

### Too Many False Rejects

**Problem**: Authorized users are being denied

**Solutions**:
```python
# Increase thresholds
RECOGNITION_THRESHOLD = 0.7  # was 0.6
MIN_MATCH_CONFIDENCE = 0.7   # was 0.75

# Disable strict features
LIVENESS_ENABLED = False
REQUIRE_BLINK = False

# Lower quality requirements
QUALITY_BLUR_THRESHOLD = 80   # was 100
```

### Too Many False Accepts

**Problem**: Unknown people are being granted access

**Solutions**:
```python
# Decrease thresholds
RECOGNITION_THRESHOLD = 0.5  # was 0.6
MIN_MATCH_CONFIDENCE = 0.85  # was 0.75

# Enable strict features
LIVENESS_ENABLED = True
LIVENESS_METHOD = 'combined'
REQUIRE_BLINK = True

# Stricter quality
QUALITY_BLUR_THRESHOLD = 120  # was 100
ENABLE_QUALITY_CHECKS = True
```

### Poor Performance

**Problem**: System is slow or laggy

**Solutions**:
```python
# Reduce processing load
PROCESS_EVERY_N_FRAMES = 3  # was 1
FRAME_WIDTH = 480            # was 640
FRAME_HEIGHT = 360           # was 480

# Disable expensive features
USE_INSIGHTFACE = False
ENABLE_FACE_ALIGNMENT = False
LIVENESS_ENABLED = False

# Enable GPU
USE_GPU = True
```

---

## Configuration Best Practices

1. **Start with presets** - Use balanced preset and adjust as needed
2. **Test thoroughly** - Test after each configuration change
3. **Document changes** - Keep notes on why you changed settings
4. **Backup config** - Save working configurations
5. **Monitor metrics** - Track false accept/reject rates
6. **Adjust gradually** - Make small incremental changes
7. **Use version control** - Track configuration changes in git

---

## See Also

- [README.md](../README.md) - Overview and features
- [INSTALLATION.md](INSTALLATION.md) - Installation guide
- [USAGE.md](USAGE.md) - Usage instructions
- [API.md](API.md) - API reference
