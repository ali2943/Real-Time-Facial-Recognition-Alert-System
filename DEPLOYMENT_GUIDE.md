# Deployment Guide: Real-Time Facial Recognition System

## Table of Contents
1. [System Overview](#system-overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [User Enrollment](#user-enrollment)
6. [Running the System](#running-the-system)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)
9. [Security Considerations](#security-considerations)

---

## System Overview

This is a real-time facial recognition system designed for:
- **Access control** (door entry, computer login)
- **Attendance tracking**
- **Security monitoring**
- **Demo/educational purposes**

**Key Features:**
- Deep learning-based face detection (MTCNN or RetinaFace)
- Embedding-based recognition (FaceNet or InsightFace/ArcFace)
- Liveness detection (anti-spoofing)
- Real-time processing (15-30 FPS on modern laptops)
- Modular, well-documented codebase

**Comparable to:** Mobile Face Unlock (iPhone Face ID, Android Face Unlock)  
**Limitations:** 2D RGB camera only (no IR, no depth sensor)

---

## Hardware Requirements

### Minimum Requirements
- **CPU:** Dual-core 2.0 GHz or better
- **RAM:** 4 GB
- **Camera:** 720p webcam (1280x720)
- **OS:** Windows 10, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Storage:** 2 GB free space

### Recommended Requirements
- **CPU:** Quad-core 2.5 GHz or better (Intel i5/i7, AMD Ryzen)
- **RAM:** 8 GB
- **Camera:** 1080p webcam (1920x1080) with good low-light performance
- **GPU:** NVIDIA GPU with CUDA support (optional, 2-3x speedup)
- **OS:** Windows 11, macOS 12+, or Ubuntu 20.04+
- **Storage:** 5 GB free space

### Camera Recommendations
**Good:**
- Logitech C920/C922
- Microsoft LifeCam HD-3000
- Any laptop built-in camera (2020+)

**Better:**
- Logitech Brio 4K
- Razer Kiyo Pro
- Cameras with HDR/auto-light adjustment

**Critical:** Ensure good lighting! No camera can compensate for poor lighting.

---

## Installation

### Step 1: Install Python

**Minimum:** Python 3.8  
**Recommended:** Python 3.10 or 3.11

**Check your Python version:**
```bash
python --version
# or
python3 --version
```

**Install Python if needed:**
- **Windows:** Download from [python.org](https://www.python.org/downloads/)
- **macOS:** `brew install python@3.11` (if you have Homebrew)
- **Linux:** `sudo apt install python3.11 python3.11-venv`

### Step 2: Clone Repository

```bash
git clone https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System.git
cd Real-Time-Facial-Recognition-Alert-System
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

**Standard Installation (FaceNet only):**
```bash
pip install -r requirements.txt
```

**Full Installation (with InsightFace for best accuracy):**
```bash
pip install -r requirements.txt
pip install insightface onnxruntime
```

**GPU Acceleration (optional, NVIDIA GPU only):**
```bash
pip install onnxruntime-gpu
# Requires NVIDIA CUDA Toolkit 11.x
```

### Step 5: Verify Installation

```bash
# Test imports
python -c "import cv2; import tensorflow; import mtcnn; print('✓ Core dependencies OK')"

# Test InsightFace (optional)
python -c "import insightface; print('✓ InsightFace OK')"
```

**Expected Output:**
```
✓ Core dependencies OK
✓ InsightFace OK  # if InsightFace installed
```

---

## Configuration

### Quick Start Presets

Edit `config.py` and choose a preset:

#### Preset 1: Maximum Accuracy (Recommended for Production)
```python
# Model & Features
USE_INSIGHTFACE = True
INSIGHTFACE_MODEL = 'buffalo_l'
ENABLE_QUALITY_CHECKS = True
ENABLE_FACE_ALIGNMENT = True
LIVENESS_ENABLED = True
LIVENESS_METHOD = 'combined'

# Thresholds (strict)
RECOGNITION_THRESHOLD = 0.5  # Auto-adjusted for InsightFace
MIN_MATCH_CONFIDENCE = 0.85
OVERALL_QUALITY_THRESHOLD = 80
```

**Performance:** ~10-15 FPS on modern laptop  
**Security:** High (defeats most spoofing attacks)  
**UX:** May reject valid users in poor lighting

#### Preset 2: Balanced (Recommended for Most Use Cases)
```python
# Model & Features
USE_INSIGHTFACE = True
INSIGHTFACE_MODEL = 'buffalo_s'  # Smaller, faster model
ENABLE_QUALITY_CHECKS = True
ENABLE_FACE_ALIGNMENT = True
LIVENESS_ENABLED = False  # Disable for speed

# Thresholds (moderate)
RECOGNITION_THRESHOLD = 0.6  # Auto-adjusted
MIN_MATCH_CONFIDENCE = 0.75
OVERALL_QUALITY_THRESHOLD = 75
```

**Performance:** ~20-25 FPS  
**Security:** Medium (good for supervised access control)  
**UX:** Good balance

#### Preset 3: Maximum Speed (Demo/Low-Security)
```python
# Model & Features
USE_INSIGHTFACE = False  # Use FaceNet instead
ENABLE_QUALITY_CHECKS = True
ENABLE_FACE_ALIGNMENT = False
LIVENESS_ENABLED = False

# Thresholds (lenient)
RECOGNITION_THRESHOLD = 1.2  # FaceNet scale
MIN_MATCH_CONFIDENCE = 0.70
OVERALL_QUALITY_THRESHOLD = 70
```

**Performance:** ~30 FPS  
**Security:** Basic (supervised use only)  
**UX:** Very smooth, rarely rejects

### Advanced Tuning

#### Recognition Threshold
Controls how strict face matching is.

**FaceNet (USE_INSIGHTFACE = False):**
- `0.8-1.0`: Very strict (high security, may reject valid users)
- `1.0-1.2`: **Recommended** (balanced)
- `1.2-1.4`: Lenient (convenience, higher false accept rate)

**InsightFace (USE_INSIGHTFACE = True):**
- `0.4-0.5`: Very strict
- `0.5-0.6`: **Recommended** (auto-adjusted by system)
- `0.6-0.7`: Lenient

**How to tune:**
1. Enroll yourself
2. Run system with `DEBUG_MODE = True`
3. Watch console output for distance values
4. Adjust threshold so:
   - Your face: distance < threshold (recognized)
   - Other faces: distance > threshold (rejected)

#### Quality Thresholds

**BLUR_THRESHOLD:**
- Lower quality camera: `80`
- Standard webcam: `100` (default)
- High quality camera: `120`

**BRIGHTNESS_RANGE:**
- Dark environment: `(30, 200)`
- Normal lighting: `(40, 220)` (default)
- Bright/outdoor: `(50, 240)`

---

## User Enrollment

### Basic Enrollment

```bash
python enroll_user.py --name "John Doe" --samples 10
```

**Process:**
1. Look at camera
2. Press **SPACE** to capture each sample
3. System shows quality score (aim for 80+)
4. Capture 10 samples at different:
   - Angles (slightly left, right, up, down)
   - Distances (closer, farther)
   - Expressions (neutral, smile)

**Tips for Best Results:**
- Good lighting (face well-lit, no harsh shadows)
- Neutral background (avoid clutter)
- Remove glasses/hat if not worn daily
- Capture some samples with glasses if worn sometimes

### View Enrolled Users

```bash
python list_users.py
```

### Remove a User

```bash
python remove_user.py --name "John Doe"
```

---

## Running the System

### Basic Usage

```bash
python main.py
```

**Controls:**
- **q**: Quit system
- System displays:
  - **ACCESS GRANTED** (green) for authorized users
  - **ACCESS DENIED** (red) for unknown persons
  - Real-time FPS and uptime

### Advanced Options

```bash
# Use specific camera (if multiple cameras)
python main.py --camera 1

# Enable debug output
# (edit config.py: DEBUG_MODE = True)
```

### What Happens During Recognition

**Normal Flow:**
1. Camera captures frame
2. MTCNN detects faces
3. Quality check (blur, brightness, contrast)
4. Face alignment (if enabled)
5. Generate embedding (FaceNet/InsightFace)
6. Compare with database
7. Liveness check (if enabled)
8. Display result (GRANTED/DENIED)

**Debug Output (DEBUG_MODE = True):**
```
[DEBUG] MTCNN raw detections: 1
[DEBUG] Valid face detected: conf=0.987, size=112x135px
[DEBUG] Face extracted: size=152x175px
[DEBUG] Face quality score: 87.5/100
[DEBUG] Searching database for match...
[DEBUG] Best match: John Doe, Distance: 0.42, Confidence: 89.3%
[SUCCESS] Access Granted: John Doe
```

---

## Troubleshooting

### No Faces Detected

**Symptoms:** System shows "System Ready" but doesn't detect your face

**Solutions:**
1. **Check lighting:** Face should be well-lit
2. **Check distance:** 1-3 feet from camera is optimal
3. **Lower detection threshold:**
   ```python
   FACE_DETECTION_CONFIDENCE = 0.5  # Lower from 0.7
   ```
4. **Check camera permissions:** Ensure camera access allowed
5. **Test camera:**
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   cv2.imshow('Test', frame)
   cv2.waitKey(0)
   ```

### False Rejections (Not Recognizing Enrolled Users)

**Symptoms:** System denies access to enrolled users

**Solutions:**
1. **Check enrollment quality:**
   - Re-enroll with 15-20 samples
   - Ensure good lighting during enrollment
   - Capture varied angles

2. **Increase threshold:**
   ```python
   # For FaceNet:
   RECOGNITION_THRESHOLD = 1.2  # Increase from 1.0
   
   # For InsightFace:
   RECOGNITION_THRESHOLD = 0.65  # Increase from 0.6
   
   MIN_MATCH_CONFIDENCE = 0.70  # Decrease from 0.75
   ```

3. **Check quality scores:**
   - If enrollment quality < 80, lighting may be poor
   - If runtime quality < 75, face may be blurry/dark

4. **Disable strict checks temporarily:**
   ```python
   ENABLE_QUALITY_CHECKS = False  # Test if quality is the issue
   ```

### False Accepts (Recognizing Wrong Person)

**Symptoms:** System grants access to unauthorized persons

**Solutions:**
1. **Decrease threshold (stricter):**
   ```python
   # For FaceNet:
   RECOGNITION_THRESHOLD = 0.9  # Decrease from 1.0
   
   # For InsightFace:
   RECOGNITION_THRESHOLD = 0.5  # Decrease from 0.6
   
   MIN_MATCH_CONFIDENCE = 0.85  # Increase from 0.75
   ```

2. **Enable liveness detection:**
   ```python
   LIVENESS_ENABLED = True
   LIVENESS_METHOD = 'combined'
   ```

3. **Increase enrollment samples:**
   - Re-enroll with 20-30 samples
   - Cover more variations

4. **Use InsightFace:**
   ```python
   USE_INSIGHTFACE = True
   INSIGHTFACE_MODEL = 'buffalo_l'
   ```

### Poor Performance (Low FPS)

**Symptoms:** System is slow, laggy, < 10 FPS

**Solutions:**
1. **Enable frame skipping:**
   ```python
   FRAME_SKIP = 2  # Process every 2nd frame
   ```

2. **Use smaller model:**
   ```python
   INSIGHTFACE_MODEL = 'buffalo_s'  # Instead of buffalo_l
   # OR
   USE_INSIGHTFACE = False  # Use FaceNet (faster)
   ```

3. **Disable expensive features:**
   ```python
   LIVENESS_ENABLED = False
   ENABLE_QUALITY_CHECKS = False  # Last resort
   ```

4. **Lower camera resolution:**
   ```python
   FRAME_WIDTH = 640  # Down from 1280
   FRAME_HEIGHT = 480  # Down from 720
   ```

5. **Enable GPU (if available):**
   ```python
   GPU_ENABLED = True
   ```

### Liveness Check Failures

**Symptoms:** Real users rejected as "spoofing attempts"

**Solutions:**
1. **Lower texture threshold:**
   ```python
   TEXTURE_ANALYSIS_THRESHOLD = 0.6  # Down from 0.7
   ```

2. **Use gentler liveness method:**
   ```python
   LIVENESS_METHOD = 'motion'  # Instead of 'combined'
   ```

3. **Check lighting:**
   - Very low/high light affects texture analysis
   - Ensure even lighting on face

4. **Disable if not needed:**
   ```python
   LIVENESS_ENABLED = False  # For supervised access control
   ```

---

## Performance Tuning

### Benchmarking Your System

```bash
# Run with DEBUG_MODE = True
python main.py

# Watch console for:
# - FPS (top right of video)
# - Processing time per frame
# - Detection confidence
# - Recognition distances
```

**Target Performance:**
- **Excellent:** 25-30 FPS (smooth, real-time)
- **Good:** 15-25 FPS (acceptable for access control)
- **Acceptable:** 10-15 FPS (usable but may feel sluggish)
- **Poor:** < 10 FPS (consider optimization)

### Optimization Priority

**If FPS < 10:**
1. Enable `FRAME_SKIP = 2`
2. Switch to `USE_INSIGHTFACE = False`
3. Disable `LIVENESS_ENABLED`
4. Lower camera resolution

**If FPS 10-15:**
1. Switch to `INSIGHTFACE_MODEL = 'buffalo_s'`
2. Set `FRAME_SKIP = 2` only if needed
3. Keep quality checks enabled

**If FPS > 20:**
- You have headroom for more features
- Enable `LIVENESS_ENABLED = True`
- Use `LIVENESS_METHOD = 'combined'`
- Increase enrollment samples to 20-30

---

## Security Considerations

### Threat Model

**What This System CAN Defend Against:**
- ✓ Static photos (printed or on screen)
- ✓ Low-quality video replays
- ✓ Different person trying to gain access
- ✓ Poorly-lit attempts to obscure identity

**What This System CANNOT Defend Against:**
- ✗ High-quality 3D masks (would need depth sensor)
- ✗ Identical twins (would need IR or other biometrics)
- ✗ Sophisticated video replay with motion (need challenge-response)
- ✗ Surgical face changes

### Security Levels

**Level 1: Basic (Supervised Access)**
```python
USE_INSIGHTFACE = False
LIVENESS_ENABLED = False
MIN_MATCH_CONFIDENCE = 0.70
```
Use for: Demo, low-security office access with human supervision

**Level 2: Standard (Typical Access Control)**
```python
USE_INSIGHTFACE = True
LIVENESS_ENABLED = False
MIN_MATCH_CONFIDENCE = 0.75
```
Use for: Office building access, time tracking, supervised kiosks

**Level 3: High Security (Unattended Systems)**
```python
USE_INSIGHTFACE = True
LIVENESS_ENABLED = True
LIVENESS_METHOD = 'combined'
MIN_MATCH_CONFIDENCE = 0.85
REQUIRE_BLINK = True
```
Use for: Secure facility access, financial systems, healthcare

**Level 4: Maximum Security (Challenge-Response)**
```python
# All Level 3 settings, plus:
# Implement challenge-response in main.py:
# liveness_detector.start_challenge()
# Prompt user: "Please blink twice" or "Turn head left"
```
Use for: Military, critical infrastructure, government

### Best Practices

1. **Enrollment:**
   - Require in-person enrollment (don't accept emailed photos)
   - Verify identity before enrollment (ID check)
   - Capture 20+ samples in varied conditions
   - Re-enroll if person's appearance changes significantly

2. **Deployment:**
   - Place camera at eye level, 2-3 feet from user
   - Ensure good, even lighting (avoid backlighting)
   - Use tamper-evident camera mounts
   - Log all access attempts with photos

3. **Monitoring:**
   - Review unknown face captures regularly
   - Monitor for repeated failed attempts (potential attack)
   - Track false reject rate (user frustration indicator)
   - Audit access logs weekly

4. **Updates:**
   - Keep dependencies updated (security patches)
   - Re-train database quarterly (appearance changes)
   - Monitor for new attack methods in research

---

## Next Steps

- **For Developers:** See [ARCHITECTURE.md](ARCHITECTURE.md) for code structure
- **For Advanced Users:** See [ACCURACY_ENHANCEMENTS.md](ACCURACY_ENHANCEMENTS.md) for fine-tuning
- **For Troubleshooting:** Enable `DEBUG_MODE = True` and check console output

**Questions or Issues?**
- Open an issue on GitHub
- Check existing issues for solutions
- Contribute improvements via pull requests

---

*Last Updated: January 2026*  
*System Version: 2.0*  
*Compatible with: Python 3.8-3.11*
