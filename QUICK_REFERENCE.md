# Security Door Access Control System - Quick Reference Guide

## 🚪 Overview
This system operates as a continuous 24/7 security door access control system using facial recognition to grant or deny access.

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Enroll Authorized Users
```bash
python enroll_user.py --name "John Doe" --samples 5
```

### 3. Run the System
```bash
python main.py
```

The system will now run continuously until you press 'q' to quit.

## 🎯 System Behavior

### When an Authorized Person Approaches:
1. Face is detected and recognized
2. Large **"ACCESS GRANTED"** message appears in GREEN for 2 seconds
3. Person's name is displayed below the message
4. Console prints: `[SUCCESS] Access Granted: John Doe`
5. Event logged to `access_log.txt`: `[2026-01-31 14:30:45] ACCESS GRANTED - John Doe`
6. System returns to "System Ready" state after 2 seconds
7. 3-second cooldown before next access attempt

### When an Unauthorized Person Approaches:
1. Face is detected but not recognized
2. Large **"ACCESS DENIED"** message appears in RED for 3 seconds
3. "Unknown Person" warning displayed below
4. Console prints: `[FAILURE] Access Denied: Unknown Person`
5. Photo saved to `unknown_faces/unknown_YYYYMMDD_HHMMSS_N.jpg`
6. Event logged to `access_log.txt`: `[2026-01-31 14:31:12] ACCESS DENIED - Unknown (Photo: unknown_20260131_143112_0.jpg)`
7. System returns to "System Ready" state after 3 seconds
8. 3-second cooldown before next access attempt

### When No One is Present:
- Displays "System Ready" indicator
- Shows FPS (frames per second)
- Shows system uptime (HH:MM:SS)
- Shows last access event

## 📝 Access Log Format

The system maintains a log file at `access_log.txt`:

```
[2026-01-31 14:30:45] ACCESS GRANTED - John Doe
[2026-01-31 14:31:12] ACCESS DENIED - Unknown (Photo: unknown_20260131_143112_0.jpg)
[2026-01-31 14:32:05] ACCESS GRANTED - Jane Smith
[2026-01-31 14:33:22] ACCESS DENIED - Unknown (Photo: unknown_20260131_143322_1.jpg)
```

## ⚙️ Configuration Options

Edit `config.py` to customize:

```python
# Access Control Settings
ACCESS_GRANTED_DISPLAY_TIME = 2  # Seconds to show success message
ACCESS_DENIED_DISPLAY_TIME = 3   # Seconds to show failure message
ACCESS_COOLDOWN = 3              # Seconds between access attempts

# Camera Settings
AUTO_RECONNECT_CAMERA = True     # Auto-reconnect on camera failure
MAX_RECONNECT_ATTEMPTS = 5       # Maximum reconnection attempts

# Detection Settings
FACE_DETECTION_CONFIDENCE = 0.5  # Lower = more sensitive
MIN_FACE_SIZE = 20               # Minimum face size in pixels
FRAME_SKIP = 1                   # Process every Nth frame (1 = all frames)

# Logging
LOG_FILE_PATH = "access_log.txt"
SAVE_UNKNOWN_FACES = True
```

## 🔧 Troubleshooting

### System Not Detecting Faces
- Ensure good lighting
- Check camera is working
- Lower `FACE_DETECTION_CONFIDENCE` (already set to 0.5)
- Reduce `MIN_FACE_SIZE` (already set to 20)

### Camera Disconnection
- System auto-reconnects by default
- Will try up to 5 times (configurable)
- Check camera connection and drivers

### False Positives/Negatives
- Adjust `RECOGNITION_THRESHOLD` in config.py
- Lower = stricter (fewer false positives)
- Higher = looser (fewer false negatives)
- Enroll users with multiple samples from different angles

### Access Log Not Created
- Log file created automatically on first access event
- Check write permissions
- Configure custom path via `LOG_FILE_PATH`

## 🛡️ Security Features

1. **Continuous Operation**: Never stops, auto-recovers from errors
2. **Access Logging**: All events logged with timestamps
3. **Unknown Face Capture**: Photos of unauthorized persons saved
4. **Cooldown Protection**: Prevents spam with 3-second cooldown
5. **Error Recovery**: Handles all errors gracefully without crashing
6. **Camera Auto-Reconnect**: Automatic reconnection on failure

## 📊 System Monitoring

The system displays:
- **FPS**: Current frames per second
- **Uptime**: Total system running time (HH:MM:SS)
- **Last Event**: Most recent access event (granted or denied)
- **System Ready**: Indicator when idle

## 🔍 User Management

### List Authorized Users
```bash
python list_users.py
```

### Remove a User
```bash
python remove_user.py --name "John Doe"
```

### Enroll Additional Users
```bash
python enroll_user.py --name "Jane Smith" --samples 5
```

## ⚠️ Important Notes

1. **Continuous Operation**: The system runs 24/7 - press 'q' to quit
2. **Error Handling**: System never crashes - all errors logged and recovered
3. **Camera Required**: Webcam or IP camera must be connected
4. **Good Lighting**: Ensure adequate lighting for best face detection
5. **Multiple Samples**: Enroll users with 5+ samples from different angles
6. **Audit Trail**: All access events logged for security auditing

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review README.md for detailed documentation
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Verify camera is working: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

---

**System Status**: ✅ Fully Operational
**Version**: 2.0 - Security Door Access Control Edition
**Last Updated**: January 31, 2026
