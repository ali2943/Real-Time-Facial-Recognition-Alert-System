# Real-Time Facial Recognition Alert System

A comprehensive real-time facial recognition system configured as a **continuous 24/7 security door access control system**. The system uses live camera feed to identify whether a person is authorized or unauthorized by comparing detected faces with pre-stored facial data, displaying appropriate access control alerts.

## 🎯 Features

- **Continuous 24/7 Operation**: Never stops - automatically recovers from errors and camera disconnections
- **Real-time Face Detection**: Uses MTCNN for robust face detection
- **Face Recognition**: Employs FaceNet embeddings for accurate face matching
- **Access Control Feedback**: 
  - Large **"ACCESS GRANTED"** message in green for authorized users with name display
  - Large **"ACCESS DENIED"** message in red for unauthorized individuals
  - Console logging of all access attempts
  - "System Ready" indicator when idle
- **Comprehensive Error Handling**: Handles all detection errors, camera failures, and edge cases gracefully
- **Automatic Camera Reconnection**: Auto-reconnects if camera disconnects (configurable)
- **Access Event Logging**: Logs all access attempts with timestamps to `access_log.txt`
- **Unknown Face Capture**: Automatically saves images of unauthorized persons with timestamps
- **User Management**: Easy enrollment and removal of authorized users
- **Access Cooldown**: Prevents spam with configurable cooldown period between access attempts
- **Performance Monitoring**: Real-time FPS and system uptime display
- **Modular Architecture**: Clean, maintainable, and extensible code structure

## 📋 Requirements

- Python 3.8 or higher
- Webcam or IP camera
- GPU (optional, for better performance)

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System.git
cd Real-Time-Facial-Recognition-Alert-System
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Enroll Authorized Users

Before running the system, you need to enroll authorized users:

```bash
python enroll_user.py --name "John Doe" --samples 5
```

- `--name`: Name of the person to enroll (required)
- `--samples`: Number of face samples to capture (default: 5)

**Instructions during enrollment**:
- Position your face in the camera frame
- Press **SPACE** to capture each sample
- Press **q** to quit enrollment

### 2. Run the Security Door Access Control System

Start the continuous security door access control system:

```bash
python main.py
```

Optional arguments:
- `--camera`: Camera device index (default: 0)

**System Behavior**:
- **ACCESS GRANTED**: When an authorized person is detected:
  - Large green "ACCESS GRANTED" text appears on screen for 2 seconds
  - Person's name is displayed below the message
  - Console prints: `[SUCCESS] Access Granted: [Person Name]`
  - Event is logged to `access_log.txt` with timestamp
  
- **ACCESS DENIED**: When an unauthorized person is detected:
  - Large red "ACCESS DENIED" text appears on screen for 3 seconds
  - "Unknown Person" warning displayed below the message
  - Console prints: `[FAILURE] Access Denied: Unknown Person`
  - Unknown face photo saved with timestamp to `unknown_faces/` directory
  - Event is logged to `access_log.txt` with timestamp
  
- **System Ready**: When no face is detected, displays "System Ready" indicator
- **Cooldown Period**: 3-second cooldown between access attempts prevents spam
- **Continuous Operation**: System runs 24/7, recovering from any errors automatically
- **Auto-Reconnect**: If camera disconnects, system attempts to reconnect automatically
- Press **q** to quit the application

### 3. Manage Users

**List all authorized users**:
```bash
python list_users.py
```

**Remove a user**:
```bash
python remove_user.py --name "John Doe"
```

## 📁 Project Structure

```
Real-Time-Facial-Recognition-Alert-System/
├── config.py                   # Configuration settings (including access control)
├── face_detector.py            # Face detection module (MTCNN) with error handling
├── face_recognition_model.py   # Face recognition module (FaceNet)
├── database_manager.py         # Database management for embeddings
├── utils.py                    # Utility functions (access control displays, logging)
├── main.py                     # Main application (continuous door access control)
├── enroll_user.py             # User enrollment script
├── list_users.py              # List authorized users
├── remove_user.py             # Remove users from database
├── requirements.txt           # Python dependencies
├── access_log.txt             # Access event log (auto-created)
├── database/                  # Stored face embeddings (auto-created)
└── unknown_faces/            # Saved unknown face images (auto-created)
```

## ⚙️ Configuration

Edit `config.py` to customize system parameters:

- **Face Detection**:
  - `FACE_DETECTION_CONFIDENCE`: Minimum confidence threshold (default: 0.5, lower for better detection)
  - `MIN_FACE_SIZE`: Minimum face size in pixels (default: 20)

- **Face Recognition**:
  - `RECOGNITION_THRESHOLD`: Maximum distance for a match (default: 0.6, lower = stricter)

- **Security Door Access Control**:
  - `ACCESS_GRANTED_DISPLAY_TIME`: Duration to show granted message (default: 2 seconds)
  - `ACCESS_DENIED_DISPLAY_TIME`: Duration to show denied message (default: 3 seconds)
  - `ACCESS_COOLDOWN`: Cooldown between access attempts (default: 3 seconds)
  - `LOG_FILE_PATH`: Path to access log file (default: "access_log.txt")
  - `AUTO_RECONNECT_CAMERA`: Auto-reconnect on camera failure (default: True)
  - `MAX_RECONNECT_ATTEMPTS`: Maximum reconnection attempts (default: 5)
  - `FRAME_SKIP`: Process every Nth frame for performance (default: 1)
  - `ENABLE_AUDIO_FEEDBACK`: Play sounds for access events (default: False, not implemented)

- **Display**:
  - `BBOX_COLOR_LEGIT`: Color for authorized users (Green: 0, 255, 0)
  - `BBOX_COLOR_UNKNOWN`: Color for unauthorized users (Red: 0, 0, 255)
  - `ACCESS_GRANTED_COLOR`: Color for granted message (Green: 0, 255, 0)
  - `ACCESS_DENIED_COLOR`: Color for denied message (Red: 0, 0, 255)

- **Alerts**:
  - `SAVE_UNKNOWN_FACES`: Enable/disable saving unknown faces (default: True)
  - `UNKNOWN_FACES_DIR`: Directory for unknown face images (default: "unknown_faces")

## 🔧 System Workflow - Security Door Access Control

1. **Initialize**: Load models (MTCNN, FaceNet) and authorized user database
2. **Continuous Loop**: Run 24/7 with error recovery
3. **Capture**: Read frames from camera feed with reconnection on failure
4. **Detect**: Identify faces using MTCNN (with error handling)
5. **Extract**: Crop and preprocess face regions
6. **Encode**: Generate face embeddings using FaceNet
7. **Compare**: Match against stored authorized embeddings
8. **Access Decision**:
   - If matched: Display ACCESS GRANTED, log event, show person name
   - If not matched: Display ACCESS DENIED, save photo, log event
9. **Cooldown**: Wait configured time before processing next access attempt
10. **System Ready**: Display ready status when idle
11. **Recovery**: Automatically recover from any errors and continue operation

## 🎨 Technical Details

### Face Detection
- **Algorithm**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Features**: Robust to various lighting conditions and face angles
- **Output**: Bounding boxes and facial landmarks

### Face Recognition
- **Model**: FaceNet (Keras implementation)
- **Embedding Size**: 128-dimensional vectors
- **Similarity Metric**: Euclidean distance
- **Threshold**: Configurable (default: 0.6)

### Database
- **Storage**: Pickle format for fast I/O
- **Structure**: Dictionary mapping names to embedding lists
- **Support**: Multiple samples per person for better accuracy

## 🧪 Testing

Test the system with:

1. **Authorized Users**: Verify ACCESS GRANTED message appears in green with person's name
2. **Unknown Persons**: Verify ACCESS DENIED message appears in red
3. **Continuous Operation**: Verify system doesn't crash on errors
4. **Different Conditions**: Test under various lighting and angles
5. **Performance**: Monitor FPS and system uptime
6. **Access Log**: Check `access_log.txt` for event logging
7. **Camera Disconnection**: Test auto-reconnection by disconnecting/reconnecting camera
8. **Cooldown**: Verify cooldown period prevents spam access attempts

## 🔒 Security Features

- **Continuous 24/7 Operation**: Never stops, automatically recovers from all errors
- **Access Event Logging**: All access attempts logged with timestamps to `access_log.txt`
- **Unknown Face Capture**: Automatically saves images of unauthorized persons
- **Access Cooldown**: Prevents spam with 3-second cooldown between attempts
- **Robust Error Handling**: Handles MTCNN errors, camera failures, detection errors gracefully
- **Automatic Recovery**: Camera auto-reconnection on disconnection
- **Configurable Thresholds**: Adjust recognition threshold for security vs. convenience
- **Timestamped Audit Trail**: Unknown face images saved with timestamps

## 📋 Access Log Format

The system logs all access events to `access_log.txt` in the following format:

```
[2026-01-31 14:30:45] ACCESS GRANTED - John Doe
[2026-01-31 14:31:12] ACCESS DENIED - Unknown (Photo: unknown_20260131_143112_0.jpg)
[2026-01-31 14:32:05] ACCESS GRANTED - Jane Smith
```

## 🚀 Performance Optimization

- Face detection optimized with lower confidence threshold (0.5) for better detection
- Smaller minimum face size (20 pixels) for better detection at distance
- Frame skipping option available via `FRAME_SKIP` config for better performance
- Embeddings generated only for detected faces
- Efficient database lookup using vectorized operations
- Optional GPU acceleration for deep learning models
- Error recovery without restart overhead

## 🔧 Troubleshooting

**System Not Detecting Faces**:
- Ensure good lighting conditions
- Check camera is working (`--camera` parameter)
- Lower `FACE_DETECTION_CONFIDENCE` in config.py (already set to 0.5)
- Ensure face is within camera frame

**Camera Disconnection Issues**:
- `AUTO_RECONNECT_CAMERA` is enabled by default
- System will attempt reconnection up to `MAX_RECONNECT_ATTEMPTS` times
- Check camera connection and drivers

**False Positives/Negatives**:
- Adjust `RECOGNITION_THRESHOLD` in config.py
- Lower threshold = stricter matching (fewer false positives)
- Higher threshold = looser matching (fewer false negatives)
- Enroll users with multiple samples from different angles

**System Crashes**:
- The system is designed to NEVER crash
- All errors are caught and logged
- If system exits, check console for critical errors
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Access Log Not Created**:
- Log file is created automatically on first access event
- Check write permissions in current directory
- Configure custom path via `LOG_FILE_PATH` in config.py

## 📝 Notes

- Ensure good lighting for optimal face detection
- Enroll users with multiple samples from different angles
- Adjust `RECOGNITION_THRESHOLD` if getting false positives/negatives
- Lower threshold = stricter matching (fewer false positives)
- Higher threshold = looser matching (fewer false negatives)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Author

Ali2943

## 🙏 Acknowledgments

- MTCNN for face detection
- FaceNet for face recognition embeddings
- OpenCV for computer vision operations
- Keras and TensorFlow for deep learning support
