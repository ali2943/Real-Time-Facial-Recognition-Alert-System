# Real-Time Facial Recognition Alert System

A comprehensive real-time facial recognition system that uses a live camera feed to identify whether a person is authorized (legit) or unauthorized (unknown) by comparing detected faces with pre-stored facial data, displaying appropriate alerts.

## 🎯 Features

- **Real-time Face Detection**: Uses MTCNN for robust face detection
- **Face Recognition**: Employs FaceNet embeddings for accurate face matching
- **Visual Alerts**: 
  - Green bounding boxes and "Legit Person" labels for authorized users
  - Red bounding boxes and "Alert: Unknown Person" for unauthorized individuals
- **User Management**: Easy enrollment and removal of authorized users
- **Unknown Face Logging**: Automatically saves images of unknown persons with timestamps
- **Performance Monitoring**: Real-time FPS display
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

### 2. Run the Facial Recognition System

Start the real-time facial recognition:

```bash
python main.py
```

Optional arguments:
- `--camera`: Camera device index (default: 0)

**During operation**:
- Authorized users will be highlighted with **green** boxes and labeled with their names
- Unauthorized persons will be highlighted with **red** boxes and labeled as "Alert: Unknown Person"
- Unknown faces are automatically saved to the `unknown_faces/` directory
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
├── config.py                   # Configuration settings
├── face_detector.py            # Face detection module (MTCNN)
├── face_recognition_model.py   # Face recognition module (FaceNet)
├── database_manager.py         # Database management for embeddings
├── utils.py                    # Utility functions
├── main.py                     # Main application
├── enroll_user.py             # User enrollment script
├── list_users.py              # List authorized users
├── remove_user.py             # Remove users from database
├── requirements.txt           # Python dependencies
├── database/                  # Stored face embeddings (auto-created)
└── unknown_faces/            # Saved unknown face images (auto-created)
```

## ⚙️ Configuration

Edit `config.py` to customize system parameters:

- **Face Detection**:
  - `FACE_DETECTION_CONFIDENCE`: Minimum confidence threshold (0-1)
  - `MIN_FACE_SIZE`: Minimum face size in pixels

- **Face Recognition**:
  - `RECOGNITION_THRESHOLD`: Maximum distance for a match (lower = stricter)
  - Default: 0.6 (adjust based on your needs)

- **Display**:
  - `BBOX_COLOR_LEGIT`: Color for authorized users (Green: 0, 255, 0)
  - `BBOX_COLOR_UNKNOWN`: Color for unauthorized users (Red: 0, 0, 255)

- **Alerts**:
  - `SAVE_UNKNOWN_FACES`: Enable/disable saving unknown faces
  - `UNKNOWN_FACES_DIR`: Directory for unknown face images

## 🔧 System Workflow

1. **Initialize**: Load models (MTCNN, FaceNet) and authorized user database
2. **Capture**: Read frames from camera feed
3. **Detect**: Identify faces using MTCNN
4. **Extract**: Crop and preprocess face regions
5. **Encode**: Generate face embeddings using FaceNet
6. **Compare**: Match against stored authorized embeddings
7. **Classify**: Determine if person is authorized or unknown
8. **Alert**: Display appropriate visual alerts
9. **Log**: Save unknown faces (optional)

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

1. **Authorized Users**: Verify green boxes and correct name labels
2. **Unknown Persons**: Verify red boxes and alert messages
3. **Different Conditions**: Test under various lighting and angles
4. **Performance**: Monitor FPS for real-time capability

## 🔒 Security Features

- Unknown face detection and alerting
- Automatic logging of unauthorized access attempts
- Configurable recognition threshold for security vs. convenience
- Timestamped unknown face images for audit trail

## 🚀 Performance Optimization

- Face detection runs on each frame
- Embeddings generated only for detected faces
- Efficient database lookup using vectorized operations
- Optional GPU acceleration for deep learning models

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
