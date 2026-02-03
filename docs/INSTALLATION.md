# Installation Guide

This guide provides detailed instructions for installing and setting up the Real-Time Facial Recognition Alert System.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Step-by-Step Installation](#step-by-step-installation)
- [Virtual Environment Setup](#virtual-environment-setup)
- [GPU Acceleration Setup](#gpu-acceleration-setup)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Camera**: Webcam or IP camera (minimum 720p)
- **Processor**: Intel i5 / AMD Ryzen 5 or equivalent

### Recommended Requirements

- **Operating System**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9 or 3.10
- **RAM**: 8GB or more
- **Storage**: 5GB free space (for models and data)
- **Camera**: 1080p webcam
- **Processor**: Intel i7 / AMD Ryzen 7 or better
- **GPU**: NVIDIA GPU with CUDA support (RTX 2060 or better)

### Optimal Requirements

- **RAM**: 16GB
- **Processor**: Intel i9 / AMD Ryzen 9
- **GPU**: NVIDIA RTX 3060 or better
- **Camera**: High-quality 1080p or 4K webcam

---

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone repository
git clone https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System.git
cd Real-Time-Facial-Recognition-Alert-System

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Package Install

```bash
# Clone repository
git clone https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System.git
cd Real-Time-Facial-Recognition-Alert-System

# Install as editable package
pip install -e .
```

### Method 3: From PyPI (Future)

```bash
# When published to PyPI
pip install real-time-facial-recognition
```

---

## Step-by-Step Installation

### Windows

```powershell
# 1. Install Python 3.8+
# Download from https://www.python.org/downloads/

# 2. Clone repository
git clone https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System.git
cd Real-Time-Facial-Recognition-Alert-System

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "from src.core.face_detector import FaceDetector; print('Installation successful!')"
```

### macOS

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python 3.8+
brew install python@3.9

# 3. Clone repository
git clone https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System.git
cd Real-Time-Facial-Recognition-Alert-System

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install dependencies
pip install -r requirements.txt

# 7. Verify installation
python -c "from src.core.face_detector import FaceDetector; print('Installation successful!')"
```

### Linux (Ubuntu/Debian)

```bash
# 1. Update package list
sudo apt update

# 2. Install Python and dependencies
sudo apt install python3.9 python3.9-venv python3-pip git

# 3. Clone repository
git clone https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System.git
cd Real-Time-Facial-Recognition-Alert-System

# 4. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install system dependencies for OpenCV
sudo apt install libgl1-mesa-glx libglib2.0-0

# 7. Install Python dependencies
pip install -r requirements.txt

# 8. Verify installation
python -c "from src.core.face_detector import FaceDetector; print('Installation successful!')"
```

---

## Virtual Environment Setup

### Why Use a Virtual Environment?

- Isolates project dependencies
- Prevents version conflicts
- Makes deployment easier
- Allows multiple Python projects

### Creating Virtual Environment

#### Using venv (Built-in)

```bash
# Create
python -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Deactivate
deactivate
```

#### Using conda

```bash
# Create
conda create -n face-recognition python=3.9

# Activate
conda activate face-recognition

# Deactivate
conda deactivate
```

---

## GPU Acceleration Setup

### NVIDIA GPU (CUDA)

#### Windows

```powershell
# 1. Install CUDA Toolkit 11.8
# Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive

# 2. Install cuDNN 8.6
# Download from: https://developer.nvidia.com/cudnn

# 3. Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# 4. Verify GPU detection
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

#### Linux

```bash
# 1. Install NVIDIA drivers
sudo apt install nvidia-driver-525

# 2. Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# 3. Install TensorFlow GPU
pip install tensorflow[and-cuda]

# 4. Verify
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### Apple Silicon (M1/M2)

```bash
# Install TensorFlow Metal plugin for GPU acceleration
pip install tensorflow-metal

# Verify
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## Troubleshooting

### Common Installation Issues

#### Issue: "pip install" fails

**Solution**:
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Try installing packages individually
pip install opencv-python
pip install tensorflow
pip install mtcnn
# ... continue for each package
```

#### Issue: OpenCV import error on Linux

**Solution**:
```bash
sudo apt install libgl1-mesa-glx libglib2.0-0
```

#### Issue: TensorFlow GPU not detected

**Solution**:
```bash
# Check CUDA installation
nvcc --version

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Reinstall TensorFlow GPU
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

#### Issue: MTCNN download fails

**Solution**:
```bash
# Download models manually
mkdir -p ~/.keras/mtcnn_weights
# Download weights from: https://github.com/ipazc/mtcnn/tree/master/mtcnn_weights
```

#### Issue: Permission denied on Linux/macOS

**Solution**:
```bash
# Don't use sudo with pip
# Use virtual environment instead
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Issue: "No module named 'src'"

**Solution**:
```bash
# Run from project root directory
cd Real-Time-Facial-Recognition-Alert-System

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Platform-Specific Issues

#### macOS: "SSL: CERTIFICATE_VERIFY_FAILED"

```bash
# Install certificates
/Applications/Python\ 3.9/Install\ Certificates.command
```

#### Windows: "Microsoft Visual C++ 14.0 is required"

```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++"
```

---

## Verification

### Verify Installation

```bash
# Test all imports
python -c "
from src.core.face_detector import FaceDetector
from src.core.face_recognition_model import FaceRecognitionModel
from src.core.database_manager import DatabaseManager
from src.security.liveness_detector import LivenessDetector
from config import config
print('✓ All imports successful!')
"

# Run basic tests
python tests/test_modules.py

# Check camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera not found')"
```

### Test Run

```bash
# Enroll a test user
python scripts/enroll_user.py --name "Test User" --samples 5

# Run system (press 'q' to quit)
python scripts/main.py

# Clean up test user
python scripts/remove_user.py --name "Test User"
```

---

## Next Steps

After successful installation:

1. Read [USAGE.md](USAGE.md) for usage instructions
2. Review [CONFIGURATION.md](CONFIGURATION.md) for configuration options
3. Check [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
4. Explore [API.md](API.md) for API documentation

---

## Getting Help

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System/issues)
2. Read the [Troubleshooting](#troubleshooting) section in main README
3. Ask in [GitHub Discussions](https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System/discussions)
4. Open a new issue with details about your problem
