# Examples and Usage Guide

## Quick Start Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System.git
cd Real-Time-Facial-Recognition-Alert-System

# Install dependencies
pip install -r requirements.txt
```

### 2. Enroll Your First User
```bash
# Enroll yourself with 5 face samples
python enroll_user.py --name "John Doe" --samples 5
```

**During enrollment:**
- Make sure your face is well-lit and clearly visible
- Position your face in the green bounding box
- Press **SPACE** to capture each sample
- Vary your head angle slightly between samples for better recognition
- Press **q** to cancel if needed

### 3. Run the System
```bash
# Start the facial recognition system
python main.py
```

**During operation:**
- The system will show a live camera feed
- Authorized users will have **green** boxes with their names
- Unknown persons will have **red** boxes with an alert
- Press **q** to quit

## Common Use Cases

### Example 1: Home Security System
```bash
# Enroll family members
python enroll_user.py --name "Alice" --samples 7
python enroll_user.py --name "Bob" --samples 7
python enroll_user.py --name "Charlie" --samples 7

# List enrolled users
python list_users.py

# Run the system
python main.py
```

### Example 2: Office Access Control
```bash
# Enroll employees
python enroll_user.py --name "Employee_001" --samples 10
python enroll_user.py --name "Employee_002" --samples 10

# Run with different camera (if needed)
python main.py --camera 1
```

### Example 3: Adding Multiple Samples
```bash
# Enroll user multiple times to add more samples
python enroll_user.py --name "John Doe" --samples 3
# Run again to add more samples
python enroll_user.py --name "John Doe" --samples 3
```

## Configuration Tuning

### Adjusting Recognition Threshold

Edit `config.py`:

```python
# For stricter matching (fewer false positives, more false negatives)
RECOGNITION_THRESHOLD = 0.4  # Lower value

# For looser matching (more false positives, fewer false negatives)
RECOGNITION_THRESHOLD = 0.8  # Higher value

# Default balanced value
RECOGNITION_THRESHOLD = 0.6  # Recommended
```

### Adjusting Face Detection Confidence

```python
# For stricter face detection (miss some faces but fewer false detections)
FACE_DETECTION_CONFIDENCE = 0.95

# For more permissive detection (detect more faces but may have false positives)
FACE_DETECTION_CONFIDENCE = 0.7

# Default value
FACE_DETECTION_CONFIDENCE = 0.9
```

## Troubleshooting

### Problem: "No face detected" during enrollment
**Solutions:**
- Ensure proper lighting (face should be well-lit)
- Move closer to the camera
- Remove glasses, hats, or face coverings
- Make sure face is fully visible in frame

### Problem: Authorized users not recognized
**Solutions:**
- Increase `RECOGNITION_THRESHOLD` in `config.py`
- Enroll more samples: `python enroll_user.py --name "User" --samples 10`
- Ensure consistent lighting during enrollment and recognition
- Re-enroll with better quality samples

### Problem: Unknown persons incorrectly recognized
**Solutions:**
- Decrease `RECOGNITION_THRESHOLD` in `config.py`
- Enroll more diverse samples of authorized users
- Increase face detection confidence

### Problem: Low FPS / Slow performance
**Solutions:**
- Use a GPU if available (TensorFlow will auto-detect)
- Reduce camera resolution in `config.py`:
  ```python
  FRAME_WIDTH = 320
  FRAME_HEIGHT = 240
  ```
- Skip frames for processing (modify main.py to process every Nth frame)

### Problem: Camera not opening
**Solutions:**
- Try different camera index: `python main.py --camera 1`
- Check camera permissions
- Ensure camera is not being used by another application
- On Linux, you may need to add user to video group:
  ```bash
  sudo usermod -a -G video $USER
  ```

## Advanced Usage

### Viewing Unknown Faces
Unknown faces are automatically saved to `unknown_faces/` directory:
```bash
ls -lt unknown_faces/  # View saved unknown faces
```

### Database Management
```bash
# View all enrolled users
python list_users.py

# Remove a user
python remove_user.py --name "John Doe"

# Backup database
cp database/embeddings.pkl database/embeddings_backup.pkl

# Restore database
cp database/embeddings_backup.pkl database/embeddings.pkl
```

### Using with IP Camera
Modify `main.py` to use IP camera URL:
```python
# Replace VideoCapture(camera_index) with IP camera URL
cap = cv2.VideoCapture('http://192.168.1.100:8080/video')
```

## Performance Tips

1. **Better Recognition Accuracy:**
   - Enroll users with 7-10 samples
   - Capture samples from different angles
   - Use consistent lighting
   - Ensure high-quality camera

2. **Faster Processing:**
   - Use GPU-enabled TensorFlow
   - Reduce frame resolution
   - Process every 2nd or 3rd frame

3. **Better Detection:**
   - Ensure good lighting
   - Avoid strong backlighting
   - Keep face size at least 100x100 pixels

## Security Recommendations

1. **Physical Security:**
   - Place camera in secure location
   - Use tamper-proof camera housing
   - Ensure proper lighting

2. **System Security:**
   - Backup embeddings database regularly
   - Monitor unknown_faces directory
   - Review logs periodically
   - Set appropriate recognition thresholds

3. **Privacy:**
   - Inform users about facial recognition
   - Comply with local privacy laws
   - Secure database files
   - Implement access controls

## Integration Examples

### Sending Email Alerts
Add to `utils.py`:
```python
import smtplib
from email.mime.text import MIMEText

def send_alert_email(unknown_face_path):
    msg = MIMEText(f"Unknown person detected: {unknown_face_path}")
    msg['Subject'] = 'Security Alert: Unknown Person'
    msg['From'] = 'security@example.com'
    msg['To'] = 'admin@example.com'
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('user@gmail.com', 'password')
        server.send_message(msg)
```

### Logging to File
Add to `main.py`:
```python
import logging

logging.basicConfig(
    filename='facial_recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# In process_frame function:
if matched_name:
    logging.info(f"Authorized access: {matched_name}")
else:
    logging.warning(f"Unauthorized access detected")
```

### Database with SQLite
Replace pickle database with SQLite for better scalability:
```python
import sqlite3
import numpy as np

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('users.db')
        self.create_table()
    
    def create_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                embedding BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
```

## Testing Checklist

- [ ] System starts without errors
- [ ] Camera feed displays correctly
- [ ] Face detection works for all users
- [ ] Authorized users recognized with green boxes
- [ ] Unknown persons detected with red boxes
- [ ] FPS is acceptable (>10 FPS)
- [ ] Unknown faces saved correctly
- [ ] User enrollment works properly
- [ ] Database operations work correctly
- [ ] System handles multiple faces
- [ ] System works in different lighting
- [ ] System quits cleanly with 'q' key
