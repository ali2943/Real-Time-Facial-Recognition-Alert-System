# Usage Guide

Comprehensive guide for using the Real-Time Facial Recognition Alert System.

---

## Table of Contents

- [Enrollment Process](#enrollment-process)
- [Running the System](#running-the-system)
- [Keyboard Controls](#keyboard-controls)
- [Understanding Output](#understanding-output)
- [Log Analysis](#log-analysis)
- [User Management](#user-management)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

---

## Enrollment Process

### Basic Enrollment

Enroll a new user with default settings (5 samples):

```bash
python scripts/enroll_user.py --name "John Doe"
```

### Recommended Enrollment

For better accuracy, use 10-15 samples:

```bash
python scripts/enroll_user.py --name "John Doe" --samples 15
```

### High-Quality Enrollment

Enable quality checks during enrollment:

```bash
python scripts/enroll_user.py --name "John Doe" --samples 15 --quality-check
```

### Custom Camera

Use a specific camera:

```bash
python scripts/enroll_user.py --name "John Doe" --camera 1
```

### Enrollment Tips

**For Best Results:**

1. **Lighting**
   - Ensure even, natural lighting
   - Avoid harsh shadows or backlighting
   - Position light source in front of you

2. **Camera Position**
   - Keep face 1-2 feet from camera
   - Look directly at camera
   - Keep face level with camera

3. **Facial Expression**
   - Maintain neutral expression
   - Keep eyes open normally
   - Remove glasses if possible (unless always worn)

4. **Sample Variety**
   - Slight head turns (±10 degrees)
   - Minimal position changes
   - Keep face centered in frame

5. **What to Avoid**
   - Extreme angles or rotations
   - Blurry or motion-blurred images
   - Obstructions (hands, masks, sunglasses)
   - Extreme lighting conditions

### Enrollment Output

During enrollment, you'll see:
```
Enrolling user: John Doe
Samples to capture: 15

Sample 1/15: ✓ Captured
Sample 2/15: ✗ Quality too low, retrying...
Sample 2/15: ✓ Captured
...
Sample 15/15: ✓ Captured

✓ Successfully enrolled John Doe with 15 samples
Embeddings saved to database
```

---

## Running the System

### Standard Mode

```bash
python scripts/main.py
```

### Custom Camera

```bash
python scripts/main.py --camera 1
```

### Security Levels

```bash
# High security (strict thresholds, liveness enabled)
python scripts/main.py --security-level high

# Balanced (recommended)
python scripts/main.py --security-level balanced

# Lenient (faster, less secure)
python scripts/main.py --security-level lenient
```

### Debug Mode

```bash
python scripts/main.py --debug
```

### Headless Mode (No Display)

```bash
python scripts/main.py --headless
```

---

## Keyboard Controls

While the system is running, use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| **'q'** | Quit the application |
| **'s'** | Save current frame as image |
| **'r'** | Reset system state |
| **'p'** | Pause/Resume processing |
| **'d'** | Toggle debug display |
| **'l'** | Toggle liveness detection |
| **'h'** | Show help overlay |
| **'f'** | Toggle FPS display |
| **'1-3'** | Switch security levels |

---

## Understanding Output

### Visual Feedback

#### ACCESS GRANTED
```
┌─────────────────────────────────────┐
│                                     │
│        ✓ ACCESS GRANTED             │
│        Welcome, John Doe!           │
│        Confidence: 94%              │
│                                     │
└─────────────────────────────────────┘
```
- **Green text** - Access approved
- **User name** displayed
- **Confidence score** shown
- **Face box** highlighted in green

#### ACCESS DENIED
```
┌─────────────────────────────────────┐
│                                     │
│        ✗ ACCESS DENIED              │
│        Unknown Person               │
│                                     │
└─────────────────────────────────────┘
```
- **Red text** - Access rejected
- **"Unknown Person"** message
- **Face box** highlighted in red
- **Image saved** to unknown_faces/ (if enabled)

#### SYSTEM READY
```
┌─────────────────────────────────────┐
│                                     │
│        ⊙ SYSTEM READY               │
│        Monitoring...                │
│                                     │
└─────────────────────────────────────┘
```
- **Blue/white text** - Idle state
- **No face detected** or processing
- **System operational** and waiting

### Console Output

#### Normal Operation
```
[2024-01-15 10:23:45] System started
[2024-01-15 10:23:46] Camera initialized: Index 0
[2024-01-15 10:23:47] Loaded 3 users from database
[2024-01-15 10:23:50] Face detected
[2024-01-15 10:23:51] Liveness check: PASS (score: 0.85)
[2024-01-15 10:23:52] Recognition: John Doe (94% confidence)
[2024-01-15 10:23:52] ACCESS GRANTED
```

#### Error Conditions
```
[2024-01-15 10:24:10] WARNING: Low face quality (blur)
[2024-01-15 10:24:15] WARNING: Liveness check failed
[2024-01-15 10:24:20] ERROR: Camera disconnected, attempting reconnect...
[2024-01-15 10:24:22] INFO: Camera reconnected successfully
```

### Performance Metrics

Displayed on screen:
- **FPS**: Frames per second (15-30 typical)
- **Uptime**: System running time
- **Faces**: Number of faces detected
- **Queue**: Processing queue size

---

## Log Analysis

### Access Log Location

Logs are stored in `logs/access_log.txt`

### Log Format

```
[2024-01-15 10:23:52] ACCESS GRANTED | John Doe | Confidence: 0.94 | Liveness: 0.85
[2024-01-15 10:25:33] ACCESS DENIED | Unknown | Reason: No match found
[2024-01-15 10:27:14] ACCESS DENIED | Unknown | Reason: Liveness check failed
```

### Analyzing Logs

```bash
# View recent access attempts
tail -n 50 logs/access_log.txt

# Count successful access
grep "ACCESS GRANTED" logs/access_log.txt | wc -l

# Find specific user's access
grep "John Doe" logs/access_log.txt

# Check denied access
grep "ACCESS DENIED" logs/access_log.txt

# View today's logs
grep "$(date +%Y-%m-%d)" logs/access_log.txt
```

### Log Rotation

Logs are rotated daily. Old logs are stored as:
- `access_log_2024-01-14.txt`
- `access_log_2024-01-13.txt`
- etc.

---

## User Management

### List Users

```bash
# List all enrolled users
python scripts/list_users.py

# Output:
# Enrolled Users (3):
# 1. John Doe (15 samples)
# 2. Jane Smith (10 samples)
# 3. Bob Johnson (12 samples)
```

### Remove User

```bash
# Remove a user
python scripts/remove_user.py --name "John Doe"

# Force removal (no confirmation)
python scripts/remove_user.py --name "John Doe" --force
```

### Re-enroll User

To update a user's data:

```bash
# Remove old data
python scripts/remove_user.py --name "John Doe"

# Re-enroll with new samples
python scripts/enroll_user.py --name "John Doe" --samples 15
```

### Export/Import Database

```bash
# Export database
cp database/embeddings.pkl database/backup_$(date +%Y%m%d).pkl

# Import database
cp database/backup_20240115.pkl database/embeddings.pkl
```

---

## Advanced Usage

### Multiple Cameras

```bash
# Camera 0 (default)
python scripts/main.py --camera 0

# Camera 1 (external webcam)
python scripts/main.py --camera 1

# IP camera
python scripts/main.py --camera "rtsp://192.168.1.100:554/stream"
```

### Custom Configuration

```bash
# Use custom config file
python scripts/main.py --config custom_config.py

# Override specific settings
python scripts/main.py --threshold 0.7 --liveness-enabled
```

### Batch Processing

Process recorded video:

```bash
python scripts/main.py --input video.mp4 --output results.json
```

### API Mode

Run as REST API server:

```bash
python scripts/api_server.py --port 8080
```

Then use:
```bash
curl -X POST -F "image=@face.jpg" http://localhost:8080/recognize
```

---

## Best Practices

### 1. Enrollment

- ✅ **Enroll 15+ samples** per user
- ✅ **Use quality checks** during enrollment
- ✅ **Ensure good lighting** and camera quality
- ✅ **Re-enroll every 6-12 months** for accuracy
- ✅ **Vary head positions slightly** during enrollment

### 2. Operation

- ✅ **Monitor access logs** regularly
- ✅ **Calibrate thresholds** for your environment
- ✅ **Update database** when users' appearance changes
- ✅ **Keep system updated** with latest code
- ✅ **Backup database** before major changes

### 3. Security

- ✅ **Enable liveness detection** for sensitive areas
- ✅ **Review denied access** images regularly
- ✅ **Use high security preset** for critical applications
- ✅ **Monitor false accept/reject rates**
- ✅ **Encrypt database** if storing sensitive data

### 4. Performance

- ✅ **Use GPU acceleration** if available
- ✅ **Reduce resolution** if FPS is low
- ✅ **Close other camera applications**
- ✅ **Ensure adequate lighting**
- ✅ **Process every Nth frame** if needed

### 5. Maintenance

- ✅ **Clean camera lens** regularly
- ✅ **Review logs** weekly
- ✅ **Backup database** monthly
- ✅ **Update dependencies** quarterly
- ✅ **Test system** after configuration changes

---

## Troubleshooting Common Usage Issues

### Issue: User not recognized

**Possible Causes & Solutions:**
1. **Poor enrollment quality**
   - Re-enroll with more samples (15+)
   - Ensure good lighting during enrollment

2. **Appearance changed significantly**
   - Re-enroll user
   - Update samples periodically

3. **Threshold too strict**
   - Lower threshold in config (0.6 → 0.7)
   - Check recommended values for your model

4. **Liveness detection too strict**
   - Adjust liveness threshold
   - Temporarily disable to isolate issue

### Issue: High false accept rate

**Solutions:**
1. **Lower threshold** (0.7 → 0.6)
2. **Enable liveness detection**
3. **Improve enrollment quality**
4. **Use high security preset**

### Issue: Poor performance

**Solutions:**
1. **Enable GPU acceleration**
2. **Reduce frame resolution**
3. **Process every 2nd or 3rd frame**
4. **Disable unnecessary features**
5. **Close resource-intensive applications**

---

## Example Workflows

### Daily Security Door

```bash
# Morning: Start system
python scripts/main.py --security-level balanced

# Throughout day: System runs continuously
# Logs access attempts automatically

# Evening: Review logs
grep "$(date +%Y-%m-%d)" logs/access_log.txt

# Weekly: Check for unknowns
ls -lh unknown_faces/
```

### High-Security Application

```bash
# Use maximum security
python scripts/main.py --security-level high --liveness-enabled

# Review all access attempts
tail -f logs/access_log.txt

# Investigate denied access
ls -lht unknown_faces/ | head -10
```

### Testing New Users

```bash
# Enroll test user
python scripts/enroll_user.py --name "Test User" --samples 15

# Test recognition
python scripts/main.py

# Review results
grep "Test User" logs/access_log.txt

# Clean up
python scripts/remove_user.py --name "Test User"
```

---

## Getting Help

For additional help:

1. Check [README.md](../README.md) for overview
2. Review [CONFIGURATION.md](CONFIGURATION.md) for settings
3. See [API.md](API.md) for programming interface
4. Visit [GitHub Issues](https://github.com/ali2943/Real-Time-Facial-Recognition-Alert-System/issues)
