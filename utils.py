"""
Utility functions for the facial recognition system
"""

import os
import cv2
from datetime import datetime
import config


def display_access_granted(frame, person_name):
    """
    Display ACCESS GRANTED message with large green text
    
    Args:
        frame: Image frame
        person_name: Name of authorized person
    """
    height, width = frame.shape[:2]
    
    # Main message
    text = config.ACCESS_GRANTED_TEXT
    font = cv2.FONT_HERSHEY_BOLD
    font_scale = config.ACCESS_TEXT_FONT_SCALE
    thickness = config.ACCESS_TEXT_THICKNESS
    color = config.ACCESS_GRANTED_COLOR
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw background rectangle
    padding = 20
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (0, 0, 0),
        -1
    )
    
    # Draw main text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness
    )
    
    # Draw person name below
    name_text = person_name
    name_font_scale = 1.0
    name_thickness = 2
    (name_width, name_height), name_baseline = cv2.getTextSize(
        name_text, font, name_font_scale, name_thickness
    )
    name_x = (width - name_width) // 2
    name_y = y + text_height + 30
    
    cv2.putText(
        frame,
        name_text,
        (name_x, name_y),
        font,
        name_font_scale,
        (255, 255, 255),
        name_thickness
    )


def display_access_denied(frame):
    """
    Display ACCESS DENIED message with large red text
    
    Args:
        frame: Image frame
    """
    height, width = frame.shape[:2]
    
    # Main message
    text = config.ACCESS_DENIED_TEXT
    font = cv2.FONT_HERSHEY_BOLD
    font_scale = config.ACCESS_TEXT_FONT_SCALE
    thickness = config.ACCESS_TEXT_THICKNESS
    color = config.ACCESS_DENIED_COLOR
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw background rectangle
    padding = 20
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (0, 0, 0),
        -1
    )
    
    # Draw main text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness
    )
    
    # Draw warning below
    warning_text = "Unknown Person"
    warning_font_scale = 1.0
    warning_thickness = 2
    (warning_width, warning_height), warning_baseline = cv2.getTextSize(
        warning_text, font, warning_font_scale, warning_thickness
    )
    warning_x = (width - warning_width) // 2
    warning_y = y + text_height + 30
    
    cv2.putText(
        frame,
        warning_text,
        (warning_x, warning_y),
        font,
        warning_font_scale,
        (255, 255, 255),
        warning_thickness
    )


def display_system_ready(frame):
    """
    Display system ready status when no face is detected
    
    Args:
        frame: Image frame
    """
    text = config.SYSTEM_READY_TEXT
    cv2.putText(
        frame,
        text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        config.SYSTEM_READY_COLOR,
        2
    )


def log_access_event(event_type, person_name=None, photo_filename=None):
    """
    Log access event to file
    
    Args:
        event_type: "ACCESS GRANTED" or "ACCESS DENIED"
        person_name: Name of person (if known)
        photo_filename: Filename of saved photo (for denied access)
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if person_name:
            log_entry = f"[{timestamp}] {event_type} - {person_name}\n"
        else:
            log_entry = f"[{timestamp}] {event_type} - Unknown"
            if photo_filename:
                log_entry += f" (Photo: {photo_filename})"
            log_entry += "\n"
        
        with open(config.LOG_FILE_PATH, 'a') as f:
            f.write(log_entry)
    
    except Exception as e:
        print(f"[ERROR] Failed to log access event: {e}")


def display_system_status(frame, fps, uptime, last_event):
    """
    Display system statistics and status on frame
    
    Args:
        frame: Image frame
        fps: Current frames per second
        uptime: System uptime in seconds
        last_event: Last access event string
    """
    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    # Uptime
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    seconds = int(uptime % 60)
    uptime_text = f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}"
    cv2.putText(
        frame,
        uptime_text,
        (10, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Last event
    if last_event:
        cv2.putText(
            frame,
            last_event,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )


def draw_face_box(frame, box, label, is_authorized):
    """
    Draw bounding box and label on frame
    
    Args:
        frame: Image frame
        box: [x, y, width, height]
        label: Text label to display
        is_authorized: True for authorized, False for unauthorized
    """
    x, y, w, h = box
    
    # Choose color based on authorization
    color = config.BBOX_COLOR_LEGIT if is_authorized else config.BBOX_COLOR_UNKNOWN
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.BBOX_THICKNESS)
    
    # Prepare label text
    if is_authorized:
        text = f"{config.ALERT_MESSAGE_LEGIT}: {label}"
    else:
        text = config.ALERT_MESSAGE_UNKNOWN
    
    # Calculate text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        text, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS
    )
    
    # Draw background rectangle for text
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 10),
        (x + text_width, y),
        color,
        -1  # Filled rectangle
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y - baseline - 5),
        config.FONT,
        config.FONT_SCALE,
        (255, 255, 255),  # White text
        config.FONT_THICKNESS
    )


def save_unknown_face(face_img, face_id):
    """
    Save unknown face image with timestamp
    
    Args:
        face_img: Face image to save
        face_id: Unique identifier for the face
    """
    if not config.SAVE_UNKNOWN_FACES:
        return
    
    # Create directory if it doesn't exist
    if not os.path.exists(config.UNKNOWN_FACES_DIR):
        os.makedirs(config.UNKNOWN_FACES_DIR)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_{timestamp}_{face_id}.jpg"
    filepath = os.path.join(config.UNKNOWN_FACES_DIR, filename)
    
    # Save image
    cv2.imwrite(filepath, face_img)
    print(f"[ALERT] Unknown face saved: {filepath}")


def display_stats(frame, fps):
    """
    Display system statistics on frame (backward compatible)
    
    Args:
        frame: Image frame
        fps: Current frames per second
    """
    stats_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        stats_text,
        (10, 30),
        config.FONT,
        0.7,
        (255, 255, 255),
        2
    )
