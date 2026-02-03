"""
Utility functions for the facial recognition system
"""

import os
import cv2
from datetime import datetime
from config import config


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
    font = cv2.FONT_HERSHEY_DUPLEX  # Bolder font
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
    font = cv2.FONT_HERSHEY_DUPLEX  # Bolder font
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


def log_access_event(name, status="ATTEMPTED", confidence=None, distance=None, reason=None, photo_filename=None, person_name=None):
    """
    Log access attempts with full details
    
    New signature:
        log_access_event(name, status, confidence=None, distance=None, reason=None, photo_filename=None)
        - name: Person name or "UNKNOWN"
        - status: "GRANTED", "DENIED - LOW CONFIDENCE", etc.
    
    Old signature (deprecated, for backward compatibility):
        log_access_event(event_type, person_name=None, photo_filename=None)
        - name (event_type): "ACCESS GRANTED", "ACCESS DENIED"
        - person_name: Person's name
    
    Args:
        name: Person name or event type (e.g., "UNKNOWN", "SPOOF ATTEMPT", "ACCESS GRANTED")
        status: Status of the event (e.g., "GRANTED", "DENIED - LOW CONFIDENCE")
        confidence: Optional confidence score (0.0 to 1.0)
        distance: Optional embedding distance
        reason: Optional reason for the event
        photo_filename: Optional filename of saved photo
        person_name: Optional person name (for backward compatibility only)
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Detect old usage pattern more robustly:
        # Old usage provides person_name AND doesn't provide new-style parameters
        if person_name is not None and confidence is None and distance is None and reason is None:
            # Old usage: event_type in name, person_name provided
            event_type = name
            if person_name:
                log_entry = f"[{timestamp}] {event_type} - {person_name}"
            else:
                log_entry = f"[{timestamp}] {event_type} - Unknown"
                if photo_filename:
                    log_entry += f" (Photo: {photo_filename})"
        else:
            # New usage: name is user name or "UNKNOWN", status is the actual status
            log_entry = f"[{timestamp}] {status} - User: {name or 'UNKNOWN'}"
            
            if confidence is not None:
                log_entry += f", Confidence: {confidence:.2%}"
            
            if distance is not None:
                if distance == float('inf'):
                    log_entry += f", Distance: inf"
                else:
                    log_entry += f", Distance: {distance:.4f}"
            
            if reason:
                log_entry += f", Reason: {reason}"
            
            if photo_filename:
                log_entry += f", Photo: {photo_filename}"
        
        log_entry += "\n"
        
        # Print to console if debug mode
        if config.DEBUG_MODE:
            print(f"[LOG] {log_entry.strip()}")
        
        # Write to file
        with open(config.LOG_FILE_PATH, 'a') as f:
            f.write(log_entry)
    
    except Exception as e:
        print(f"[ERROR] Failed to write log: {e}")


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


def draw_face_box(frame, box, label, is_authorized, distance=None):
    """
    Draw bounding box and label on frame
    
    Args:
        frame: Image frame
        box: [x, y, width, height]
        label: Text label to display
        is_authorized: True for authorized, False for unauthorized
        distance: Optional distance value to display (if SHOW_DISTANCE_ON_SCREEN enabled)
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
    
    # Add distance if enabled and provided
    if config.SHOW_DISTANCE_ON_SCREEN and distance is not None:
        text += f" (D: {distance:.3f})"
    
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


def display_validation_status(frame, validation_results):
    """
    Display validation check results on frame
    
    Args:
        frame: Video frame
        validation_results: Dict of check results {check_name: (passed, message)}
    """
    y_offset = 30
    
    for check_name, (passed, message) in validation_results.items():
        color = (0, 255, 0) if passed else (0, 165, 255)  # Green/Orange
        symbol = "✓" if passed else "!"
        
        text = f"{symbol} {check_name}: {message}"
        cv2.putText(frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 25
    
    return frame

