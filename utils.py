"""
Utility functions for the facial recognition system
"""

import os
import cv2
from datetime import datetime
import config


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
    Display system statistics on frame
    
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
