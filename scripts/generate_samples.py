"""
Generate sample images showing the access control displays
"""

import cv2
import numpy as np
from src.utils.utils import (
    display_access_granted,
    display_access_denied,
    display_system_ready,
    display_system_status
)


def generate_sample_images():
    """Generate and save sample images of the access control displays"""
    
    # Create a sample frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some background to make it look more realistic
    cv2.rectangle(frame, (0, 0), (640, 480), (40, 40, 40), -1)
    
    # Generate ACCESS GRANTED image
    frame_granted = frame.copy()
    display_access_granted(frame_granted, "John Doe")
    cv2.imwrite("/tmp/access_granted_sample.jpg", frame_granted)
    print("✓ Saved access_granted_sample.jpg")
    
    # Generate ACCESS DENIED image
    frame_denied = frame.copy()
    display_access_denied(frame_denied)
    cv2.imwrite("/tmp/access_denied_sample.jpg", frame_denied)
    print("✓ Saved access_denied_sample.jpg")
    
    # Generate SYSTEM READY image
    frame_ready = frame.copy()
    display_system_ready(frame_ready)
    display_system_status(frame_ready, 29.8, 3661, "Last: GRANTED - Alice Smith")
    cv2.imwrite("/tmp/system_ready_sample.jpg", frame_ready)
    print("✓ Saved system_ready_sample.jpg")
    
    print("\nSample images generated successfully!")


if __name__ == '__main__':
    generate_sample_images()
