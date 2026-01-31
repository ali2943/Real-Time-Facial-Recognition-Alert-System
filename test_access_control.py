"""
Test script for security door access control features
"""

import cv2
import numpy as np
import os
import time
from utils import (
    display_access_granted,
    display_access_denied,
    display_system_ready,
    log_access_event,
    display_system_status
)
import config


def test_access_control_displays():
    """Test access control display functions"""
    print("\n[TEST] Testing access control display functions...")
    
    try:
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test display_access_granted
        display_access_granted(frame.copy(), "John Doe")
        print("✓ display_access_granted works")
        
        # Test display_access_denied
        display_access_denied(frame.copy())
        print("✓ display_access_denied works")
        
        # Test display_system_ready
        display_system_ready(frame.copy())
        print("✓ display_system_ready works")
        
        # Test display_system_status
        display_system_status(frame.copy(), 30.0, 3600, "Last: GRANTED - Test User")
        print("✓ display_system_status works")
        
        print("\n[TEST] Access control display tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Access control display test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_access_logging():
    """Test access event logging"""
    print("\n[TEST] Testing access event logging...")
    
    try:
        # Clean up any existing log file
        if os.path.exists(config.LOG_FILE_PATH):
            os.remove(config.LOG_FILE_PATH)
        
        # Test logging access granted
        log_access_event("ACCESS GRANTED", person_name="John Doe")
        print("✓ Logged ACCESS GRANTED event")
        
        # Test logging access denied
        log_access_event("ACCESS DENIED", photo_filename="unknown_test.jpg")
        print("✓ Logged ACCESS DENIED event")
        
        # Verify log file exists
        assert os.path.exists(config.LOG_FILE_PATH), "Log file not created"
        print(f"✓ Log file created: {config.LOG_FILE_PATH}")
        
        # Read and verify log contents
        with open(config.LOG_FILE_PATH, 'r') as f:
            log_contents = f.read()
            assert "ACCESS GRANTED" in log_contents, "ACCESS GRANTED not in log"
            assert "John Doe" in log_contents, "Person name not in log"
            assert "ACCESS DENIED" in log_contents, "ACCESS DENIED not in log"
            assert "unknown_test.jpg" in log_contents, "Photo filename not in log"
        
        print("✓ Log file contents verified")
        
        # Display log contents
        print("\n--- Log File Contents ---")
        print(log_contents)
        print("-------------------------\n")
        
        # Clean up
        os.remove(config.LOG_FILE_PATH)
        
        print("\n[TEST] Access event logging tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Access logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_values():
    """Test new configuration values"""
    print("\n[TEST] Testing new configuration values...")
    
    try:
        # Check access control config values
        assert hasattr(config, 'ACCESS_GRANTED_DISPLAY_TIME'), "Missing ACCESS_GRANTED_DISPLAY_TIME"
        assert hasattr(config, 'ACCESS_DENIED_DISPLAY_TIME'), "Missing ACCESS_DENIED_DISPLAY_TIME"
        assert hasattr(config, 'ACCESS_COOLDOWN'), "Missing ACCESS_COOLDOWN"
        assert hasattr(config, 'LOG_FILE_PATH'), "Missing LOG_FILE_PATH"
        assert hasattr(config, 'AUTO_RECONNECT_CAMERA'), "Missing AUTO_RECONNECT_CAMERA"
        assert hasattr(config, 'MAX_RECONNECT_ATTEMPTS'), "Missing MAX_RECONNECT_ATTEMPTS"
        assert hasattr(config, 'FRAME_SKIP'), "Missing FRAME_SKIP"
        
        print(f"✓ ACCESS_GRANTED_DISPLAY_TIME: {config.ACCESS_GRANTED_DISPLAY_TIME}")
        print(f"✓ ACCESS_DENIED_DISPLAY_TIME: {config.ACCESS_DENIED_DISPLAY_TIME}")
        print(f"✓ ACCESS_COOLDOWN: {config.ACCESS_COOLDOWN}")
        print(f"✓ LOG_FILE_PATH: {config.LOG_FILE_PATH}")
        print(f"✓ AUTO_RECONNECT_CAMERA: {config.AUTO_RECONNECT_CAMERA}")
        print(f"✓ MAX_RECONNECT_ATTEMPTS: {config.MAX_RECONNECT_ATTEMPTS}")
        print(f"✓ FRAME_SKIP: {config.FRAME_SKIP}")
        
        # Check updated detection values for improved accuracy
        assert config.FACE_DETECTION_CONFIDENCE == 0.7, "FACE_DETECTION_CONFIDENCE not 0.7"
        print(f"✓ FACE_DETECTION_CONFIDENCE: {config.FACE_DETECTION_CONFIDENCE}")
        
        assert config.MIN_FACE_SIZE == 60, "MIN_FACE_SIZE not 60"
        print(f"✓ MIN_FACE_SIZE: {config.MIN_FACE_SIZE}")
        
        # Check updated recognition threshold for improved accuracy
        assert config.RECOGNITION_THRESHOLD == 1.0, "RECOGNITION_THRESHOLD not 1.0"
        print(f"✓ RECOGNITION_THRESHOLD: {config.RECOGNITION_THRESHOLD}")
        
        # Check debug settings
        assert config.DEBUG_MODE == True, "DEBUG_MODE not True"
        print(f"✓ DEBUG_MODE: {config.DEBUG_MODE}")
        
        assert config.SHOW_DISTANCE_ON_SCREEN == True, "SHOW_DISTANCE_ON_SCREEN not True"
        print(f"✓ SHOW_DISTANCE_ON_SCREEN: {config.SHOW_DISTANCE_ON_SCREEN}")
        
        print("\n[TEST] Configuration value tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all access control tests"""
    print("="*60)
    print("SECURITY DOOR ACCESS CONTROL - FEATURE TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Config Values", test_config_values()))
    results.append(("Access Displays", test_access_control_displays()))
    results.append(("Access Logging", test_access_logging()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*60)
    if all_passed:
        print("✓ ALL ACCESS CONTROL TESTS PASSED!")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("="*60)
        return 1


if __name__ == '__main__':
    import sys
    exit_code = main()
    sys.exit(exit_code)
