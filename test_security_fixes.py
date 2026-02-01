"""
Test script for critical security vulnerability fixes
Tests the fixes for unknown person access vulnerability
"""

import os
import sys
import pickle
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch
import config


def test_empty_database_rejection():
    """Test that system refuses to start with empty database"""
    print("\n[TEST] Testing empty database rejection...")
    
    try:
        # Create a temporary database directory
        temp_db_dir = tempfile.mkdtemp()
        original_db_dir = config.DATABASE_DIR
        config.DATABASE_DIR = temp_db_dir
        
        # Ensure database file doesn't exist
        db_file = os.path.join(temp_db_dir, config.EMBEDDINGS_FILE)
        if os.path.exists(db_file):
            os.remove(db_file)
        
        # Create empty database file
        with open(db_file, 'wb') as f:
            pickle.dump({}, f)
        
        # Try to initialize system - should raise RuntimeError
        try:
            from main import FacialRecognitionSystem
            system = FacialRecognitionSystem()
            print("✗ System started with empty database - SECURITY VULNERABILITY!")
            return False
        except RuntimeError as e:
            if "Cannot start system with empty database" in str(e):
                print("✓ System correctly refused to start with empty database")
                print(f"  Error message: {e}")
                return True
            else:
                print(f"✗ Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup
            config.DATABASE_DIR = original_db_dir
            shutil.rmtree(temp_db_dir, ignore_errors=True)
    
    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_manager_empty_check():
    """Test that database manager handles empty database correctly"""
    print("\n[TEST] Testing database manager empty database handling...")
    
    try:
        # Create a temporary database
        temp_db_dir = tempfile.mkdtemp()
        original_db_dir = config.DATABASE_DIR
        config.DATABASE_DIR = temp_db_dir
        
        # Create empty database
        db_file = os.path.join(temp_db_dir, config.EMBEDDINGS_FILE)
        with open(db_file, 'wb') as f:
            pickle.dump({}, f)
        
        from database_manager import DatabaseManager
        from face_recognition_model import FaceRecognitionModel
        
        db_manager = DatabaseManager()
        recognizer = FaceRecognitionModel()
        
        # Create a dummy embedding
        dummy_embedding = np.random.randn(128)
        
        # Try to find match in empty database
        matched_name, distance = db_manager.find_match(dummy_embedding, recognizer)
        
        # Verify results
        if matched_name is None and distance == float('inf'):
            print("✓ Database manager correctly returns None for empty database")
            return True
        else:
            print(f"✗ Unexpected result: name={matched_name}, distance={distance}")
            return False
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        config.DATABASE_DIR = original_db_dir
        shutil.rmtree(temp_db_dir, ignore_errors=True)


def test_unknown_person_rejection():
    """Test that unknown persons are rejected"""
    print("\n[TEST] Testing unknown person rejection...")
    
    try:
        # Create a temporary database with one user
        temp_db_dir = tempfile.mkdtemp()
        original_db_dir = config.DATABASE_DIR
        config.DATABASE_DIR = temp_db_dir
        
        # Create database with one user
        db_file = os.path.join(temp_db_dir, config.EMBEDDINGS_FILE)
        test_embedding = np.random.randn(128)
        db_data = {"TestUser": [test_embedding]}
        with open(db_file, 'wb') as f:
            pickle.dump(db_data, f)
        
        from database_manager import DatabaseManager
        from face_recognition_model import FaceRecognitionModel
        
        db_manager = DatabaseManager()
        recognizer = FaceRecognitionModel()
        
        # Create a completely different embedding (unknown person)
        # Make it far enough from the stored embedding to exceed threshold
        unknown_embedding = np.random.randn(128) * 10  # Very different
        
        # Try to find match
        matched_name, distance = db_manager.find_match(unknown_embedding, recognizer)
        
        # Verify that unknown person is rejected
        if matched_name is None:
            print(f"✓ Unknown person correctly rejected (distance: {distance:.4f})")
            return True
        else:
            print(f"✗ Unknown person was matched as '{matched_name}' - SECURITY VULNERABILITY!")
            print(f"  Distance: {distance:.4f}, Threshold: {config.RECOGNITION_THRESHOLD}")
            return False
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        config.DATABASE_DIR = original_db_dir
        shutil.rmtree(temp_db_dir, ignore_errors=True)


def test_threshold_enforcement():
    """Test that recognition threshold is properly enforced"""
    print("\n[TEST] Testing threshold enforcement...")
    
    try:
        # Create a temporary database
        temp_db_dir = tempfile.mkdtemp()
        original_db_dir = config.DATABASE_DIR
        config.DATABASE_DIR = temp_db_dir
        
        # Create database with one user
        db_file = os.path.join(temp_db_dir, config.EMBEDDINGS_FILE)
        test_embedding = np.random.randn(128)
        db_data = {"TestUser": [test_embedding]}
        with open(db_file, 'wb') as f:
            pickle.dump(db_data, f)
        
        from database_manager import DatabaseManager
        from face_recognition_model import FaceRecognitionModel
        
        db_manager = DatabaseManager()
        recognizer = FaceRecognitionModel()
        
        # Test 1: Very similar embedding (should match)
        similar_embedding = test_embedding + np.random.randn(128) * 0.01  # Small noise
        matched_name, distance = db_manager.find_match(similar_embedding, recognizer)
        
        if distance < config.RECOGNITION_THRESHOLD:
            if matched_name == "TestUser":
                print(f"✓ Similar face correctly matched (distance: {distance:.4f} < threshold: {config.RECOGNITION_THRESHOLD})")
                test1_pass = True
            else:
                print(f"✗ Similar face matched wrong person: {matched_name}")
                test1_pass = False
        else:
            print(f"✗ Similar face rejected (distance: {distance:.4f} >= threshold: {config.RECOGNITION_THRESHOLD})")
            test1_pass = False
        
        # Test 2: Very different embedding (should NOT match)
        different_embedding = np.random.randn(128) * 5  # Very different
        matched_name2, distance2 = db_manager.find_match(different_embedding, recognizer)
        
        if distance2 >= config.RECOGNITION_THRESHOLD:
            if matched_name2 is None:
                print(f"✓ Different face correctly rejected (distance: {distance2:.4f} >= threshold: {config.RECOGNITION_THRESHOLD})")
                test2_pass = True
            else:
                print(f"✗ Different face incorrectly matched as: {matched_name2}")
                test2_pass = False
        else:
            print(f"✗ Different face incorrectly matched (distance: {distance2:.4f} < threshold: {config.RECOGNITION_THRESHOLD})")
            test2_pass = False
        
        return test1_pass and test2_pass
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        config.DATABASE_DIR = original_db_dir
        shutil.rmtree(temp_db_dir, ignore_errors=True)


def test_config_security_settings():
    """Test that security config settings are correctly set"""
    print("\n[TEST] Testing security configuration settings...")
    
    try:
        # Check RECOGNITION_THRESHOLD
        if config.RECOGNITION_THRESHOLD != 0.7:
            print(f"✗ RECOGNITION_THRESHOLD is {config.RECOGNITION_THRESHOLD}, expected 0.7")
            return False
        print(f"✓ RECOGNITION_THRESHOLD correctly set to {config.RECOGNITION_THRESHOLD}")
        
        # Check MIN_MATCH_CONFIDENCE
        if config.MIN_MATCH_CONFIDENCE != 0.85:
            print(f"✗ MIN_MATCH_CONFIDENCE is {config.MIN_MATCH_CONFIDENCE}, expected 0.85")
            return False
        print(f"✓ MIN_MATCH_CONFIDENCE correctly set to {config.MIN_MATCH_CONFIDENCE}")
        
        # Check new security settings
        if not hasattr(config, 'REJECT_UNKNOWN_FACES'):
            print("✗ REJECT_UNKNOWN_FACES not found in config")
            return False
        if config.REJECT_UNKNOWN_FACES != True:
            print(f"✗ REJECT_UNKNOWN_FACES is {config.REJECT_UNKNOWN_FACES}, expected True")
            return False
        print(f"✓ REJECT_UNKNOWN_FACES correctly set to {config.REJECT_UNKNOWN_FACES}")
        
        if not hasattr(config, 'REQUIRE_DATABASE_MATCH'):
            print("✗ REQUIRE_DATABASE_MATCH not found in config")
            return False
        if config.REQUIRE_DATABASE_MATCH != True:
            print(f"✗ REQUIRE_DATABASE_MATCH is {config.REQUIRE_DATABASE_MATCH}, expected True")
            return False
        print(f"✓ REQUIRE_DATABASE_MATCH correctly set to {config.REQUIRE_DATABASE_MATCH}")
        
        print("✓ All security configuration settings are correct")
        return True
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_logging():
    """Test enhanced logging with confidence and distance"""
    print("\n[TEST] Testing enhanced logging functionality...")
    
    try:
        from utils import log_access_event
        
        # Clean up any existing log file
        if os.path.exists(config.LOG_FILE_PATH):
            os.remove(config.LOG_FILE_PATH)
        
        # Test new logging format
        log_access_event("TestUser", "GRANTED", confidence=0.95, distance=0.35)
        log_access_event("UNKNOWN", distance=1.5, photo_filename="unknown_test.jpg")
        log_access_event("TestUser", "DENIED - LOW CONFIDENCE", confidence=0.75, distance=0.55)
        
        # Verify log file exists
        if not os.path.exists(config.LOG_FILE_PATH):
            print("✗ Log file not created")
            return False
        
        # Read and verify log contents
        with open(config.LOG_FILE_PATH, 'r') as f:
            log_contents = f.read()
        
        # Check for expected content
        checks = [
            ("GRANTED" in log_contents, "GRANTED event"),
            ("TestUser" in log_contents, "User name"),
            ("Confidence: 95" in log_contents or "Confidence: 0.95" in log_contents, "Confidence value"),
            ("Distance: 0.35" in log_contents or "Distance: 0.3500" in log_contents, "Distance value"),
            ("UNKNOWN" in log_contents, "Unknown person"),
            ("DENIED - LOW CONFIDENCE" in log_contents, "Low confidence denial"),
        ]
        
        all_passed = True
        for check, description in checks:
            if check:
                print(f"✓ Log contains {description}")
            else:
                print(f"✗ Log missing {description}")
                all_passed = False
        
        print("\n--- Log File Contents ---")
        print(log_contents)
        print("-------------------------\n")
        
        # Clean up
        os.remove(config.LOG_FILE_PATH)
        
        return all_passed
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all security tests"""
    print("="*70)
    print("CRITICAL SECURITY VULNERABILITY FIXES - VALIDATION TESTS")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Config Security Settings", test_config_security_settings()))
    results.append(("Enhanced Logging", test_enhanced_logging()))
    results.append(("Database Manager Empty Check", test_database_manager_empty_check()))
    results.append(("Threshold Enforcement", test_threshold_enforcement()))
    results.append(("Unknown Person Rejection", test_unknown_person_rejection()))
    results.append(("Empty Database Rejection", test_empty_database_rejection()))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:35s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*70)
    if all_passed:
        print("✓ ALL SECURITY TESTS PASSED!")
        print("="*70)
        print("\n[SUCCESS] Security vulnerability fixes are working correctly!")
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("="*70)
        print("\n[ERROR] Security vulnerability fixes have issues that need attention!")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
