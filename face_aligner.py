"""
Face Alignment Module
Aligns faces to a standard template using facial landmarks
"""

import cv2
import numpy as np
import config


class FaceAligner:
    """Face alignment using facial landmarks for consistent face orientation"""
    
    def __init__(self, desired_face_size=None):
        """
        Initialize face aligner
        
        Args:
            desired_face_size: Tuple of (width, height) for aligned face
        """
        if desired_face_size is None:
            self.desired_face_size = config.ALIGNED_FACE_SIZE
        else:
            self.desired_face_size = desired_face_size
        
        # Standard face template (relative positions for 112x112 image)
        # Based on typical face proportions
        self.template = np.array([
            [38.2946, 51.6963],  # Left eye
            [73.5318, 51.5014],  # Right eye
            [56.0252, 71.7366],  # Nose tip
            [41.5493, 92.3655],  # Left mouth corner
            [70.7299, 92.2041]   # Right mouth corner
        ], dtype=np.float32)
        
        # Scale template to desired size
        scale_x = self.desired_face_size[0] / 112.0
        scale_y = self.desired_face_size[1] / 112.0
        self.template[:, 0] *= scale_x
        self.template[:, 1] *= scale_y
        
        print(f"[INFO] Face Aligner initialized (output size: {self.desired_face_size})")
    
    def get_eye_center(self, eye_points):
        """
        Calculate center of eye from landmark points
        
        Args:
            eye_points: Eye landmark coordinates (x, y) or list of points
            
        Returns:
            Center point as numpy array
        """
        if isinstance(eye_points, (list, tuple)):
            if len(eye_points) == 2 and isinstance(eye_points[0], (int, float)):
                # Single point (x, y)
                return np.array(eye_points, dtype=np.float32)
            else:
                # Multiple points - calculate mean
                points = np.array(eye_points, dtype=np.float32)
                return np.mean(points, axis=0)
        else:
            return np.array(eye_points, dtype=np.float32)
    
    def align_face(self, image, landmarks):
        """
        Align face using eye positions and other landmarks
        
        Args:
            image: Face image
            landmarks: Dictionary with facial landmark positions
                      Expected keys: 'left_eye', 'right_eye', and optionally
                      'nose', 'mouth_left', 'mouth_right'
            
        Returns:
            Aligned face image
        """
        if landmarks is None or 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            # If no landmarks, just resize to desired size
            return cv2.resize(image, self.desired_face_size)
        
        try:
            # Extract landmark points
            left_eye = self.get_eye_center(landmarks['left_eye'])
            right_eye = self.get_eye_center(landmarks['right_eye'])
            
            # Build source points array based on available landmarks
            source_points = [left_eye, right_eye]
            
            # Add additional landmarks if available
            if 'nose' in landmarks and landmarks['nose'] is not None:
                nose = np.array(landmarks['nose'], dtype=np.float32)
                source_points.append(nose)
            
            if 'mouth_left' in landmarks and landmarks['mouth_left'] is not None:
                mouth_left = np.array(landmarks['mouth_left'], dtype=np.float32)
                source_points.append(mouth_left)
            
            if 'mouth_right' in landmarks and landmarks['mouth_right'] is not None:
                mouth_right = np.array(landmarks['mouth_right'], dtype=np.float32)
                source_points.append(mouth_right)
            
            source_points = np.array(source_points, dtype=np.float32)
            
            # Use corresponding template points
            template_points = self.template[:len(source_points)]
            
            # Calculate similarity transform
            if len(source_points) >= 3:
                # Use affine transform for 3+ points
                transform_matrix = cv2.estimateAffinePartial2D(
                    source_points, template_points
                )[0]
            else:
                # Use simple rotation/translation for 2 points (eyes only)
                transform_matrix = self._get_simple_transform(
                    left_eye, right_eye,
                    self.template[0], self.template[1]
                )
            
            # Apply transformation
            aligned_face = cv2.warpAffine(
                image,
                transform_matrix,
                self.desired_face_size,
                flags=cv2.INTER_LINEAR
            )
            
            return aligned_face
        
        except Exception as e:
            print(f"[WARNING] Face alignment failed: {e}, using resize")
            # Fallback to simple resize
            return cv2.resize(image, self.desired_face_size)
    
    def _get_simple_transform(self, src_left_eye, src_right_eye, dst_left_eye, dst_right_eye):
        """
        Calculate simple affine transform from eye positions
        
        Args:
            src_left_eye: Source left eye position
            src_right_eye: Source right eye position
            dst_left_eye: Destination left eye position
            dst_right_eye: Destination right eye position
            
        Returns:
            2x3 affine transformation matrix
        """
        # Calculate angle between eyes
        dy = src_right_eye[1] - src_left_eye[1]
        dx = src_right_eye[0] - src_left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate scale
        src_distance = np.linalg.norm(src_right_eye - src_left_eye)
        dst_distance = np.linalg.norm(dst_right_eye - dst_left_eye)
        scale = dst_distance / src_distance
        
        # Calculate center point
        src_center = (src_left_eye + src_right_eye) / 2
        dst_center = (dst_left_eye + dst_right_eye) / 2
        
        # Build transformation matrix
        # First rotate and scale around source center
        M = cv2.getRotationMatrix2D(tuple(src_center), angle, scale)
        
        # Then translate to destination center
        M[0, 2] += dst_center[0] - src_center[0]
        M[1, 2] += dst_center[1] - src_center[1]
        
        return M
    
    def align_to_template(self, face_image, landmarks):
        """
        Align to standard face template (alias for align_face)
        
        Args:
            face_image: Face image
            landmarks: Facial landmarks
            
        Returns:
            Aligned face image
        """
        return self.align_face(face_image, landmarks)
