"""
Complete 11-Stage Face Recognition Pipeline
Production-grade pipeline integrating all components
"""

import cv2
import numpy as np
from .frame_preprocessor import FramePreprocessor
from .multi_model_detector import MultiModelFaceDetector
from .face_tracker import FaceTracker
from src.quality.face_quality_checker import FaceQualityChecker
from src.security.liveness_detector import LivenessDetector
from src.core.face_aligner import FaceAligner
from src.security.face_occlusion_detector import FaceOcclusionDetector
from .face_enhancement import FaceEnhancer
from .multi_embeddings import MultiEmbeddingGenerator
from .advanced_matcher import AdvancedMatcher
from .post_processor import PostProcessor
from config import config


class CompleteFaceRecognitionPipeline:
    """
    Production-grade 11-stage face recognition pipeline
    
    Architecture:
    1. Frame Preprocessing
    2. Multi-Model Face Detection
    3. Face Tracking (Temporal Consistency)
    4. Face Quality Assessment
    5. Face Anti-Spoofing (Liveness)
    6. Face Alignment & Normalization
    7. Occlusion & Attribute Detection
    8. Face Enhancement
    9. Multi-Face Embeddings
    10. Advanced Matching
    11. Post-Processing & Verification
    """
    
    def __init__(self, face_recognition_model=None, database_manager=None,
                 enable_all_stages=True):
        """
        Initialize the complete pipeline
        
        Args:
            face_recognition_model: Face recognition model for embeddings
            database_manager: Database manager for known faces
            enable_all_stages: Whether to enable all stages (can disable for speed)
        """
        print("[INFO] Initializing Complete 11-Stage Face Recognition Pipeline...")
        
        self.enable_all_stages = enable_all_stages
        self.face_model = face_recognition_model
        self.db_manager = database_manager
        
        # Stage 1: Frame Preprocessing
        if enable_all_stages:
            self.frame_preprocessor = FramePreprocessor()
        else:
            self.frame_preprocessor = None
        
        # Stage 2: Multi-Model Face Detection
        self.face_detector = MultiModelFaceDetector()
        
        # Stage 3: Face Tracking
        self.face_tracker = FaceTracker()
        
        # Stage 4: Face Quality Assessment
        self.quality_checker = FaceQualityChecker()
        
        # Stage 5: Liveness Detection
        if enable_all_stages:
            self.liveness_detector = LivenessDetector()
        else:
            self.liveness_detector = None
        
        # Stage 6: Face Alignment
        self.face_aligner = FaceAligner()
        
        # Stage 7: Occlusion Detection
        self.occlusion_detector = FaceOcclusionDetector()
        
        # Stage 8: Face Enhancement
        if enable_all_stages:
            self.face_enhancer = FaceEnhancer()
        else:
            self.face_enhancer = None
        
        # Stage 9: Multi-Embeddings
        self.embedding_generator = MultiEmbeddingGenerator(
            face_recognition_model=face_recognition_model
        )
        
        # Stage 10: Advanced Matching
        self.matcher = AdvancedMatcher(threshold=config.RECOGNITION_THRESHOLD)
        
        # Stage 11: Post-Processing
        self.post_processor = PostProcessor()
        
        print("[INFO] ✓ Complete Pipeline Initialized Successfully")
        print(f"[INFO] - All stages enabled: {enable_all_stages}")
        print("[INFO] - Ready for production use")
    
    def process_frame(self, frame, mode='full'):
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input camera frame (BGR)
            mode: Processing mode ('full', 'fast', 'quality')
                 - 'full': All 11 stages (most accurate)
                 - 'fast': Skip some stages for speed
                 - 'quality': Focus on quality over speed
            
        Returns:
            Dictionary with recognition results and metadata
        """
        results = {
            'recognized': False,
            'name': None,
            'confidence': 0.0,
            'faces': [],
            'stage_results': {}
        }
        
        try:
            # Stage 1: Frame Preprocessing
            if self.frame_preprocessor and mode in ['full', 'quality']:
                preprocessed_frame = self.frame_preprocessor.preprocess(frame)
                results['stage_results']['preprocessing'] = 'applied'
            else:
                preprocessed_frame = frame
                results['stage_results']['preprocessing'] = 'skipped'
            
            # Stage 2: Multi-Model Face Detection
            detection_mode = 'ensemble' if mode == 'quality' else 'cascade'
            detections = self.face_detector.detect_faces(preprocessed_frame, mode=detection_mode)
            results['stage_results']['detection'] = {
                'count': len(detections),
                'mode': detection_mode
            }
            
            if len(detections) == 0:
                return results
            
            # Stage 3: Face Tracking
            # (Embeddings will be added later in the pipeline)
            tracked_faces = self.face_tracker.update(detections)
            results['stage_results']['tracking'] = {
                'tracked_count': len(tracked_faces)
            }
            
            # Process each detected face
            face_results = []
            
            for face_id, tracked_data in tracked_faces.items():
                face_result = self._process_single_face(
                    frame, tracked_data, mode
                )
                
                if face_result:
                    face_results.append(face_result)
            
            results['faces'] = face_results
            
            # If we have recognized faces, select the best one
            if face_results:
                best_face = max(face_results, key=lambda x: x.get('final_confidence', 0))
                
                if best_face.get('verified', False):
                    results['recognized'] = True
                    results['name'] = best_face.get('name')
                    results['confidence'] = best_face.get('final_confidence', 0)
            
            return results
        
        except Exception as e:
            print(f"[ERROR] Pipeline processing failed: {e}")
            results['error'] = str(e)
            return results
    
    def _process_single_face(self, frame, tracked_data, mode):
        """
        Process a single detected face through the pipeline
        
        Args:
            frame: Original frame
            tracked_data: Tracked face data
            mode: Processing mode
            
        Returns:
            Face processing results dictionary
        """
        face_result = {
            'box': tracked_data['box'],
            'tracking_confidence': tracked_data.get('confidence', 0)
        }
        
        try:
            # Extract face region
            x, y, w, h = tracked_data['box']
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                return None
            
            # Stage 4: Quality Assessment
            quality_checks = self.quality_checker.check_all(face_img)
            quality_score = self.quality_checker.get_quality_score(face_img)
            face_result['quality_score'] = quality_score
            face_result['quality_checks'] = quality_checks
            
            # Skip if quality is too low
            if quality_score < 50 and mode != 'fast':
                face_result['rejected'] = True
                face_result['reason'] = 'Low quality'
                return face_result
            
            # Stage 5: Liveness Detection
            liveness_passed = True
            if self.liveness_detector and mode in ['full', 'quality']:
                # Simple liveness check (frame-based)
                # In production, use multi-frame analysis
                liveness_result = self.liveness_detector.check_liveness(frame, tracked_data['box'])
                liveness_passed = liveness_result.get('is_live', True)
                face_result['liveness'] = liveness_result
            
            if not liveness_passed and mode != 'fast':
                face_result['rejected'] = True
                face_result['reason'] = 'Liveness check failed'
                return face_result
            
            # Stage 6: Face Alignment
            # Create landmarks from keypoints if available
            landmarks = None
            if 'keypoints' in tracked_data and tracked_data['keypoints']:
                landmarks = tracked_data['keypoints']
            
            aligned_face = self.face_aligner.align_face(face_img, landmarks)
            face_result['alignment'] = 'applied'
            
            # Stage 7: Occlusion Detection
            has_occlusion, occlusion_conf, occluded_regions = \
                self.occlusion_detector.detect_occlusion(aligned_face, landmarks)
            
            face_result['occlusion'] = {
                'detected': has_occlusion,
                'confidence': occlusion_conf,
                'regions': occluded_regions
            }
            
            # Skip if heavily occluded
            if has_occlusion and len(occluded_regions) > 1 and mode != 'fast':
                face_result['rejected'] = True
                face_result['reason'] = f'Occlusion detected: {occluded_regions}'
                return face_result
            
            # Stage 8: Face Enhancement
            if self.face_enhancer and mode in ['full', 'quality']:
                enhanced_face = self.face_enhancer.enhance(aligned_face)
                face_result['enhancement'] = 'applied'
            else:
                enhanced_face = aligned_face
                face_result['enhancement'] = 'skipped'
            
            # Stage 9: Generate Embeddings
            embedding_mode = 'ensemble' if mode == 'quality' else 'facenet'
            embedding = self.embedding_generator.generate_embedding(
                enhanced_face, mode=embedding_mode
            )
            
            if embedding is None:
                face_result['rejected'] = True
                face_result['reason'] = 'Embedding generation failed'
                return face_result
            
            face_result['embedding_generated'] = True
            
            # Stage 10: Advanced Matching
            if self.db_manager:
                # Get all known faces from database
                known_embeddings = []
                known_names = []
                
                for user_data in self.db_manager.get_all_users():
                    known_embeddings.append(user_data['embedding'])
                    known_names.append(user_data['name'])
                
                if len(known_embeddings) > 0:
                    # Match against database
                    match_name, match_conf, all_sims = self.matcher.match_embedding(
                        embedding, known_embeddings, known_names, method='hybrid'
                    )
                    
                    face_result['match_name'] = match_name
                    face_result['match_confidence'] = match_conf
                else:
                    match_name = None
                    match_conf = 0.0
            else:
                match_name = None
                match_conf = 0.0
            
            # Stage 11: Post-Processing & Verification
            verified, final_confidence, verification_reason = \
                self.post_processor.verify_recognition(
                    match_name,
                    match_conf,
                    quality_score=quality_score,
                    liveness_passed=liveness_passed,
                    tracking_confidence=tracked_data.get('confidence')
                )
            
            face_result['verified'] = verified
            face_result['final_confidence'] = final_confidence
            face_result['verification_reason'] = verification_reason
            face_result['name'] = match_name if verified else None
            
            # Add to recognition history
            if verified and match_name:
                self.post_processor.add_recognition_result(match_name)
            
            return face_result
        
        except Exception as e:
            print(f"[ERROR] Single face processing failed: {e}")
            face_result['error'] = str(e)
            return face_result
    
    def get_pipeline_stats(self):
        """
        Get pipeline statistics
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            'stages_enabled': 11 if self.enable_all_stages else 8,
            'tracking_active_faces': len(self.face_tracker.tracked_faces),
            'recognition_history_length': len(self.post_processor.recognition_history)
        }
        
        return stats
    
    def reset_pipeline(self):
        """Reset pipeline state (tracking, history, etc.)"""
        self.face_tracker = FaceTracker()
        self.post_processor.clear_history()
        
        if self.liveness_detector:
            # Reset liveness detector if it has state
            pass
        
        print("[INFO] Pipeline state reset")
