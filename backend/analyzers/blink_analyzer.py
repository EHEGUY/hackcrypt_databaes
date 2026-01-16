"""
Blink Detection Analyzer - Eye Aspect Ratio based
Returns per-frame scores + summary statistics
Integrates seamlessly with ensemble output format
"""
import cv2
import numpy as np
from scipy.spatial import distance
from typing import Dict, List, Any, Optional
import os
import logging

from analyzers.base_analyzer import BaseAnalyzer

# Try to import dlib, but allow graceful degradation
try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False

logger = logging.getLogger(__name__)


class BlinkAnalyzer(BaseAnalyzer):
    """
    Blink detection using Eye Aspect Ratio (EAR)
    based on dlib 68-point facial landmarks.
    
    Outputs:
    - Per-frame EAR values (normalized)
    - Blink count (frames below threshold)
    - Blink frequency (blinks per second)
    - Overall eye openness metric
    
    Gracefully degrades if model file is missing.
    """

    # Eye landmark indices (dlib 68-point model)
    LEFT_EYE = list(range(42, 48))
    RIGHT_EYE = list(range(36, 42))

    def __init__(self, ear_threshold: float = 0.21, model_path: Optional[str] = None):
        """
        Initialize blink analyzer
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold for blink detection
            model_path: Path to dlib model (if None, searches common locations)
        """
        super().__init__()
        self.ear_threshold = ear_threshold
        self.detector = None
        self.predictor = None
        self.model_loaded = False

        if not HAS_DLIB:
            self.logger.warning("dlib not installed - blink detection will return placeholder data")
            return

        # Find model file
        if model_path is None:
            model_path = self._find_model_file()

        if model_path and os.path.exists(model_path):
            try:
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor(model_path)
                self.model_loaded = True
                self.logger.info(
                    f"✓ BlinkAnalyzer initialized with dlib (threshold={ear_threshold})"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to load dlib model: {e}\n"
                    f"Blink detection will return placeholder data"
                )
        else:
            self.logger.warning(
                f"dlib model not found at {model_path}\n"
                f"Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                f"Extract and place in: models/shape_predictor_68_face_landmarks.dat\n"
                f"Blink detection will return placeholder data"
            )

    def _find_model_file(self) -> Optional[str]:
        """Search common locations for the dlib model"""
        common_paths = [
            "model/shape_predictor_68_face_landmarks.dat",
            "./shape_predictor_68_face_landmarks.dat",
            "shape_predictor_68_face_landmarks.dat",
            os.path.expanduser("~/.insightface/models/shape_predictor_68_face_landmarks.dat"),
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def analyze(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze frames for blink events
        
        Args:
            frames: List of video frames (BGR format)
            fps: Video frame rate (for blink frequency calculation)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with blink analysis results
        """
        if not self._validate_frames(frames):
            return {
                "available": False,
                "reason": "Invalid frames",
                "implemented": True
            }

        if len(frames) < 3:
            return {
                "available": False,
                "reason": "Insufficient frames (need >= 3)",
                "implemented": True
            }

        try:
            if not self.model_loaded:
                # Return placeholder data
                return self._placeholder_result(frames, fps)

            # Extract per-frame EAR values
            ear_values = self._extract_ear_values(frames)

            if len(ear_values) == 0:
                return {
                    "available": False,
                    "reason": "Could not detect faces in any frame",
                    "implemented": True
                }

            # Detect blink events
            blink_frames = self._detect_blinks(ear_values)

            # Compute metrics
            metrics = self._compute_blink_metrics(
                ear_values,
                blink_frames,
                fps
            )

            result = {
                "available": True,
                "implemented": True,
                "per_frame_ear": ear_values,  # Per-frame Eye Aspect Ratio
                "blink_frames": blink_frames,  # Frame indices of blinks
                "blink_count": len(blink_frames),
                "blink_frequency": metrics["blink_frequency"],  # Blinks per second
                "average_ear": metrics["average_ear"],
                "min_ear": metrics["min_ear"],
                "max_ear": metrics["max_ear"],
                "eye_openness_score": metrics["eye_openness"],  # 0-1, higher = more open
                "frames_analyzed": len(ear_values),
                "detection_quality": metrics["quality"]
            }

            return self._sanitize_output(result)

        except Exception as e:
            return self._handle_error(e)

    def _placeholder_result(self, frames: List[np.ndarray], fps: float) -> Dict[str, Any]:
        """
        Return placeholder result when model is not loaded
        This allows the system to continue without the dlib model
        """
        num_frames = len(frames)
        
        # Return neutral/average values
        return {
            "available": True,
            "implemented": True,
            "per_frame_ear": [0.35] * num_frames,  # Neutral open eye value
            "blink_frames": [],
            "blink_count": 0,
            "blink_frequency": 0.0,
            "average_ear": 0.35,
            "min_ear": 0.35,
            "max_ear": 0.35,
            "eye_openness_score": 0.5,
            "frames_analyzed": num_frames,
            "detection_quality": "unavailable",
            "note": "dlib model not loaded - returning placeholder data"
        }

    def _extract_ear_values(self, frames: List[np.ndarray]) -> List[float]:
        """
        Extract Eye Aspect Ratio for each frame
        
        Args:
            frames: List of video frames
            
        Returns:
            List of EAR values (0-1 scale normalized)
        """
        ear_values = []

        for frame in frames:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 0)

                if not faces:
                    # No face detected, use neutral value
                    ear_values.append(None)
                    continue

                # Process first face (strongest detection)
                shape = self.predictor(gray, faces[0])
                coords = np.array(
                    [(shape.part(i).x, shape.part(i).y) for i in range(68)],
                    dtype=np.float32
                )

                # Calculate EAR for both eyes
                left_ear = self._eye_aspect_ratio(coords[self.LEFT_EYE])
                right_ear = self._eye_aspect_ratio(coords[self.RIGHT_EYE])
                ear = (left_ear + right_ear) / 2.0

                ear_values.append(float(ear))

            except Exception as e:
                self.logger.warning(f"Frame processing failed: {e}")
                ear_values.append(None)

        return ear_values

    def _eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio
        
        Based on Soukupová & Tereza (2016)
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        
        Args:
            eye_points: 6 eye landmark points
            
        Returns:
            Eye aspect ratio (float)
        """
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        C = distance.euclidean(eye_points[0], eye_points[3])

        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return float(ear)

    def _detect_blinks(self, ear_values: List[Optional[float]]) -> List[int]:
        """
        Detect blink events from EAR values
        
        A blink is detected when EAR drops below threshold
        
        Args:
            ear_values: Per-frame EAR values
            
        Returns:
            List of frame indices where blinks occur
        """
        blink_frames = []
        in_blink = False

        for idx, ear in enumerate(ear_values):
            if ear is None:
                in_blink = False
                continue

            # Blink starts when EAR drops below threshold
            if ear < self.ear_threshold and not in_blink:
                blink_frames.append(idx)
                in_blink = True

            # Blink ends when EAR rises above threshold
            elif ear >= self.ear_threshold:
                in_blink = False

        return blink_frames

    def _compute_blink_metrics(
        self,
        ear_values: List[Optional[float]],
        blink_frames: List[int],
        fps: float
    ) -> Dict[str, float]:
        """
        Compute summary metrics from EAR and blink data
        
        Args:
            ear_values: Per-frame EAR values
            blink_frames: Frame indices of detected blinks
            fps: Video frame rate
            
        Returns:
            Dictionary of computed metrics
        """
        # Filter out None values
        valid_ears = [e for e in ear_values if e is not None]

        if len(valid_ears) == 0:
            return {
                "average_ear": 0.0,
                "min_ear": 0.0,
                "max_ear": 0.0,
                "blink_frequency": 0.0,
                "eye_openness": 0.5,
                "quality": "no_data"
            }

        # Basic statistics
        avg_ear = float(np.mean(valid_ears))
        min_ear = float(np.min(valid_ears))
        max_ear = float(np.max(valid_ears))

        # Blink frequency (blinks per second)
        duration_seconds = len(ear_values) / fps if fps > 0 else 1.0
        blink_frequency = len(blink_frames) / duration_seconds

        # Eye openness (inverse of how closed they typically are)
        # Normalized: 0 = mostly closed, 1 = mostly open
        # Natural range for open eyes: 0.35-0.50
        eye_openness = np.clip((avg_ear - 0.1) / 0.4, 0.0, 1.0)

        # Quality assessment
        face_detection_ratio = len(valid_ears) / len(ear_values)
        if face_detection_ratio < 0.5:
            quality = "poor"  # Face not detected in many frames
        elif face_detection_ratio < 0.8:
            quality = "fair"
        else:
            quality = "good"

        return {
            "average_ear": avg_ear,
            "min_ear": min_ear,
            "max_ear": max_ear,
            "blink_frequency": float(blink_frequency),
            "eye_openness": float(np.clip(eye_openness, 0.0, 1.0)),
            "quality": quality
        }

    def get_info(self) -> Dict[str, Any]:
        """
        Get analyzer information and capabilities
        
        Returns:
            Dictionary with analyzer metadata
        """
        return {
            "name": "BlinkAnalyzer",
            "version": "2.1.0",
            "method": "Eye Aspect Ratio (EAR)",
            "detector": "dlib 68-point facial landmarks",
            "model_loaded": self.model_loaded,
            "ear_threshold": self.ear_threshold,
            "metrics": [
                "per_frame_ear",
                "blink_count",
                "blink_frequency",
                "average_ear",
                "eye_openness_score",
                "detection_quality"
            ],
            "description": "Detects eye blinks and measures eye openness using Eye Aspect Ratio from facial landmarks"
        }