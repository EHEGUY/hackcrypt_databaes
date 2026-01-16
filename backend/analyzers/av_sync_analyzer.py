"""
Audio-Visual Sync Analyzer (Lip-Sync Detection)

Uses:
- dlib 68-point facial landmarks
- Normalized mouth opening vs audio energy correlation

Gracefully degrades if model is missing.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import os
import logging

from .base_analyzer import BaseAnalyzer

# Try to import dlib, but allow graceful degradation
try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False

try:
    import librosa
    from scipy.interpolate import interp1d
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

logger = logging.getLogger(__name__)


class AVSyncAnalyzer(BaseAnalyzer):
    """
    Audio-visual synchronization analyzer
    
    Measures lip-sync by correlating mouth opening with audio energy.
    Gracefully degrades if dlib model or audio libraries are missing.
    """

    # Inner mouth landmarks (dlib 68-point model)
    INNER_MOUTH = list(range(60, 68))

    def __init__(self):
        super().__init__()

        self.predictor = None
        self.detector = None
        self.model_loaded = False

        if not HAS_AUDIO:
            self.logger.warning("Audio libraries not available - AV sync will return placeholder data")

        if not HAS_DLIB:
            self.logger.warning("dlib not installed - AV sync will return placeholder data")
            return

        # Find model file
        model_path = self._find_model_file()

        if model_path and os.path.exists(model_path):
            try:
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor(model_path)
                self.model_loaded = True
                self.logger.info("✓ AVSyncAnalyzer initialized with dlib")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load dlib model: {e}\n"
                    f"AV sync will return placeholder data"
                )
        else:
            self.logger.warning(
                f"dlib model not found\n"
                f"Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                f"Extract and place in: model/shape_predictor_68_face_landmarks.dat\n"
                f"AV sync will return placeholder data"
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

    # ---------------------------------------------------------

    def analyze(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
        video_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:

        if not HAS_AUDIO:
            return {
                "available": False,
                "reason": "Audio libraries missing",
                "implemented": True
            }

        if video_path is None:
            return {
                "available": False,
                "reason": "Video path required",
                "implemented": True
            }

        if not self._validate_frames(frames):
            return {
                "available": False,
                "reason": "Invalid frames",
                "implemented": True
            }

        if not self.model_loaded:
            # Return placeholder data
            return self._placeholder_result(frames)

        try:
            audio_env = self._extract_audio_envelope(video_path)
            if audio_env is None or len(audio_env) < 10:
                return {
                    "available": False,
                    "reason": "No usable audio",
                    "implemented": True
                }

            lip_motion = self._extract_lip_motion(frames)
            if len(lip_motion) < 10:
                return {
                    "available": False,
                    "reason": "Insufficient lip data",
                    "implemented": True
                }

            sync_score = self._compute_sync(lip_motion, audio_env)
            if sync_score is None:
                return {
                    "available": False,
                    "reason": "Correlation failed",
                    "implemented": True
                }

            return self._sanitize_output({
                "available": True,
                "implemented": True,
                "sync_score": sync_score,
                "desync_score": 1.0 - sync_score,
                "frames_analyzed": len(frames)
            })

        except Exception as e:
            return self._handle_error(e)

    def _placeholder_result(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Return placeholder result when model is not loaded"""
        return {
            "available": True,
            "implemented": True,
            "sync_score": 0.5,
            "desync_score": 0.5,
            "frames_analyzed": len(frames),
            "note": "dlib model not loaded - returning placeholder data"
        }

    # ---------------------------------------------------------

    def _extract_audio_envelope(self, video_path: str) -> Optional[np.ndarray]:
        try:
            audio, _ = librosa.load(video_path, sr=None, mono=True)
            if len(audio) == 0:
                return None

            stft = librosa.stft(audio)
            return np.abs(stft).mean(axis=0)

        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            return None

    def _extract_lip_motion(self, frames: List[np.ndarray]) -> List[float]:
        motions: List[float] = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)

            if not faces:
                continue

            shape = self.predictor(gray, faces[0])
            mouth = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in self.INNER_MOUTH],
                dtype=np.float32
            )

            # Top inner lip (62, 63)
            top = mouth[[2, 3]].mean(axis=0)

            # Bottom inner lip (66, 67)
            bottom = mouth[[6, 7]].mean(axis=0)

            opening = np.linalg.norm(top - bottom)
            motions.append(opening)

        # Normalize to remove scale dependency
        if len(motions) > 1:
            motions = list((np.array(motions) - np.mean(motions)) /
                           (np.std(motions) + 1e-6))

        return motions

    def _compute_sync(
        self,
        lip_motion: List[float],
        audio_env: np.ndarray
    ) -> Optional[float]:

        try:
            interp = interp1d(
                np.linspace(0, 1, len(audio_env)),
                audio_env,
                fill_value="extrapolate"
            )
            audio_resampled = interp(np.linspace(0, 1, len(lip_motion)))

            corr = np.corrcoef(lip_motion, audio_resampled)[0, 1]
            return None if np.isnan(corr) else abs(float(corr))

        except Exception as e:
            self.logger.error(f"Sync computation failed: {e}")
            return None

    # ---------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "AVSyncAnalyzer",
            "version": "2.2.0",
            "method": "normalized inner-lip motion vs audio envelope correlation",
            "model_loaded": self.model_loaded,
            "dependencies": ["dlib", "librosa", "scipy"],
            "metrics": ["sync_score", "desync_score"]
        }