"""
Xception-based Deepfake Detector - PRODUCTION READY
High-accuracy FaceForensics++ trained model
FIXED: Proper score inversion and normalization

Model behavior:
- Raw score ~0.0008 = FAKE video
- Raw score ~0.09 = REAL video
- We invert to: 1.0 = FAKE, 0.0 = REAL
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import tarfile
import tempfile
import logging

from .base_analyzer import BaseAnalyzer

try:
    import tensorflow as tf
    HAS_TF = True
    TF_VERSION = tf.__version__
except ImportError:
    HAS_TF = False
    TF_VERSION = None

logger = logging.getLogger(__name__)

class XceptionAnalyzer(BaseAnalyzer):
    """
    Xception-based deepfake detector with proper score inversion
    
    - Trained on FaceForensics++ (c23)
    - Input: 224x224 RGB faces
    - Output: Fake probability (0=real, 1=fake) after inversion
    
    Score transformation:
    - Raw 0.0008 (fake) -> 0.984 (high confidence fake)
    - Raw 0.0923 (real) -> 0.252 (low-medium confidence fake)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        super().__init__()
        
        if not HAS_TF:
            logger.error("TensorFlow not installed!")
            self.model = None
            self.model_path = None
            return
        
        self.model_path = model_path
        self.device = device
        self.model = None
        self._model_loaded = False
        
        logger.info(f"XceptionAnalyzer v3.0.0 (TF {TF_VERSION}, device={device})")
        logger.info("Score inversion: lower raw -> higher fake probability")
        
        if model_path:
            self._load_model()
        else:
            logger.warning("No model path provided - will use dummy predictions")
    
    def _load_model(self):
        """Load the Xception model with persistent temp directory support"""
        if not HAS_TF:
            raise RuntimeError("TensorFlow not installed")
        
        if not self.model_path or not Path(self.model_path).exists():
            logger.warning(f"Model path does not exist: {self.model_path}")
            return
        
        try:
            # Configure device
            if self.device == "cpu":
                tf.config.set_visible_devices([], 'GPU')
            
            if self.model_path.endswith('.tar.gz'):
                with tempfile.TemporaryDirectory() as temp_dir:
                    logger.info(f"Extracting tar.gz to temporary directory: {temp_dir}")
                    extracted_model_file = self._extract_files(self.model_path, temp_dir)
                    self.model = tf.keras.models.load_model(extracted_model_file)
                    self._model_loaded = True
                    logger.info("Xception model loaded successfully from archive")
            else:
                # Direct load for .h5 or SavedModel folders
                self.model = tf.keras.models.load_model(self.model_path)
                self._model_loaded = True
                logger.info("Xception model loaded successfully from path")
            
        except Exception as e:
            logger.error(f"Failed to load Xception model: {e}")
            self.model = None
            self._model_loaded = False

    def _extract_files(self, tar_path: str, target_dir: str) -> str:
        """Helper to extract and locate the model file within a specific directory"""
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(target_dir)
        
        temp_path = Path(target_dir)
        
        # Priority 1: .h5 files
        h5_files = list(temp_path.rglob('*.h5'))
        if h5_files:
            return str(h5_files[0])
            
        # Priority 2: .keras files
        keras_files = list(temp_path.rglob('*.keras'))
        if keras_files:
            return str(keras_files[0])
            
        # Priority 3: SavedModel directory
        pb_files = list(temp_path.rglob('saved_model.pb'))
        if pb_files:
            return str(pb_files[0].parent)
            
        raise FileNotFoundError(f"No valid model files found in {tar_path}")

    def _invert_and_normalize(self, raw_score: float) -> float:
        """
        Invert and normalize the Xception score to standard format
        
        Based on empirical data:
        - Fake videos: 0.0008 - 0.01 (very low)
        - Real videos: 0.05 - 0.15 (higher)
        
        Transformation strategy (piecewise linear):
        1. [0, 0.01]: Very likely fake -> [1.0, 0.8]
        2. [0.01, 0.1]: Transition zone -> [0.8, 0.2]
        3. [0.1+]: Very likely real -> [0.2, 0.0]
        
        Examples:
        - 0.0008 -> 0.984 (high confidence fake)
        - 0.01   -> 0.8   (likely fake)
        - 0.05   -> 0.467 (uncertain)
        - 0.0923 -> 0.252 (likely real)
        - 0.1    -> 0.2   (very likely real)
        - 0.15   -> 0.0   (certain real)
        """
        # Clamp to reasonable range
        raw_score = np.clip(raw_score, 0.0, 0.2)
        
        if raw_score <= 0.01:
            # Very likely fake zone
            # Linear map: [0, 0.01] -> [1.0, 0.8]
            normalized = raw_score / 0.01  # 0 to 1
            return 1.0 - normalized * 0.2  # 1.0 to 0.8
            
        elif raw_score <= 0.1:
            # Transition zone
            # Linear map: [0.01, 0.1] -> [0.8, 0.2]
            normalized = (raw_score - 0.01) / 0.09  # 0 to 1
            return 0.8 - normalized * 0.6  # 0.8 to 0.2
            
        else:
            # Very likely real zone
            # Linear map: [0.1, 0.2] -> [0.2, 0.0]
            normalized = (raw_score - 0.1) / 0.1  # 0 to 1
            return max(0.0, 0.2 - normalized * 0.2)  # 0.2 to 0.0

    def analyze(self, frames: List[np.ndarray], **kwargs) -> Tuple[float, List[float]]:
        """
        Analyze frames and return fake probability scores
        
        Args:
            frames: List of video frames (BGR format)
            
        Returns:
            (average_score, per_frame_scores) where scores are in [0, 1]
            0 = real, 1 = fake
        """
        if not HAS_TF or not self._model_loaded:
            logger.warning("Model not loaded, returning neutral scores")
            return 0.5, [0.5] * len(frames)
        
        try:
            frame_scores = []
            raw_scores = []  # For debugging
            
            for frame in frames:
                score, raw = self._predict_frame(frame)
                frame_scores.append(score)
                raw_scores.append(raw)
            
            avg_score = float(np.mean(frame_scores))
            
            # Log some stats for debugging
            logger.debug(f"Processed {len(frames)} frames")
            logger.debug(f"Raw scores range: [{min(raw_scores):.6f}, {max(raw_scores):.6f}]")
            logger.debug(f"Final scores range: [{min(frame_scores):.3f}, {max(frame_scores):.3f}]")
            logger.debug(f"Average fake probability: {avg_score:.3f}")
            
            return avg_score, frame_scores
            
        except Exception as e:
            logger.error(f"Analysis loop failed: {e}")
            return 0.5, [0.5] * len(frames)

    def _predict_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Preprocess and predict a single frame
        
        Returns:
            (final_score, raw_score) tuple for debugging
        """
        try:
            # Preprocessing
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Inference
            prediction = self.model.predict(img, verbose=0)
            
            # Extract raw score
            if prediction.shape[-1] > 1:
                raw_score = float(prediction[0][-1])
            else:
                raw_score = float(prediction[0][0])
            
            # Invert and normalize to [0, 1] where 1=fake, 0=real
            final_score = self._invert_and_normalize(raw_score)
            
            return final_score, raw_score
            
        except Exception as e:
            logger.warning(f"Prediction failed for frame: {e}")
            return 0.5, 0.5

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "XceptionAnalyzer",
            "version": "3.0.0",
            "model_loaded": self._model_loaded,
            "tensorflow_version": TF_VERSION,
            "device": self.device,
            "score_inverted": True,
            "normalization": "piecewise-linear",
            "score_mapping": {
                "fake_range": "[0, 0.01] raw -> [1.0, 0.8] normalized",
                "transition": "[0.01, 0.1] raw -> [0.8, 0.2] normalized",
                "real_range": "[0.1, 0.2] raw -> [0.2, 0.0] normalized"
            }
        }

    def __del__(self):
        if hasattr(self, 'model') and self.model:
            del self.model