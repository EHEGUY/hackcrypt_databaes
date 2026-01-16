"""
Xception-based Deepfake Detector - FULLY FIXED
High-accuracy FaceForensics++ trained model
Supports multiple model formats and graceful fallback
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import tarfile
import tempfile
import logging
import shutil

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
    Xception-based deepfake detector
    
    - Trained on FaceForensics++ (c23)
    - Input: 224x224 RGB faces
    - Output: Fake probability
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
        
        logger.info(f"XceptionAnalyzer v1.1.0 (TF {TF_VERSION}, device={device})")
        
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
            
            # FIXED: Use a context manager that wraps the entire loading process
            # This prevents the .h5 file from being deleted before TF can read it.
            if self.model_path.endswith('.tar.gz'):
                with tempfile.TemporaryDirectory() as temp_dir:
                    logger.info(f"Extracting tar.gz to temporary directory: {temp_dir}")
                    extracted_model_file = self._extract_files(self.model_path, temp_dir)
                    self.model = tf.keras.models.load_model(extracted_model_file)
                    self._model_loaded = True
                    logger.info("✓ Xception model loaded successfully from archive")
            else:
                # Direct load for .h5 or SavedModel folders
                self.model = tf.keras.models.load_model(self.model_path)
                self._model_loaded = True
                logger.info("✓ Xception model loaded successfully from path")
            
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

    def analyze(self, frames: List[np.ndarray], **kwargs) -> Tuple[float, List[float]]:
        if not HAS_TF or not self._model_loaded:
            return 0.5, [0.5] * len(frames)
        
        try:
            frame_scores = []
            for frame in frames:
                score = self._predict_frame(frame)
                frame_scores.append(score)
            
            avg_score = float(np.mean(frame_scores))
            return avg_score, frame_scores
        except Exception as e:
            logger.error(f"Analysis loop failed: {e}")
            return 0.5, [0.5] * len(frames)

    def _predict_frame(self, frame: np.ndarray) -> float:
        """Preprocess and predict a single frame"""
        try:
            # Preprocessing
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Inference
            prediction = self.model.predict(img, verbose=0)
            
            # Handle different output shapes (e.g., sigmoid vs softmax)
            if prediction.shape[-1] > 1:
                return float(prediction[0][-1]) # Assume last index is 'fake'
            return float(prediction[0][0])
        except Exception as e:
            logger.warning(f"Prediction failed for frame: {e}")
            return 0.5

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "XceptionAnalyzer",
            "model_loaded": self._model_loaded,
            "tensorflow_version": TF_VERSION,
            "device": self.device
        }

    def __del__(self):
        if hasattr(self, 'model') and self.model:
            del self.model