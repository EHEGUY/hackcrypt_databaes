"""
Model Ensemble - Deep learning model predictions
NOW WITH XCEPTION MODEL
"""
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoImageProcessor, AutoModelForImageClassification
import gc

from .base_analyzer import BaseAnalyzer
from .xception_analyzer import XceptionAnalyzer


class ModelEnsemble(BaseAnalyzer):
    """
    Ensemble of specialized deepfake detection models
    
    Models:
    - Xception (FaceForensics++)  ← NEW!
    - General detector
    - Face swap specialist
    - GAN detector
    - Synthetic media detector
    """
    
    MODEL_CONFIGS = [
        ("general_detector", "prithivMLmods/Deep-Fake-Detector-v2-Model"),
        ("face_swap_specialist", "prithivMLmods/deepfake-detector-model-v1"),
        ("gan_detector", "prithivMLmods/Deep-Fake-Detector-v2-Model"),
        ("synthetic_media_detector", "Organika/sdxl-detector"),
    ]
    
    def __init__(
        self, 
        device: Optional[str] = None, 
        lazy_load: bool = True,
        xception_model_path="/Users/emaad/Developer/emaad/hyckcrypt/model/xception5o-tensorflow2-default-v1.tar.gz"
    ):
        """
        Initialize model ensemble
        
        Args:
            device: Device to run models on (cuda/cpu)
            lazy_load: Load models on-demand
            xception_model_path: Path to xception_deepfake_image_5o.h5
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lazy_load = lazy_load
        self.models = {}
        
        # Initialize Xception
        self.xception = XceptionAnalyzer(
            model_path=xception_model_path,
            device="gpu" if self.device == "cuda" else "cpu"
        )
        
        if not lazy_load:
            self._load_all_models()
        
        self.logger.info(f"Initialized on {self.device} (lazy_load={lazy_load}, xception={xception_model_path is not None})")
    
    def analyze(self, frames: List[np.ndarray], **kwargs) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """
        Run ensemble prediction on frames
        
        Args:
            frames: List of video frames (BGR)
            
        Returns:
            (model_scores, per_frame_traces)
        """
        if not self._validate_frames(frames):
            return {}, {}
        
        model_scores = {}
        per_frame_traces = {}
        
        # 1. Run Xception first (highest accuracy)
        try:
            xception_avg, xception_traces = self.xception.analyze(frames)
            model_scores["xception"] = xception_avg
            per_frame_traces["xception"] = xception_traces
        except Exception as e:
            self.logger.error(f"Xception failed: {e}")
            model_scores["xception"] = 0.5
            per_frame_traces["xception"] = [None] * len(frames)
        
        # 2. Run other models
        for key, _ in self.MODEL_CONFIGS:
            per_frame_traces[key] = []
        
        for key, name in self.MODEL_CONFIGS:
            try:
                scores, traces = self._predict_single_model(key, name, frames)
                model_scores[key] = scores
                per_frame_traces[key] = traces
                
            except Exception as e:
                self.logger.error(f"Model {key} failed: {e}")
                model_scores[key] = 0.5
                per_frame_traces[key] = [None] * len(frames)
        
        return model_scores, per_frame_traces
    
    def _load_all_models(self):
        """Load all models into memory"""
        for key, name in self.MODEL_CONFIGS:
            self._load_model(key, name)
    
    def _load_model(self, key: str, name: str):
        """Load a single model"""
        if key in self.models:
            return
        
        try:
            self.logger.info(f"Loading {key}...")
            
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            model = AutoModelForImageClassification.from_pretrained(
                name, 
                trust_remote_code=True,
                torch_dtype=dtype
            )
            processor = AutoImageProcessor.from_pretrained(
                name,
                trust_remote_code=True
            )
            
            model.to(self.device).eval()
            
            self.models[key] = {
                "model": model,
                "processor": processor
            }
            
            self.logger.info(f"✓ Loaded {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to load {key}: {e}")
            raise
    
    def _unload_model(self, key: str):
        """Unload a model to free memory"""
        if key in self.models:
            del self.models[key]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            self.logger.debug(f"Unloaded {key}")
    
    @torch.no_grad()
    def _predict_single_model(
        self, 
        key: str, 
        name: str, 
        frames: List[np.ndarray]
    ) -> Tuple[float, List[float]]:
        """
        Run prediction for a single model
        
        Returns:
            (average_score, per_frame_scores)
        """
        # Load model if needed
        if self.lazy_load:
            self._load_model(key, name)
        
        if key not in self.models:
            return 0.5, [0.5] * len(frames)
        
        model_data = self.models[key]
        frame_scores = []
        
        for frame in frames:
            try:
                score = self._predict_frame(frame, model_data)
                frame_scores.append(float(score))
                
            except Exception as e:
                self.logger.warning(f"Frame prediction failed for {key}: {e}")
                frame_scores.append(None)
        
        # Compute average (ignoring None values)
        valid_scores = [s for s in frame_scores if s is not None]
        avg_score = float(np.mean(valid_scores)) if valid_scores else 0.5
        
        # Unload if lazy loading
        if self.lazy_load:
            self._unload_model(key)
        
        return avg_score, frame_scores
    
    def _predict_frame(self, frame: np.ndarray, model_data: Dict) -> float:
        """Predict single frame"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Preprocess
        inputs = model_data["processor"](pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        outputs = model_data["model"](**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        
        # Extract fake probability
        fake_prob = self._extract_fake_prob(probs, model_data["model"].config)
        
        return float(fake_prob)
    
    def _extract_fake_prob(self, probs: np.ndarray, config) -> float:
        """Extract fake probability from model output"""
        try:
            labels = {v: k for k, v in config.id2label.items()}
            
            # Find label containing 'fake', 'deepfake', or 'synthetic'
            for label_text, label_idx in labels.items():
                label_lower = label_text.lower()
                
                if any(x in label_lower for x in ['fake', 'deepfake', 'synthetic']):
                    # Make sure it's not the 'real' class
                    if 'real' not in label_lower:
                        return float(probs[label_idx])
            
            # Fallback: assume last class is fake
            return float(probs[-1])
            
        except Exception as e:
            self.logger.warning(f"Label extraction failed: {e}")
            return 0.5
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        loaded = list(self.models.keys())
        if self.xception.model is not None:
            loaded.append("xception")
        return loaded
    
    def get_info(self) -> Dict[str, Any]:
        """Get ensemble information"""
        models_info = [
            {"key": "xception", "name": "Xception (FaceForensics++)", "loaded": self.xception.model is not None}
        ]
        models_info.extend([
            {"key": key, "name": name, "loaded": key in self.models}
            for key, name in self.MODEL_CONFIGS
        ])
        
        return {
            "name": "ModelEnsemble",
            "version": "2.0.0",
            "device": self.device,
            "lazy_load": self.lazy_load,
            "models": models_info,
            "num_models": len(self.MODEL_CONFIGS) + 1,  # +1 for Xception
            "description": "Ensemble of specialized deepfake detection models including Xception"
        }
    
    def __del__(self):
        """Cleanup"""
        for key in list(self.models.keys()):
            self._unload_model(key)