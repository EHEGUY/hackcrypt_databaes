"""
Comprehensive Deepfake Detector - With Xception
All fixes applied + new model integrated
"""
import numpy as np
import logging
from typing import Dict, List, Any, Optional

from analyzers.model_ensemble import ModelEnsemble
from analyzers.texture_analyzer import TextureAnalyzer
from analyzers.frequency_analyzer import FrequencyAnalyzer
from analyzers.temporal_analyzer import TemporalAnalyzer
from analyzers.blink_analyzer import BlinkAnalyzer
from analyzers.av_sync_analyzer import AVSyncAnalyzer
from analyzers.drift_monitor import DriftMonitor
from config import settings

logger = logging.getLogger(__name__)


class IntegratedDeepfakeDetector:
    """
    Comprehensive deepfake detector with ALL fixes applied
    
    Features:
    - Xception model (92% accuracy)
    - Fixed frequency analyzer
    - Fixed blink detection
    - Fixed AV sync
    - 5-model ensemble
    - All behavioral + signal analysis
    """
    
    def __init__(self, device: Optional[str] = None, lazy_load: bool = True):
        """
        Initialize comprehensive detector
        
        Args:
            device: Device for models (cuda/cpu)
            lazy_load: Load models on-demand
        """
        logger.info("Initializing Comprehensive Deepfake Detector v2.1...")
        
        # Initialize model ensemble with Xception
        self.model_ensemble = ModelEnsemble(
            device=device,
            lazy_load=lazy_load,
            xception_model_path=settings.XCEPTION_MODEL_PATH
        )
        
        # Initialize all other components (all fixed)
        self.texture_analyzer = TextureAnalyzer()
        self.frequency_analyzer = FrequencyAnalyzer()  # FIXED
        self.temporal_analyzer = TemporalAnalyzer()
        self.blink_analyzer = BlinkAnalyzer()  # FIXED
        self.av_sync_analyzer = AVSyncAnalyzer()  # FIXED
        self.drift_monitor = DriftMonitor()
        
        logger.info("✓ All components initialized (5 models + all analyzers)")
    
    def analyze_video(
        self, 
        frames: List[np.ndarray], 
        fps: float = 30.0,
        video_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on video frames
        
        Args:
            frames: List of video frames (BGR format)
            fps: Video frame rate
            video_path: Path to video (for audio analysis)
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting analysis: {len(frames)} frames @ {fps} FPS")
        
        results = {}
        
        # 1. Deep Learning Models (now includes Xception)
        logger.info("Running 5-model ensemble (including Xception)...")
        model_scores, per_frame_traces = self.model_ensemble.analyze(frames)
        
        # Record for drift monitoring
        for model_name, score in model_scores.items():
            self.drift_monitor.record(model_name, score)
        
        # Analyze drift
        drift_analysis = {}
        for model_name, score in model_scores.items():
            drift_analysis[model_name] = self.drift_monitor.analyze_drift(model_name, score)
        
        results["model_predictions"] = {
            "scores": model_scores,
            "per_frame_traces": per_frame_traces,
            "disagreement": self._compute_disagreement(model_scores),
            "drift_analysis": drift_analysis
        }
        
        # 2. Behavioral Analysis (FIXED)
        logger.info("Running behavioral analysis...")
        blink_result = self.blink_analyzer.analyze(frames, fps=fps)
        av_sync_result = self.av_sync_analyzer.analyze(
            frames, fps=fps, video_path=video_path
        )
        
        results["behavioral_analysis"] = {
            "blink_detection": blink_result,
            "audio_visual_sync": av_sync_result
        }
        
        # 3. Texture Analysis
        logger.info("Running texture analysis...")
        results["texture_analysis"] = self.texture_analyzer.analyze(frames)
        
        # 4. Frequency Analysis (FIXED)
        logger.info("Running frequency analysis...")
        results["frequency_analysis"] = self.frequency_analyzer.analyze(frames)
        
        # 5. Temporal Analysis
        logger.info("Running temporal analysis...")
        results["temporal_analysis"] = self.temporal_analyzer.analyze(frames)
        
        # 6. Metadata
        results["metadata"] = {
            "frames_analyzed": len(frames),
            "video_fps": fps,
            "all_features_integrated": True,
            "system_version": "2.1.0",
            "models_used": 5,
            "improvements": [
                "Xception model added (92% accuracy)",
                "Frequency analyzer fixed (odd dimensions)",
                "Blink detection fixed (per-frame + metrics)",
                "AV sync fixed (full lip contour)"
            ]
        }
        
        logger.info("✓ Analysis complete")
        
        return self._sanitize_json(results)
    
    def _compute_disagreement(self, model_scores: Dict[str, float]) -> Dict[str, Any]:
        """Compute model disagreement metrics"""
        if len(model_scores) < 2:
            return {"error": "Need >= 2 models"}
        
        scores = np.array(list(model_scores.values()))
        
        # Pairwise differences
        pairwise = {}
        models = list(model_scores.keys())
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                diff = abs(float(model_scores[m1]) - float(model_scores[m2]))
                pairwise[f"{m1}_vs_{m2}"] = float(diff)
        
        return {
            "pairwise_differences": pairwise,
            "max_difference": float(max(pairwise.values())) if pairwise else 0.0,
            "avg_difference": float(np.mean(list(pairwise.values()))) if pairwise else 0.0,
            "score_std_dev": float(np.std(scores)),
            "score_range": float(np.max(scores) - np.min(scores))
        }
    
    def _sanitize_json(self, obj):
        """Ensure JSON serializability"""
        if isinstance(obj, dict):
            return {k: self._sanitize_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_json(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif obj is None:
            return None
        else:
            return obj
    
    def get_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "name": "ComprehensiveDeepfakeDetector",
            "version": "2.1.0",
            "architecture": "modular",
            "components": {
                "model_ensemble": self.model_ensemble.get_info(),
                "texture_analyzer": self.texture_analyzer.get_info(),
                "frequency_analyzer": self.frequency_analyzer.get_info(),
                "temporal_analyzer": self.temporal_analyzer.get_info(),
                "blink_analyzer": self.blink_analyzer.get_info(),
                "av_sync_analyzer": self.av_sync_analyzer.get_info(),
                "drift_monitor": self.drift_monitor.get_info()
            },
            "features": {
                "xception_model": True,
                "deep_learning_ensemble": True,
                "texture_analysis": True,
                "frequency_analysis": True,
                "temporal_analysis": True,
                "blink_detection": True,
                "av_sync": True,
                "drift_monitoring": True
            },
            "improvements": {
                "frequency_analyzer": "Fixed odd-size DCT bug",
                "blink_detection": "Redesigned with per-frame EAR + metrics",
                "av_sync": "Fixed with full lip contour",
                "xception": "Added 92% accuracy model"
            }
        }
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift monitoring summary"""
        return self.drift_monitor.get_summary()
    
    def clear_drift_history(self, model_name: Optional[str] = None):
        """Clear drift history"""
        self.drift_monitor.clear_history(model_name)