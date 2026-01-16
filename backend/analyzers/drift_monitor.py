"""
Model Drift Monitor - Track prediction drift over time
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Monitor model predictions over time to detect distribution shift
    
    Tracks:
    - Historical predictions
    - Statistical anomalies
    - Drift magnitude
    """
    
    def __init__(self, history_path: str = "drift_history.json", max_history: int = 1000):
        """
        Initialize drift monitor
        
        Args:
            history_path: Path to save prediction history
            max_history: Maximum number of predictions to keep per model
        """
        self.history_path = history_path
        self.max_history = max_history
        self.history = self._load_history()
        
        logger.info(f"Drift monitor initialized (history_path={history_path})")
    
    def record(self, model_name: str, score: float):
        """
        Record a prediction for drift tracking
        
        Args:
            model_name: Name of the model
            score: Prediction score
        """
        self.history[model_name].append(float(score))
        
        # Trim history if needed
        if len(self.history[model_name]) > self.max_history:
            self.history[model_name] = self.history[model_name][-self.max_history:]
        
        self._save_history()
    
    def analyze_drift(self, model_name: str, current_score: float) -> Dict[str, Any]:
        """
        Analyze drift for a specific model
        
        Args:
            model_name: Name of the model
            current_score: Current prediction score
            
        Returns:
            Drift analysis results
        """
        history = self.history.get(model_name, [])
        
        if len(history) < 10:
            return {
                "status": "insufficient_history",
                "drift_detected": False,
                "history_count": len(history)
            }
        
        # Compute statistics
        arr = np.array(history, dtype=np.float32)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        
        # Z-score
        z_score = (current_score - mean) / (std + 1e-8)
        
        # Drift detection (z-score > 3 is anomalous)
        is_anomaly = abs(z_score) > 3.0
        drift_magnitude = abs(current_score - mean)
        
        return {
            "status": "analyzed",
            "current_score": float(current_score),
            "historical_mean": mean,
            "historical_std": std,
            "z_score": float(z_score),
            "drift_detected": is_anomaly,
            "drift_magnitude": float(drift_magnitude),
            "history_count": len(history)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked models"""
        summary = {}
        
        for model_name, scores in self.history.items():
            if len(scores) > 0:
                arr = np.array(scores, dtype=np.float32)
                summary[model_name] = {
                    "count": len(scores),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr))
                }
        
        return summary
    
    def clear_history(self, model_name: Optional[str] = None):
        """
        Clear history for a model or all models
        
        Args:
            model_name: Model to clear (None = clear all)
        """
        if model_name is None:
            self.history.clear()
            logger.info("Cleared all drift history")
        elif model_name in self.history:
            del self.history[model_name]
            logger.info(f"Cleared drift history for {model_name}")
        
        self._save_history()
    
    def _load_history(self) -> Dict[str, List[float]]:
        """Load history from disk"""
        if Path(self.history_path).exists():
            try:
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded drift history: {len(data)} models")
                    return defaultdict(list, data)
            except Exception as e:
                logger.warning(f"Failed to load drift history: {e}")
        
        return defaultdict(list)
    
    def _save_history(self):
        """Save history to disk"""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(dict(self.history), f)
        except Exception as e:
            logger.warning(f"Failed to save drift history: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get monitor information"""
        return {
            "name": "DriftMonitor",
            "version": "1.0.0",
            "history_path": self.history_path,
            "max_history": self.max_history,
            "tracked_models": list(self.history.keys()),
            "total_predictions": sum(len(h) for h in self.history.values()),
            "description": "Tracks model predictions over time to detect distribution shift"
        }