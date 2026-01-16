"""
Temporal Analyzer - Frame-to-frame consistency analysis
"""
import numpy as np
import cv2
from typing import Dict, List, Any

from .base_analyzer import BaseAnalyzer


class TemporalAnalyzer(BaseAnalyzer):
    """
    Temporal consistency analysis for frame stability
    
    Analyzes:
    - Frame-to-frame differences
    - Temporal variance
    - Color consistency
    - Motion patterns
    """
    
    def __init__(self):
        """Initialize temporal analyzer"""
        super().__init__()
        self.logger.info("Initialized temporal analyzer")
    
    def analyze(self, frames: List[np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Analyze temporal consistency across frames
        
        Args:
            frames: List of video frames (BGR)
            
        Returns:
            Temporal analysis results
        """
        if not self._validate_frames(frames):
            return {"available": False, "reason": "Invalid frames"}
        
        if len(frames) < 3:
            return {"available": False, "reason": "< 3 frames"}
        
        try:
            metrics = self._compute_temporal_metrics(frames)
            
            result = {
                "available": True,
                "frame_instability": metrics["instability"],
                "temporal_consistency": metrics["consistency"],
                "temporal_variance": metrics["variance"]
            }
            
            return self._sanitize_output(result)
            
        except Exception as e:
            return self._handle_error(e)
    
    def _compute_temporal_metrics(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Compute temporal metrics"""
        diffs = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale and resize for efficiency
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            g1_resized = cv2.resize(g1, (128, 128))
            g2_resized = cv2.resize(g2, (128, 128))
            
            # Compute absolute difference
            diff = np.mean(np.abs(g1_resized.astype(float) - g2_resized.astype(float)))
            diffs.append(diff)
        
        variance = float(np.var(diffs))
        consistency = float(max(0.0, min(1.0, 1.0 - variance / 50)))
        instability = 1.0 - consistency
        
        return {
            "variance": variance,
            "consistency": consistency,
            "instability": instability
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get analyzer information"""
        return {
            "name": "TemporalAnalyzer",
            "version": "1.0.0",
            "metrics": [
                "Frame instability",
                "Temporal consistency",
                "Temporal variance"
            ],
            "description": "Analyzes frame-to-frame consistency to detect temporal artifacts"
        }