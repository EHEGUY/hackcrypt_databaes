"""
Texture Analyzer - LBP-based skin texture analysis
"""
import numpy as np
import cv2
from typing import Dict, List, Any
from skimage.feature import local_binary_pattern

from .base_analyzer import BaseAnalyzer


class TextureAnalyzer(BaseAnalyzer):
    """
    Local Binary Pattern (LBP) texture analysis for skin authenticity
    
    Analyzes:
    - Texture entropy (complexity)
    - Edge density (pore visibility)
    - High-frequency energy (fine details)
    - Naturalness scoring
    """
    
    def __init__(self, radius: int = 1, points: int = 8):
        """
        Initialize texture analyzer
        
        Args:
            radius: LBP radius
            points: Number of LBP points
        """
        super().__init__()
        self.radius = radius
        self.points = points
        self.logger.info(f"Initialized: radius={radius}, points={points}")
    
    def analyze(self, frames: List[np.ndarray], max_frames: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Analyze texture patterns in frames
        
        Args:
            frames: List of video frames (BGR)
            max_frames: Maximum frames to analyze
            
        Returns:
            Texture analysis results
        """
        if not self._validate_frames(frames):
            return {"error": "Invalid frames", "implemented": False}
        
        try:
            metrics = self._compute_texture_metrics(frames[:max_frames])
            naturalness = self._compute_naturalness(metrics)
            anomalies = self._detect_anomalies(metrics)
            
            result = {
                "implemented": True,
                "lbp_texture_entropy": metrics["lbp_entropy"],
                "edge_density": metrics["edge_density"],
                "high_frequency_energy": metrics["hf_energy"],
                "texture_naturalness_score": naturalness,
                "pore_visibility": metrics["edge_density"],
                "frames_analyzed": metrics["frame_count"],
                "anomaly_flags": anomalies
            }
            
            return self._sanitize_output(result)
            
        except Exception as e:
            return self._handle_error(e)
    
    def _compute_texture_metrics(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Compute raw texture metrics"""
        lbp_entropies = []
        edge_densities = []
        hf_energies = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # LBP entropy
            lbp = local_binary_pattern(gray, self.points, self.radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=self.points + 2, range=(0, self.points + 2))
            hist = hist.astype(float) / (hist.sum() + 1e-8)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            lbp_entropies.append(float(entropy))
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_densities.append(float(edge_density))
            
            # High-frequency energy
            hf_energy = self._compute_hf_energy(gray)
            hf_energies.append(hf_energy)
        
        return {
            "lbp_entropy": float(np.mean(lbp_entropies)),
            "edge_density": float(np.mean(edge_densities)),
            "hf_energy": float(np.mean(hf_energies)),
            "frame_count": len(frames)
        }
    
    def _compute_hf_energy(self, gray: np.ndarray) -> float:
        """Compute high-frequency energy using FFT"""
        dft = np.fft.fft2(gray)
        magnitude = np.abs(np.fft.fftshift(dft))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # Create high-frequency mask
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        hf_mask = dist > min(h, w) * 0.3
        
        hf_energy = np.sum(magnitude[hf_mask]) / (np.sum(magnitude) + 1e-8)
        return float(hf_energy)
    
    def _compute_naturalness(self, metrics: Dict[str, float]) -> float:
        """
        Compute naturalness score from metrics
        
        Natural skin should have:
        - Moderate LBP entropy (~5.0)
        - Visible pores/edges (~0.2)
        - Adequate high-frequency content (~0.4)
        """
        lbp_score = 1.0 - min(1.0, abs(metrics["lbp_entropy"] - 5.0) / 5.0)
        edge_score = 1.0 - min(1.0, abs(metrics["edge_density"] - 0.2) / 0.3)
        hf_score = 1.0 - min(1.0, abs(metrics["hf_energy"] - 0.4) / 0.5)
        
        naturalness = (lbp_score + edge_score + hf_score) / 3.0
        return float(np.clip(naturalness, 0.0, 1.0))
    
    def _detect_anomalies(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Detect texture anomalies"""
        return {
            "over_smoothed": metrics["lbp_entropy"] < 3.5 and metrics["edge_density"] < 0.05,
            "low_texture_complexity": metrics["lbp_entropy"] < 3.0,
            "missing_fine_details": metrics["edge_density"] < 0.05
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get analyzer information"""
        return {
            "name": "TextureAnalyzer",
            "version": "1.0.0",
            "method": "Local Binary Pattern (LBP)",
            "parameters": {
                "radius": self.radius,
                "points": self.points
            },
            "metrics": [
                "LBP entropy",
                "Edge density",
                "High-frequency energy",
                "Naturalness score"
            ],
            "description": "Analyzes skin texture patterns to detect over-smoothing and synthetic artifacts"
        }