"""
Frequency Analyzer - FFT/DCT compression artifact detection
FIXED: Handles odd-sized images correctly
"""
import numpy as np
import cv2
from typing import Dict, List, Any

from .base_analyzer import BaseAnalyzer


class FrequencyAnalyzer(BaseAnalyzer):
    """
    Frequency domain analysis for compression artifacts and GAN fingerprints
    
    Analyzes:
    - DCT blockiness (JPEG artifacts)
    - High-frequency noise (GAN patterns)
    - Low-frequency anomalies
    - DCT coefficient distribution
    """
    
    def __init__(self):
        """Initialize frequency analyzer"""
        super().__init__()
        self.logger.info("Initialized frequency analyzer")
    
    def analyze(self, frames: List[np.ndarray], max_frames: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Analyze frequency domain characteristics
        
        Args:
            frames: List of video frames (BGR)
            max_frames: Maximum frames to analyze
            
        Returns:
            Frequency analysis results
        """
        if not self._validate_frames(frames):
            return {"error": "Invalid frames", "implemented": False}
        
        try:
            metrics = self._compute_frequency_metrics(frames[:max_frames])
            anomaly_score = self._compute_anomaly_score(metrics)
            anomalies = self._detect_anomalies(metrics)
            
            result = {
                "implemented": True,
                "blockiness_score": metrics["blockiness"],
                "high_frequency_noise": metrics["hf_noise"],
                "low_frequency_anomalies": metrics["lf_anomaly"],
                "dct_artifact_score": metrics["dct_artifact"],
                "frequency_anomaly_score": anomaly_score,
                "frames_analyzed": metrics["frame_count"],
                "anomaly_flags": anomalies
            }
            
            return self._sanitize_output(result)
            
        except Exception as e:
            return self._handle_error(e)
    
    def _compute_frequency_metrics(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Compute frequency domain metrics"""
        blockiness_scores = []
        hf_noises = []
        lf_anomalies = []
        dct_artifacts = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # FIX: Ensure even dimensions for DCT
            h, w = gray.shape
            if h % 2 != 0:
                gray = gray[:h-1, :]
            if w % 2 != 0:
                gray = gray[:, :w-1]
            
            # Now DCT will work
            dct = cv2.dct(gray)
            blockiness_scores.append(self._measure_blockiness(dct))
            dct_artifacts.append(self._measure_dct_anomaly(dct))
            
            # FFT analysis
            fft = np.fft.fft2(gray)
            magnitude = np.abs(np.fft.fftshift(fft))
            hf_noises.append(self._measure_high_freq(magnitude))
            lf_anomalies.append(self._measure_low_freq(magnitude))
        
        return {
            "blockiness": float(np.mean(blockiness_scores)),
            "hf_noise": float(np.mean(hf_noises)),
            "lf_anomaly": float(np.mean(lf_anomalies)),
            "dct_artifact": float(np.mean(dct_artifacts)),
            "frame_count": len(frames)
        }
    
    def _measure_blockiness(self, dct: np.ndarray) -> float:
        """Measure 8x8 block artifacts (JPEG compression)"""
        h, w = dct.shape
        block_energy = 0.0
        count = 0
        
        # Sample block boundaries
        for i in range(8, h, 8):
            for j in range(8, w, 8):
                if i < h and j < w:
                    block_energy += abs(dct[i, j])
                    count += 1
        
        if count == 0:
            return 0.0
        
        total_energy = np.mean(np.abs(dct))
        return float(min(1.0, (block_energy / count) / (total_energy + 1e-8)))
    
    def _measure_high_freq(self, magnitude: np.ndarray) -> float:
        """Measure high-frequency energy (GAN fingerprints)"""
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # High-frequency region (outer 35%)
        outer_radius = min(h, w) * 0.35
        mask = ((y - cy)**2 + (x - cx)**2) > outer_radius**2
        
        hf_energy = np.sum(magnitude[mask])
        total = np.sum(magnitude)
        return float(hf_energy / (total + 1e-8))
    
    def _measure_low_freq(self, magnitude: np.ndarray) -> float:
        """Measure low-frequency anomalies (DC component)"""
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        
        # DC region (center 10x10)
        dc_region = magnitude[cy-5:cy+5, cx-5:cx+5]
        dc_energy = np.sum(dc_region)
        total = np.sum(magnitude)
        ratio = dc_energy / (total + 1e-8)
        
        # Deviation from expected ratio (~0.5)
        deviation = abs(ratio - 0.5) / 0.5
        return float(min(1.0, deviation))
    
    def _measure_dct_anomaly(self, dct: np.ndarray) -> float:
        """Measure DCT coefficient concentration"""
        flat = np.abs(dct).ravel()
        sorted_dct = np.sort(flat)[::-1]
        
        # Energy concentration in top 10%
        top10_energy = np.sum(sorted_dct[:int(len(sorted_dct) * 0.1)])
        total_energy = np.sum(sorted_dct)
        concentration = top10_energy / (total_energy + 1e-8)
        
        # Natural images have concentration ~0.7-0.9
        return 0.2 if 0.7 <= concentration <= 0.9 else 0.8
    
    def _compute_anomaly_score(self, metrics: Dict[str, float]) -> float:
        """Compute overall frequency anomaly score"""
        scores = [
            metrics["blockiness"],
            metrics["hf_noise"],
            metrics["lf_anomaly"]
        ]
        return float(np.mean(scores))
    
    def _detect_anomalies(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Detect frequency-based anomalies"""
        return {
            "heavy_compression": metrics["blockiness"] > 0.7,
            "gan_artifacts": metrics["hf_noise"] > 0.7,
            "frequency_manipulation": metrics["lf_anomaly"] > 0.6
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get analyzer information"""
        return {
            "name": "FrequencyAnalyzer",
            "version": "1.0.1",
            "methods": ["FFT", "DCT"],
            "metrics": [
                "Blockiness (JPEG artifacts)",
                "High-frequency noise",
                "Low-frequency anomalies",
                "DCT coefficient distribution"
            ],
            "description": "Analyzes frequency domain to detect compression artifacts and GAN fingerprints"
        }