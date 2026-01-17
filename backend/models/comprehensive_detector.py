"""
Comprehensive Deepfake Detector - With Evidence Convergence
Measures cross-modal alignment instead of fake disagreement
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
from .model_disagreement import compute_evidence_convergence
from config import settings

logger = logging.getLogger(__name__)


class IntegratedDeepfakeDetector:
    """
    Comprehensive deepfake detector with Evidence Convergence Analysis
    
    Features:
    - 5-model ensemble (Xception + 4 specialists)
    - Evidence convergence (not fake disagreement)
    - Per-model normalization
    - Cross-modal correlation analysis
    - Temporal stability measurement
    - All behavioral + signal analysis
    """
    
    def __init__(self, device: Optional[str] = None, lazy_load: bool = True):
        """
        Initialize comprehensive detector
        
        Args:
            device: Device for models (cuda/cpu)
            lazy_load: Load models on-demand
        """
        logger.info("Initializing Comprehensive Deepfake Detector v3.0 (Evidence Convergence)...")
        
        # Initialize model ensemble with Xception
        self.model_ensemble = ModelEnsemble(
            device=device,
            lazy_load=lazy_load,
            xception_model_path=settings.XCEPTION_MODEL_PATH
        )
        
        # Initialize all other components
        self.texture_analyzer = TextureAnalyzer()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.blink_analyzer = BlinkAnalyzer()
        self.av_sync_analyzer = AVSyncAnalyzer()
        self.drift_monitor = DriftMonitor()
        
        logger.info("✓ All components initialized (5 models + convergence analysis)")
    
    def analyze_video(
        self, 
        frames: List[np.ndarray], 
        fps: float = 30.0,
        video_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis with evidence convergence
        
        Args:
            frames: List of video frames (BGR format)
            fps: Video frame rate
            video_path: Path to video (for audio analysis)
            
        Returns:
            Complete analysis results with evidence convergence
        """
        logger.info(f"Starting analysis: {len(frames)} frames @ {fps} FPS")
        
        results = {}
        
        # ========== 1. DEEP LEARNING MODELS ==========
        logger.info("Running 5-model ensemble (including Xception)...")
        model_scores, per_frame_traces = self.model_ensemble.analyze(frames)
        
        # Record for drift monitoring
        for model_name, score in model_scores.items():
            self.drift_monitor.record(model_name, score)
        
        # Analyze drift (provides historical context)
        drift_analysis = {}
        for model_name, score in model_scores.items():
            drift_analysis[model_name] = self.drift_monitor.analyze_drift(model_name, score)
        
        # ========== 2. BEHAVIORAL ANALYSIS ==========
        logger.info("Running behavioral analysis...")
        blink_result = self.blink_analyzer.analyze(frames, fps=fps)
        av_sync_result = self.av_sync_analyzer.analyze(
            frames, fps=fps, video_path=video_path
        )
        
        behavioral_analysis = {
            "blink_detection": blink_result,
            "audio_visual_sync": av_sync_result
        }
        
        # Classify blink evidence direction
        blink_evidence = self._classify_blink_evidence(blink_result)
        behavioral_analysis["blink_evidence_classification"] = blink_evidence
        
        results["behavioral_analysis"] = behavioral_analysis
        
        # ========== 3. TEXTURE ANALYSIS ==========
        logger.info("Running texture analysis...")
        texture_result = self.texture_analyzer.analyze(frames)
        results["texture_analysis"] = texture_result
        
        # ========== 4. FREQUENCY ANALYSIS ==========
        logger.info("Running frequency analysis...")
        frequency_result = self.frequency_analyzer.analyze(frames)
        results["frequency_analysis"] = frequency_result
        
        # ========== 5. TEMPORAL ANALYSIS ==========
        logger.info("Running temporal analysis...")
        temporal_result = self.temporal_analyzer.analyze(frames)
        results["temporal_analysis"] = temporal_result
        
        # ========== 6. CORRELATED SIGNAL ANALYSIS ==========
        # Identify if GAN + frequency + texture correlate
        correlated_signals = self._analyze_correlated_signals(
            model_scores,
            frequency_result,
            texture_result
        )
        results["correlated_signal_analysis"] = correlated_signals
        
        # ========== 7. EVIDENCE CONVERGENCE (REPLACES DISAGREEMENT) ==========
        logger.info("Computing evidence convergence...")
        
        # Extract model histories from drift monitor
        model_histories = {
            model: self.drift_monitor.history.get(model, [])
            for model in model_scores.keys()
        }
        
        # Prepare signal analysis for convergence
        signal_analysis = {
            "behavioral_analysis": behavioral_analysis,
            "texture_analysis": texture_result,
            "frequency_analysis": frequency_result,
            "temporal_analysis": temporal_result
        }
        
        # Prepare temporal data
        temporal_data = {
            "per_frame_traces": per_frame_traces
        }
        
        # Compute evidence convergence
        convergence = compute_evidence_convergence(
            model_predictions=model_scores,
            signal_analysis=signal_analysis,
            temporal_data=temporal_data,
            model_histories=model_histories
        )
        
        results["evidence_convergence"] = convergence
        
        # ========== 8. MODEL SATURATION CHECK ==========
        saturation_flags = self._check_model_saturation(model_scores, per_frame_traces)
        results["model_saturation_flags"] = saturation_flags
        
        # ========== 9. OVERALL ASSESSMENT ==========
        results["model_predictions"] = {
            "raw_scores": model_scores,
            "per_frame_traces": per_frame_traces,
            "drift_analysis": drift_analysis,
            "note": "See evidence_convergence for proper interpretation"
        }
        
        # ========== 10. METADATA ==========
        results["metadata"] = {
            "frames_analyzed": len(frames),
            "video_fps": fps,
            "system_version": "3.0.0",
            "models_used": 5,
            "analysis_type": "evidence_convergence",
            "improvements": [
                "Evidence convergence replaces fake disagreement",
                "Per-model normalization to history",
                "Cross-modal correlation analysis",
                "Temporal stability measurement",
                "Blink evidence classification",
                "Model saturation detection"
            ]
        }
        
        logger.info("✓ Analysis complete with evidence convergence")
        
        return self._sanitize_json(results)
    
    def _classify_blink_evidence(self, blink_result: Dict) -> Dict[str, Any]:
        """
        Classify blink detection as supporting/contradicting/inconclusive
        
        Natural blinking:
        - Frequency: 10-20 blinks/min
        - Eye openness: 0.4-0.7
        """
        if not blink_result.get("available"):
            return {
                "classification": "unavailable",
                "reason": "Blink detection not available"
            }
        
        freq = blink_result.get("blink_frequency", 0.0)
        openness = blink_result.get("eye_openness_score", 0.5)
        
        # Convert to blinks per minute
        blinks_per_min = freq * 60
        
        # Natural range
        if 10 <= blinks_per_min <= 20 and 0.4 <= openness <= 0.7:
            return {
                "classification": "contradicts_manipulation",
                "confidence": "moderate",
                "reason": f"Natural blink pattern ({blinks_per_min:.1f}/min, openness={openness:.2f})",
                "interpretation": "Blink behavior is within natural human range"
            }
        
        # Anomalous
        elif blinks_per_min == 0:
            return {
                "classification": "supports_manipulation",
                "confidence": "strong",
                "reason": "No blinks detected",
                "interpretation": "Absence of blinking suggests synthetic generation"
            }
        
        elif blinks_per_min > 30:
            return {
                "classification": "supports_manipulation",
                "confidence": "moderate",
                "reason": f"Excessive blink rate ({blinks_per_min:.1f}/min)",
                "interpretation": "Unnaturally high blink frequency"
            }
        
        else:
            return {
                "classification": "inconclusive",
                "confidence": "low",
                "reason": f"Borderline metrics (rate={blinks_per_min:.1f}/min, openness={openness:.2f})",
                "interpretation": "Blink behavior neither strongly supports nor contradicts manipulation"
            }
    
    def _analyze_correlated_signals(
        self,
        model_scores: Dict[str, float],
        frequency_result: Dict,
        texture_result: Dict
    ) -> Dict[str, Any]:
        """
        Analyze if GAN detector + frequency + texture correlate
        If they rise together, treat as ONE piece of evidence
        """
        signals = {}
        
        # Extract relevant scores
        if "gan_detector" in model_scores:
            signals["gan_detector"] = model_scores["gan_detector"]
        
        if frequency_result.get("implemented"):
            signals["blockiness"] = frequency_result.get("blockiness_score", 0.5)
            signals["low_freq_anomaly"] = frequency_result.get("low_frequency_anomalies", 0.5)
        
        if texture_result.get("implemented"):
            # Low naturalness = high manipulation evidence
            naturalness = texture_result.get("texture_naturalness_score", 0.5)
            signals["texture_anomaly"] = 1.0 - naturalness
        
        if len(signals) < 2:
            return {
                "available": False,
                "reason": "Insufficient correlated signals"
            }
        
        # Compute correlation
        scores = list(signals.values())
        
        if len(scores) >= 2:
            # Pairwise correlation
            from scipy.stats import pearsonr
            try:
                correlations = []
                signal_names = list(signals.keys())
                
                for i in range(len(scores)):
                    for j in range(i+1, len(scores)):
                        corr, _ = pearsonr([scores[i]], [scores[j]])
                        if not np.isnan(corr):
                            correlations.append({
                                "pair": f"{signal_names[i]}_vs_{signal_names[j]}",
                                "correlation": float(corr)
                            })
                
                avg_correlation = float(np.mean([c["correlation"] for c in correlations]))
                
                # High correlation means they should be treated as one evidence source
                if avg_correlation > 0.7:
                    interpretation = "high_correlation_treat_as_one_evidence"
                    evidence_count = 1
                elif avg_correlation > 0.4:
                    interpretation = "moderate_correlation_partial_overlap"
                    evidence_count = len(signals) * 0.6
                else:
                    interpretation = "low_correlation_independent_evidence"
                    evidence_count = len(signals)
                
                return {
                    "available": True,
                    "signals": signals,
                    "correlations": correlations,
                    "average_correlation": round(avg_correlation, 4),
                    "interpretation": interpretation,
                    "effective_evidence_count": round(evidence_count, 2)
                }
            
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
        
        return {
            "available": False,
            "reason": "Correlation computation failed"
        }
    
    def _check_model_saturation(
        self,
        model_scores: Dict[str, float],
        per_frame_traces: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Detect models with saturated/collapsed outputs
        These should be downweighted
        """
        saturation_flags = {}
        
        for model, score in model_scores.items():
            trace = per_frame_traces.get(model, [])
            
            # Filter valid scores
            valid = [s for s in trace if s is not None]
            
            if len(valid) < 3:
                continue
            
            arr = np.array(valid)
            variance = float(np.var(arr))
            mean = float(np.mean(arr))
            
            # Check saturation
            is_saturated = False
            reason = ""
            
            # Low variance = collapsed
            if variance < 0.001:
                is_saturated = True
                reason = "near_zero_variance"
            
            # Stuck at extremes
            elif mean > 0.95 or mean < 0.05:
                if variance < 0.01:
                    is_saturated = True
                    reason = "saturated_at_extreme"
            
            if is_saturated:
                saturation_flags[model] = {
                    "saturated": True,
                    "reason": reason,
                    "variance": round(variance, 6),
                    "mean": round(mean, 4),
                    "recommendation": "downweight_to_0.1"
                }
        
        return {
            "flags": saturation_flags,
            "count": len(saturation_flags),
            "interpretation": "Saturated models have no discriminative power and should be downweighted"
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
            "name": "IntegratedDeepfakeDetector",
            "version": "3.0.0",
            "architecture": "evidence_convergence",
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
                "evidence_convergence": True,
                "per_model_normalization": True,
                "cross_modal_correlation": True,
                "temporal_stability": True,
                "blink_evidence_classification": True,
                "model_saturation_detection": True
            },
            "improvements": {
                "v3.0": "Evidence convergence replaces fake disagreement",
                "normalization": "Per-model historical normalization",
                "correlation": "Identifies correlated signal groups",
                "classification": "Blink evidence directional classification"
            }
        }
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift monitoring summary"""
        return self.drift_monitor.get_summary()
    
    def clear_drift_history(self, model_name: Optional[str] = None):
        """Clear drift history"""
        self.drift_monitor.clear_history(model_name)