"""
Evidence Convergence Analysis - NOT fake disagreement scores
Measures how independent modalities align on manipulation evidence
"""
import numpy as np
from typing import Dict, List, Tuple, Optional,Any
import logging

logger = logging.getLogger(__name__)


class HistoricalNormalizer:
    """
    Normalize model outputs to their own historical behavior
    This prevents "agreement by mutual blindness"
    """
    
    def __init__(self):
        self.history = {}
    
    def normalize(self, model_name: str, score: float, history: List[float]) -> float:
        """
        Normalize score to percentile rank within model's own history
        
        Args:
            model_name: Model identifier
            score: Raw score to normalize
            history: Historical scores for this model
            
        Returns:
            Percentile rank (0-1)
        """
        if not history or len(history) < 5:
            # Insufficient history, return raw score
            return score
        
        arr = np.array(history)
        percentile = np.searchsorted(np.sort(arr), score) / len(arr)
        return float(percentile)
    
    def normalize_batch(self, model_scores: Dict[str, float], 
                       model_histories: Dict[str, List[float]]) -> Dict[str, float]:
        """Normalize all model scores"""
        normalized = {}
        
        for model, score in model_scores.items():
            history = model_histories.get(model, [])
            normalized[model] = self.normalize(model, score, history)
        
        return normalized


def compute_evidence_convergence(
    model_predictions: Dict[str, float],
    signal_analysis: Optional[Dict[str, Any]] = None,
    temporal_data: Optional[Dict[str, Any]] = None,
    model_histories: Optional[Dict[str, List[float]]] = None
) -> Dict:
    """
    Measure evidence convergence across independent modalities
    
    This replaces fake "disagreement" metrics with real forensic reasoning:
    - How many independent modalities detect anomalies?
    - Do correlated signals agree (frequency + GAN + texture)?
    - Is evidence stable across time?
    - What's the overall evidence density?
    
    Args:
        model_predictions: Raw model scores
        signal_analysis: Frequency/texture/temporal signals
        temporal_data: Per-frame data for stability analysis
        model_histories: Historical predictions for normalization
        
    Returns:
        Evidence convergence report (measurements + interpretation)
    """
    
    if not model_predictions or len(model_predictions) < 2:
        return _empty_convergence_report()
    
    # ========== STEP 1: NORMALIZE TO PER-MODEL HISTORY ==========
    normalizer = HistoricalNormalizer()
    
    if model_histories:
        normalized_scores = normalizer.normalize_batch(model_predictions, model_histories)
    else:
        # Fallback: use raw scores but flag lack of normalization
        normalized_scores = model_predictions.copy()
    
    # ========== STEP 2: IDENTIFY INDEPENDENT MODALITIES ==========
    # These are modalities that measure fundamentally different things
    independent_modalities = _identify_independent_modalities(
        normalized_scores, 
        signal_analysis
    )
    
    # ========== STEP 3: COUNT ACTIVE MODALITIES ==========
    # How many independent modalities show evidence of manipulation?
    activation_threshold = 0.6  # Normalized score above this = "active"
    
    active_modalities = []
    for modality, score in independent_modalities.items():
        if score > activation_threshold:
            active_modalities.append(modality)
    
    convergence_strength = len(active_modalities) / len(independent_modalities)
    
    # ========== STEP 4: MEASURE CROSS-MODAL CORRELATION ==========
    # Do correlated signals agree? (frequency + GAN + texture)
    correlated_groups = _identify_correlated_groups(normalized_scores, signal_analysis)
    group_agreements = _measure_group_agreement(correlated_groups)
    
    # ========== STEP 5: TEMPORAL STABILITY ==========
    # Does evidence remain consistent across frames?
    if temporal_data:
        stability = _measure_temporal_stability(temporal_data)
    else:
        stability = {"status": "unavailable"}
    
    # ========== STEP 6: EVIDENCE DENSITY ==========
    # How many total signals (not just models) indicate manipulation?
    evidence_density = _compute_evidence_density(
        normalized_scores,
        signal_analysis
    )
    
    # ========== STEP 7: COMPUTE OVERALL CONVERGENCE ==========
    return {
        # Raw data
        "normalized_scores": {k: round(float(v), 4) for k, v in normalized_scores.items()},
        "raw_scores": {k: round(float(v), 4) for k, v in model_predictions.items()},
        "normalization_applied": model_histories is not None,
        
        # Independent modality analysis
        "independent_modalities": {
            "identified": list(independent_modalities.keys()),
            "count": len(independent_modalities),
            "active": active_modalities,
            "active_count": len(active_modalities),
            "convergence_strength": round(convergence_strength, 4)
        },
        
        # Correlated signal groups
        "correlated_signal_groups": {
            "groups": correlated_groups,
            "agreements": group_agreements,
            "interpretation": _interpret_group_agreements(group_agreements)
        },
        
        # Temporal stability
        "temporal_stability": stability,
        
        # Evidence density
        "evidence_density": evidence_density,
        
        # Overall assessment
        "convergence_summary": {
            "strength": round(convergence_strength, 4),
            "interpretation": _interpret_convergence_strength(convergence_strength),
            "cross_modal_agreement": _summarize_cross_modal_agreement(group_agreements),
            "temporal_consistency": stability.get("overall_stability", "unknown")
        },
        
        # Guidance
        "forensic_interpretation": {
            "high_convergence": "Multiple independent modalities detect consistent manipulation artifacts",
            "moderate_convergence": "Some modalities detect anomalies, others neutral",
            "low_convergence": "Little agreement across modalities - inconclusive or authentic",
            "current_state": _interpret_convergence_strength(convergence_strength)
        }
    }


def _identify_independent_modalities(
    normalized_scores: Dict[str, float],
    signal_analysis: Optional[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Identify truly independent modalities (not measuring same thing)
    
    Independent modalities:
    - Face-swap specialist (face region artifacts)
    - Xception (should use spikes, but if scalar: localized artifacts)
    - Behavioral signals (blink, AV sync)
    - Frequency domain (DCT/FFT artifacts)
    - Texture (skin patterns)
    
    NOT independent:
    - GAN detector + frequency + texture often correlate
    """
    independent = {}
    
    # Model-based modalities
    if "face_swap_specialist" in normalized_scores:
        # But downweight if saturated
        score = normalized_scores["face_swap_specialist"]
        variance = _estimate_variance(score)
        if variance < 0.01:
            # Saturated model - downweight severely
            independent["face_swap_specialist"] = score * 0.1
        else:
            independent["face_swap_specialist"] = score
    
    if "xception" in normalized_scores:
        # TODO: Should use spike analysis, not scalar
        # For now, include scalar but flag limitation
        independent["xception"] = normalized_scores["xception"]
    
    # Signal-based modalities
    if signal_analysis:
        # Behavioral (truly independent)
        behavioral = signal_analysis.get("behavioral_analysis", {})
        if behavioral:
            blink = behavioral.get("blink_detection", {})
            av_sync = behavioral.get("audio_visual_sync", {})
            
            if blink.get("available"):
                # Convert blink metrics to evidence score
                blink_score = _blink_to_evidence_score(blink)
                independent["behavioral_blink"] = blink_score
            
            if av_sync.get("available"):
                desync = av_sync.get("desync_score", 0.5)
                independent["behavioral_av_sync"] = desync
        
        # Texture (independent from models)
        texture = signal_analysis.get("texture_analysis", {})
        if texture.get("implemented"):
            naturalness = texture.get("texture_naturalness_score", 0.5)
            # Invert: low naturalness = high manipulation evidence
            independent["texture"] = 1.0 - naturalness
    
    return independent


def _identify_correlated_groups(
    normalized_scores: Dict[str, float],
    signal_analysis: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Identify groups of signals that often correlate
    
    Common correlated group:
    - GAN detector + frequency anomalies + texture artifacts
    """
    groups = []
    
    # Group 1: GAN + Frequency + Texture artifacts
    gan_freq_texture = {}
    
    if "gan_detector" in normalized_scores:
        gan_freq_texture["gan_detector"] = normalized_scores["gan_detector"]
    
    if signal_analysis:
        freq = signal_analysis.get("frequency_analysis", {})
        if freq.get("implemented"):
            freq_score = freq.get("frequency_anomaly_score", 0.5)
            gan_freq_texture["frequency_anomalies"] = freq_score
        
        texture = signal_analysis.get("texture_analysis", {})
        if texture.get("implemented"):
            # Use blockiness + low texture as evidence
            blockiness = freq.get("blockiness_score", 0.5)
            gan_freq_texture["blockiness"] = blockiness
    
    if len(gan_freq_texture) >= 2:
        groups.append({
            "name": "gan_frequency_texture",
            "signals": gan_freq_texture,
            "interpretation": "GAN artifacts often correlate with frequency/texture anomalies"
        })
    
    # Group 2: General detector + Synthetic media (similar architectures)
    general_synthetic = {}
    
    if "general_detector" in normalized_scores:
        general_synthetic["general_detector"] = normalized_scores["general_detector"]
    
    if "synthetic_media_detector" in normalized_scores:
        general_synthetic["synthetic_media_detector"] = normalized_scores["synthetic_media_detector"]
    
    if len(general_synthetic) == 2:
        groups.append({
            "name": "general_synthetic",
            "signals": general_synthetic,
            "interpretation": "Similar architecture detectors"
        })
    
    return groups


def _measure_group_agreement(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure whether correlated signals agree"""
    agreements = {}
    
    for group in groups:
        signals = group["signals"]
        if len(signals) < 2:
            continue
        
        scores = list(signals.values())
        
        # Measure variance and correlation
        variance = float(np.var(scores))
        mean = float(np.mean(scores))
        
        # Low variance = agreement
        agreement_score = 1.0 - min(1.0, variance / 0.25)
        
        agreements[group["name"]] = {
            "mean": round(mean, 4),
            "variance": round(variance, 4),
            "agreement_score": round(agreement_score, 4),
            "interpretation": "agree" if agreement_score > 0.7 else "diverge"
        }
    
    return agreements


def _measure_temporal_stability(temporal_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Measure whether evidence is stable across frames
    """
    per_frame_traces = temporal_data.get("per_frame_traces", {})
    
    if not per_frame_traces:
        return {"status": "no_temporal_data"}
    
    stabilities = {}
    
    for model, trace in per_frame_traces.items():
        if not trace or len(trace) < 3:
            continue
        
        # Filter out None values
        valid = [v for v in trace if v is not None]
        if len(valid) < 3:
            continue
        
        arr = np.array(valid)
        variance = float(np.var(arr))
        
        # Low variance = stable evidence
        stability = 1.0 - min(1.0, variance / 0.25)
        stabilities[model] = round(stability, 4)
    
    if stabilities:
        overall = float(np.mean(list(stabilities.values())))
        return {
            "status": "measured",
            "per_model_stability": stabilities,
            "overall_stability": round(overall, 4),
            "interpretation": "stable" if overall > 0.7 else "variable"
        }
    
    return {"status": "insufficient_data"}


def _compute_evidence_density(
    normalized_scores: Dict[str, float],
    signal_analysis: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Count total number of signals indicating manipulation
    """
    threshold = 0.6
    total_signals = 0
    active_signals = 0
    
    # Count model signals
    for score in normalized_scores.values():
        total_signals += 1
        if score > threshold:
            active_signals += 1
    
    # Count non-model signals
    if signal_analysis:
        freq = signal_analysis.get("frequency_analysis", {})
        if freq.get("implemented"):
            total_signals += 3  # blockiness, HF noise, LF anomaly
            if freq.get("blockiness_score", 0) > threshold:
                active_signals += 1
            if freq.get("high_frequency_noise", 0) > threshold:
                active_signals += 1
            if freq.get("low_frequency_anomalies", 0) > threshold:
                active_signals += 1
        
        texture = signal_analysis.get("texture_analysis", {})
        if texture.get("implemented"):
            total_signals += 1
            naturalness = texture.get("texture_naturalness_score", 0.5)
            if naturalness < 0.4:  # Low naturalness = evidence
                active_signals += 1
    
    density = active_signals / total_signals if total_signals > 0 else 0.0
    
    return {
        "total_signals": total_signals,
        "active_signals": active_signals,
        "density": round(density, 4),
        "interpretation": _interpret_evidence_density(density)
    }


def _blink_to_evidence_score(blink_data: Dict) -> float:
    """
    Convert blink metrics to evidence score
    
    Natural blinking:
    - Frequency: 10-20 blinks/min
    - Eye openness: 0.4-0.7
    
    Evidence of manipulation:
    - No blinks (0/min)
    - Too many blinks (>30/min)
    - Unnaturally low eye openness
    """
    freq = blink_data.get("blink_frequency", 0.0)
    openness = blink_data.get("eye_openness_score", 0.5)
    
    # Convert to blinks per minute
    blinks_per_min = freq * 60
    
    # Natural range: 10-20
    if 10 <= blinks_per_min <= 20 and 0.4 <= openness <= 0.7:
        return 0.2  # Low evidence (natural)
    elif blinks_per_min == 0 or blinks_per_min > 30:
        return 0.8  # High evidence (anomalous)
    else:
        return 0.5  # Neutral


def _estimate_variance(score: float) -> float:
    """Estimate variance from single score (placeholder)"""
    # This is a hack - real implementation needs historical data
    return 0.05


def _interpret_convergence_strength(strength: float) -> str:
    """Interpret convergence strength"""
    if strength > 0.7:
        return "high_convergence"
    elif strength > 0.4:
        return "moderate_convergence"
    else:
        return "low_convergence"


def _interpret_group_agreements(agreements: Dict[str, Any]) -> str:
    """Interpret group agreement patterns"""
    if not agreements:
        return "no_correlated_groups"
    
    agree_count = sum(1 for v in agreements.values() if v.get("interpretation") == "agree")
    
    if agree_count == len(agreements):
        return "all_correlated_groups_agree"
    elif agree_count > 0:
        return "partial_agreement_in_correlated_groups"
    else:
        return "correlated_groups_diverge"


def _summarize_cross_modal_agreement(agreements: Dict[str, Any]) -> str:
    """Summarize cross-modal agreement"""
    if not agreements:
        return "unknown"
    
    agree_scores = [v.get("agreement_score", 0.5) for v in agreements.values()]
    avg = np.mean(agree_scores)
    
    if avg > 0.7:
        return "strong"
    elif avg > 0.4:
        return "moderate"
    else:
        return "weak"


def _interpret_evidence_density(density: float) -> str:
    """Interpret evidence density"""
    if density > 0.7:
        return "high_density"
    elif density > 0.4:
        return "moderate_density"
    else:
        return "low_density"


def _empty_convergence_report() -> Dict:
    """Return empty report for insufficient data"""
    return {
        "error": "Insufficient data for convergence analysis",
        "model_count": 0,
        "message": "Need at least 2 models with predictions"
    }


# ========== BACKWARD COMPATIBILITY ==========
# Keep old function name but redirect to new implementation

def compute_model_disagreement(model_predictions: Dict[str, float]) -> Dict:
    """
    DEPRECATED: Use compute_evidence_convergence instead
    
    This function is kept for backward compatibility but now returns
    evidence convergence data instead of fake disagreement metrics.
    """
    logger.warning(
        "compute_model_disagreement is deprecated. "
        "Use compute_evidence_convergence for accurate analysis."
    )
    
    # Return basic convergence without signal analysis
    return compute_evidence_convergence(
        model_predictions=model_predictions,
        signal_analysis=None,
        temporal_data=None,
        model_histories=None
    )