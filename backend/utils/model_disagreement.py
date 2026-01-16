"""
Model Disagreement - Real measurement of model consensus
NOT fake "agreement scores"
"""
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_model_disagreement(model_predictions: Dict[str, float]) -> Dict:
    """
    Measure how much models disagree with each other.
    
    This is NOT "1.0 - std(scores)". That's fake.
    
    Real disagreement means:
    - Different pairwise differences
    - Models split into consensus groups
    - Outlier models that deviate from median
    - High variance in predictions
    
    Args:
        model_predictions: Dict[model_name -> float score]
        
    Returns:
        Structured disagreement report (measurements only, no verdicts)
    """
    
    if not model_predictions or len(model_predictions) < 2:
        return _empty_report()
    
    models = list(model_predictions.keys())
    scores = np.array(list(model_predictions.values()), dtype=np.float32)
    
    # ========== PAIRWISE DIFFERENCES ==========
    # How much does each pair of models differ?
    pairwise = {}
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            diff = abs(model_predictions[m1] - model_predictions[m2])
            pairwise[f"{m1}_vs_{m2}"] = round(float(diff), 4)
    
    max_pairwise = max(pairwise.values()) if pairwise else 0.0
    avg_pairwise = np.mean(list(pairwise.values())) if pairwise else 0.0
    
    # ========== RANKING ==========
    # In what order do models rank the content?
    ranking = sorted(model_predictions.items(), key=lambda x: x[1], reverse=True)
    
    # ========== DISTRIBUTION STATS ==========
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_range = score_max - score_min
    score_mean = float(np.mean(scores))
    score_median = float(np.median(scores))
    score_std = float(np.std(scores))
    score_iqr = float(np.percentile(scores, 75) - np.percentile(scores, 25))
    
    # ========== CONSENSUS GROUPS ==========
    # Which models agree with each other?
    consensus_groups = _find_consensus_groups(model_predictions, threshold=0.1)
    
    # ========== OUTLIERS ==========
    # Which models deviate from the median?
    outliers = _find_outliers(model_predictions, median=score_median, threshold=0.3)
    
    # ========== DISAGREEMENT METRICS ==========
    # Pure measurements, no interpretation
    return {
        # Individual scores
        "model_scores": {k: round(float(v), 4) for k, v in model_predictions.items()},
        
        # Pairwise disagreement
        "pairwise_analysis": {
            "differences": pairwise,
            "max_difference": round(max_pairwise, 4),
            "avg_difference": round(avg_pairwise, 4),
            "note": "Larger differences mean models fundamentally disagree"
        },
        
        # Ranking (relative order)
        "ranking_by_score": {
            "order": [(m, round(float(s), 4)) for m, s in ranking],
            "note": "Check if models agree on relative ranking"
        },
        
        # Distribution statistics
        "distribution": {
            "min": score_min,
            "max": score_max,
            "range": round(score_range, 4),
            "mean": round(score_mean, 4),
            "median": round(score_median, 4),
            "std_dev": round(score_std, 4),
            "iqr": round(score_iqr, 4),
            "note": "High std_dev or IQR indicates models spread across range"
        },
        
        # Consensus groups
        "consensus_groups": {
            "groups": consensus_groups,
            "num_groups": len(consensus_groups),
            "interpretation": _interpret_consensus_groups(consensus_groups, len(models))
        },
        
        # Outliers (models that deviate)
        "outliers": {
            "models": outliers,
            "count": len(outliers),
            "note": "Models > 0.3 away from median"
        },
        
        # Summary metrics (data-driven, not semantic)
        "summary_metrics": {
            "num_models": len(models),
            "max_pairwise_difference": round(max_pairwise, 4),
            "avg_pairwise_difference": round(avg_pairwise, 4),
            "score_std_dev": round(score_std, 4),
            "outlier_count": len(outliers)
        },
        
        # Guidance (still not a verdict)
        "interpretation_guide": {
            "low_disagreement": "std < 0.1: Models closely agree",
            "moderate_disagreement": "std 0.1-0.3: Models differ but trend same direction",
            "high_disagreement": "std > 0.3: Models fundamentally disagree",
            "note": "Disagreement is measurement, not verdict. Examine why models differ."
        }
    }


def _find_consensus_groups(model_preds: Dict[str, float], threshold: float = 0.1) -> List[Dict]:
    """
    Group models with similar scores.
    
    Args:
        model_preds: Model predictions
        threshold: Maximum difference within group
        
    Returns:
        List of consensus groups
    """
    if not model_preds:
        return []
    
    # Sort by score
    sorted_items = sorted(model_preds.items(), key=lambda x: x[1])
    
    groups = []
    current_group = [sorted_items[0]]
    
    for i in range(1, len(sorted_items)):
        prev_score = current_group[-1][1]
        curr_model, curr_score = sorted_items[i]
        
        if abs(curr_score - prev_score) <= threshold:
            current_group.append((curr_model, curr_score))
        else:
            groups.append(_format_group(current_group))
            current_group = [(curr_model, curr_score)]
    
    if current_group:
        groups.append(_format_group(current_group))
    
    return groups


def _format_group(group: List[Tuple[str, float]]) -> Dict:
    """Format a single consensus group"""
    models = [m for m, _ in group]
    scores = [s for _, s in group]
    
    return {
        "models": models,
        "count": len(models),
        "score_range": [round(min(scores), 4), round(max(scores), 4)],
        "avg_score": round(float(np.mean(scores)), 4)
    }


def _find_outliers(model_preds: Dict[str, float], median: float, threshold: float = 0.3) -> Dict:
    """
    Find models that deviate from median by > threshold.
    
    Args:
        model_preds: Model predictions
        median: Median score across models
        threshold: Deviation threshold
        
    Returns:
        Dict of outlier models
    """
    outliers = {}
    
    for model, score in model_preds.items():
        deviation = abs(score - median)
        if deviation > threshold:
            outliers[model] = {
                "score": round(float(score), 4),
                "median": round(float(median), 4),
                "deviation": round(float(deviation), 4),
                "direction": "above" if score > median else "below"
            }
    
    return outliers


def _interpret_consensus_groups(groups: List[Dict], total_models: int) -> str:
    """Describe consensus group structure (measurement only)"""
    if len(groups) == 1:
        return "All models in single consensus group (high agreement)"
    elif len(groups) == total_models:
        return "Each model in separate group (no agreement)"
    else:
        avg_group_size = total_models / len(groups)
        return f"{len(groups)} consensus groups, avg {avg_group_size:.1f} models per group"


def _empty_report() -> Dict:
    """Return empty report for insufficient data"""
    return {
        "error": "Insufficient models for disagreement analysis",
        "model_count": 0,
        "message": "Need at least 2 distinct models"
    }


def compute_frame_level_disagreement(frame_predictions: Dict[str, List[float]]) -> Dict:
    """
    Measure disagreement across ALL frames (not per-model average).
    
    This shows if models disagree consistently or if disagreement changes per frame.
    
    Args:
        frame_predictions: Dict[model -> list of per-frame scores]
        
    Returns:
        Frame-level disagreement trends
    """
    if not frame_predictions:
        return {"error": "No frame predictions"}
    
    models = list(frame_predictions.keys())
    num_frames = len(next(iter(frame_predictions.values())))
    
    if len(models) < 2:
        return {"error": "Need at least 2 models"}
    
    # Compute disagreement per frame
    frame_disagreements = []
    
    for frame_idx in range(num_frames):
        frame_scores = {}
        for model in models:
            score = frame_predictions[model][frame_idx]
            if score is not None:
                frame_scores[model] = score
        
        if len(frame_scores) >= 2:
            frame_dis = compute_model_disagreement(frame_scores)
            max_diff = frame_dis.get("pairwise_analysis", {}).get("max_difference", 0.0)
            frame_disagreements.append(float(max_diff))
    
    if not frame_disagreements:
        return {"error": "No valid frame disagreement data"}
    
    frame_disagreements = np.array(frame_disagreements)
    
    return {
        "temporal_disagreement_trend": {
            "mean_disagreement": round(float(np.mean(frame_disagreements)), 4),
            "max_disagreement": round(float(np.max(frame_disagreements)), 4),
            "min_disagreement": round(float(np.min(frame_disagreements)), 4),
            "std_disagreement": round(float(np.std(frame_disagreements)), 4)
        },
        "frames_with_high_disagreement": int(sum(1 for d in frame_disagreements if d > 0.4)),
        "num_frames_analyzed": int(len(frame_disagreements)),
        "interpretation": {
            "stable_disagreement": np.std(frame_disagreements) < 0.1,
            "note": "If disagreement is stable, models consistently agree/disagree. If variable, disagreement changes per frame."
        }
    }