"""
Failure Flag System - Hierarchical, not opinions
CRITICAL > MAJOR > MINOR severity

These are NOT red flags (interpretive).
These ARE system state indicators (factual).
"""
import logging

logger = logging.getLogger(__name__)


# ========== REFERENCE: What belongs at each level ==========
SEVERITY_GUIDE = {
    "critical": "Analysis cannot reliably proceed. Results may be garbage.",
    "major": "Results significantly compromised. Interpretation requires extreme care.",
    "minor": "Reduced functionality. Informational only. Does not block use."
}


def compute_hierarchical_failure_flags(context: dict) -> dict:
    """
    Compute failure flags with explicit severity tiers.
    
    This replaces compute_failure_flags() from original code.
    """
    
    flags = {
        "critical": [],
        "major": [],
        "minor": []
    }
    
    frames_analyzed = context.get("frames_analyzed", 0)
    frames_total = context.get("frames_total", 0)
    fps = context.get("video_fps", 0)
    audio_present = context.get("audio_present", False)
    blink_measurement_available = context.get("blink_measurement_available", False)
    model_predictions = context.get("model_predictions", {})
    
    # ========== CRITICAL FLAGS ==========
    # These mean: "Don't even look at the results without understanding these"
    
    if frames_analyzed == 0:
        flags["critical"].append({
            "flag": "frame_extraction_failed",
            "meaning": "No frames could be extracted from video file",
            "consequence": "All temporal and frame-based analysis is invalid"
        })
    
    if frames_analyzed < 5:
        flags["critical"].append({
            "flag": "insufficient_frames",
            "count": frames_analyzed,
            "threshold": 5,
            "meaning": "Fewer than 5 frames analyzed",
            "consequence": "Temporal analysis impossible, most metrics unreliable"
        })
    
    if not model_predictions or len(model_predictions) == 0:
        flags["critical"].append({
            "flag": "no_model_predictions",
            "meaning": "Models did not produce any scores",
            "consequence": "Cannot interpret detection scores"
        })
    
    # ========== MAJOR FLAGS ==========
    # These mean: "Results exist but are heavily qualified"
    
    if 5 <= frames_analyzed < 20:
        flags["major"].append({
            "flag": "very_few_frames",
            "count": frames_analyzed,
            "meaning": "Only 5-20 frames analyzed",
            "consequence": "Temporal trends not meaningful, all measurements noisy"
        })
    
    if fps < 15:
        flags["major"].append({
            "flag": "very_low_fps",
            "fps": fps,
            "threshold": 15,
            "meaning": f"Video FPS is {fps:.1f}, below 15",
            "consequence": "Temporal consistency unreliable, blink detection impossible"
        })
    
    if fps < 20:
        flags["major"].append({
            "flag": "low_fps",
            "fps": fps,
            "threshold": 20,
            "meaning": f"Video FPS is {fps:.1f}, below 20",
            "consequence": "Blink detection degraded, temporal analysis reduced"
        })
    
    if frames_total > 0:
        face_detection_ratio = frames_analyzed / frames_total
        if face_detection_ratio < 0.5:
            flags["major"].append({
                "flag": "intermittent_face_detection",
                "ratio": round(face_detection_ratio, 2),
                "threshold": 0.5,
                "meaning": f"Face detected in {face_detection_ratio*100:.0f}% of frames (< 50%)",
                "consequence": "Gaps in analysis, no continuity for temporal metrics"
            })
        elif face_detection_ratio < 0.7:
            flags["major"].append({
                "flag": "poor_face_tracking",
                "ratio": round(face_detection_ratio, 2),
                "threshold": 0.7,
                "meaning": f"Face detected in {face_detection_ratio*100:.0f}% of frames (< 70%)",
                "consequence": "Temporal metrics less reliable"
            })
    
    if not audio_present:
        flags["major"].append({
            "flag": "audio_missing",
            "meaning": "No audio track detected in video",
            "consequence": "Audio-visual sync analysis unavailable"
        })
    
    if blink_measurement_available == False and fps >= 20 and frames_analyzed > 30:
        flags["major"].append({
            "flag": "blink_detection_failed_despite_suitable_conditions",
            "fps": fps,
            "frames": frames_analyzed,
            "meaning": "FPS and frame count were adequate, but blink detection failed",
            "consequence": "Face landmarks not detected reliably despite good conditions",
            "implication": "May indicate face size too small or quality issue"
        })
    
    # Model saturation (sign of model collapse)
    if model_predictions:
        scores = list(model_predictions.values())
        saturated_count = sum(1 for s in scores if s < 0.05 or s > 0.95)
        saturation_ratio = saturated_count / len(scores) if len(scores) > 0 else 0
        
        if saturation_ratio > 0.5:
            flags["major"].append({
                "flag": "model_score_saturation",
                "saturated_models": saturated_count,
                "total_models": len(scores),
                "meaning": f"> 50% of models produced extreme scores (0 or 1)",
                "consequence": "Models may have collapsed; disagreement analysis unreliable"
            })
    
    # ========== MINOR FLAGS ==========
    # These mean: "FYI, this feature is not working at full capacity"
    
    if 20 <= frames_analyzed < 30:
        flags["minor"].append({
            "flag": "suboptimal_frame_count",
            "count": frames_analyzed,
            "target": 30,
            "meaning": "20-30 frames analyzed (optimal is 30+)",
            "consequence": "Temporal metrics have higher uncertainty"
        })
    
    if 15 <= fps < 20:
        flags["minor"].append({
            "flag": "low_fps_temporal_degraded",
            "fps": fps,
            "meaning": f"FPS is {fps:.1f} (15-20 range)",
            "consequence": "Temporal analysis quality reduced"
        })
    
    # Implementation status (these are always present)
    flags["minor"].append({
        "flag": "expression_analysis_is_stub",
        "meaning": "Expression analysis uses placeholder implementation",
        "consequence": "Expression scores not meaningful until real FACS implemented"
    })
    
    flags["minor"].append({
        "flag": "skin_texture_analysis_is_stub",
        "meaning": "Skin texture analysis uses placeholder implementation",
        "consequence": "Skin scores not meaningful until real LBP/texture analysis implemented"
    })
    
    flags["minor"].append({
        "flag": "frequency_analysis_is_stub",
        "meaning": "Frequency domain analysis uses placeholder implementation",
        "consequence": "Frequency scores not meaningful until real FFT/DCT analysis implemented"
    })
    
    flags["minor"].append({
        "flag": "lighting_analysis_is_stub",
        "meaning": "Lighting consistency uses placeholder implementation",
        "consequence": "Lighting scores not meaningful until real analysis implemented"
    })
    
    # ========== OVERALL ASSESSMENT ==========
    
    is_critical = len(flags["critical"]) > 0
    is_major = len(flags["major"]) > 0
    
    if is_critical:
        reliability = "severely_compromised"
        action = "DO NOT INTERPRET RESULTS"
    elif is_major:
        reliability = "degraded_but_usable"
        action = "Interpret with extreme caution. Check each major flag."
    else:
        reliability = "nominal"
        action = "Standard interpretation applies. Check minor flags for features not fully implemented."
    
    return {
        "flags_by_severity": flags,
        "reliability_state": reliability,
        "recommended_action": action,
        "critical_count": len(flags["critical"]),
        "major_count": len(flags["major"]),
        "minor_count": len(flags["minor"]),
        "total_count": len(flags["critical"]) + len(flags["major"]) + len(flags["minor"]),
        
        # For easy reference
        "summary": {
            "critical": [f["flag"] for f in flags["critical"]],
            "major": [f["flag"] for f in flags["major"]],
            "minor": [f["flag"] for f in flags["minor"]]
        },
        
        "guidance": {
            "if_critical": "Stop. Results are unreliable. Address critical flags first.",
            "if_major": "Proceed carefully. Every interpretation requires checking relevant major flag.",
            "if_minor_only": "Results valid. Minor flags are informational (mostly stubs and non-critical degradations)."
        }
    }


def get_failure_explanations() -> dict:
    """Get detailed explanations (for documentation/UI)"""
    return {
        # CRITICAL
        "frame_extraction_failed": {
            "severity": "critical",
            "what_it_means": "Video processing could not extract any frames",
            "why_it_matters": "Without frames, there is nothing to analyze",
            "how_to_fix": "Check video file format, codec, corruption"
        },
        "insufficient_frames": {
            "severity": "critical",
            "what_it_means": "Fewer than 5 frames were extracted/detected",
            "why_it_matters": "No meaningful temporal data possible",
            "how_to_fix": "Use longer videos or different face detection settings"
        },
        "no_model_predictions": {
            "severity": "critical",
            "what_it_means": "Models produced no output",
            "why_it_matters": "Cannot determine if content is deepfake-like",
            "how_to_fix": "Check model loading, GPU memory, input format"
        },
        
        # MAJOR
        "very_few_frames": {
            "severity": "major",
            "what_it_means": "Only 5-20 frames analyzed",
            "why_it_matters": "Temporal trends are noise, not signal",
            "how_to_fix": "Use longer videos"
        },
        "very_low_fps": {
            "severity": "major",
            "what_it_means": "Video FPS below 15",
            "why_it_matters": "Cannot detect blinks, temporal analysis collapsed",
            "how_to_fix": "Use 24+ FPS video sources"
        },
        "intermittent_face_detection": {
            "severity": "major",
            "what_it_means": "Face lost for > 50% of video",
            "why_it_matters": "Discontinuous data, temporal analysis meaningless",
            "how_to_fix": "Check video quality, face size, lighting"
        },
        "audio_missing": {
            "severity": "major",
            "what_it_means": "No audio track found",
            "why_it_matters": "Audio-visual sync cannot be measured",
            "how_to_fix": "Use video with audio, or disable audio-sync analysis"
        },
        "model_score_saturation": {
            "severity": "major",
            "what_it_means": "Models all predicted 0 or 1 (collapsed)",
            "why_it_matters": "Models may have failed; disagreement is meaningless",
            "how_to_fix": "Check input preprocessing, model loading"
        },
        
        # MINOR (stubs)
        "expression_analysis_is_stub": {
            "severity": "minor",
            "what_it_means": "Expression scores are placeholder values",
            "why_it_matters": "Facial expression analysis not yet implemented",
            "consequence": "Ignore expression scores for now"
        }
    }


# Backwards compatibility
compute_failure_flags = compute_hierarchical_failure_flags