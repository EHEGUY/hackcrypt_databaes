import numpy as np
from typing import List, Optional, Dict
import logging

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


class VideoAggregator:
    """
    Forensic-grade temporal aggregation.

    NO VERDICTS.
    NO CONFIDENCE THEATER.
    PURE MEASUREMENT.
    """

    def __init__(self, smoothing_sigma: float = 2.0):
        self.smoothing_sigma = smoothing_sigma

    def aggregate(
        self,
        probs: List[float],
        quality_scores: Optional[List[float]] = None
    ) -> Dict[str, float]:

        if not probs:
            return self._empty_report()

        scores = np.array(probs, dtype=np.float32)

        # Temporal smoothing (optional, descriptive only)
        smoothed = self._smooth(scores)

        # Central tendency (robust)
        if quality_scores is not None and len(quality_scores) == len(scores):
            w = np.clip(np.array(quality_scores), 0.0, 1.0)
            w = w / (np.sum(w) + 1e-8)
            central = float(np.sum(smoothed * w))
        else:
            central = float(np.median(smoothed))

        # Distribution metrics
        std = float(np.std(scores))
        iqr = float(np.percentile(scores, 75) - np.percentile(scores, 25))
        entropy = self._entropy(scores)

        # Saturation check (model collapse indicator)
        saturation = float(np.mean((scores < 0.05) | (scores > 0.95)))

        # Temporal behavior
        drift = float(np.mean(np.abs(np.diff(smoothed))))
        periodicity = self._weak_periodicity(scores)

        return {
            # Central tendency
            "central_tendency": central,

            # Distribution
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std_dev": std,
            "iqr": iqr,
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "entropy": entropy,

            # Temporal behavior
            "temporal_drift": drift,
            "temporal_instability": std,
            "weak_periodicity_score": periodicity,

            # Model behavior
            "score_saturation_ratio": saturation,

            # Metadata
            "num_frames": int(len(scores))
        }

    # -------------------- helpers --------------------

    def _smooth(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) < 5:
            return scores
        if HAS_SCIPY:
            return gaussian_filter1d(scores, sigma=self.smoothing_sigma)
        return scores  # fallback: no fake smoothing

    def _entropy(self, scores: np.ndarray) -> float:
        hist, _ = np.histogram(scores, bins=10, range=(0, 1), density=True)
        hist = hist + 1e-8
        return float(-np.sum(hist * np.log2(hist)))

    def _weak_periodicity(self, scores: np.ndarray) -> float:
        if len(scores) < 10:
            return 0.0

        s = scores - np.mean(scores)
        ac = np.correlate(s, s, mode="full")
        ac = ac[len(ac)//2:]

        # Normalize
        ac /= (ac[0] + 1e-8)

        # Exclude trivial lag
        return float(np.max(np.abs(ac[2:10])))

    def _empty_report(self) -> Dict[str, float]:
        return {
            "central_tendency": 0.5,
            "mean": 0.5,
            "median": 0.5,
            "std_dev": 0.0,
            "iqr": 0.0,
            "min": 0.5,
            "max": 0.5,
            "entropy": 0.0,
            "temporal_drift": 0.0,
            "temporal_instability": 0.0,
            "weak_periodicity_score": 0.0,
            "score_saturation_ratio": 0.0,
            "num_frames": 0
        }
