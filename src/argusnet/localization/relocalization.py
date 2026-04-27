"""Map-based relocalization for ArgusNet (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.localization.transforms import SE3
from argusnet.localization.vio import VIOState

try:
    from scipy.optimize import minimize as _scipy_minimize

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

__all__ = ["MapRelocalizor", "RelocalizationResult"]


@dataclass(frozen=True)
class RelocalizationResult:
    success: bool
    corrected_state: VIOState | None
    confidence: float


class MapRelocalizor:
    """Corrects VIO drift by matching sensor footprint against a prior semantic map.

    Phase 4 (current): combines a coarse grid search with optional Nelder-Mead
    simplex refinement for sub-cell alignment accuracy.

    Search strategy
    ---------------
    1. Coarse grid over ``[-search_radius_m, +search_radius_m]`` at
       ``search_steps`` intervals to locate the best starting cell.
    2. If *refine* is ``True`` and scipy is available, launch a Nelder-Mead
       optimisation starting from the best grid point.  The objective is the
       negative score (we minimise).  This recovers sub-cell accuracy without
       the cost of a fine grid.
    """

    def __init__(
        self,
        footprint_radius_m: float = 30.0,
        confidence_threshold: float = 0.6,
        search_radius_m: float = 20.0,
        search_steps: int = 7,
        refine: bool = True,
    ) -> None:
        self.footprint_radius_m = footprint_radius_m
        self.confidence_threshold = confidence_threshold
        self.search_radius_m = search_radius_m
        self.search_steps = search_steps
        self.refine = refine

    def relocalize(
        self,
        state: VIOState,
        semantic_map,  # SemanticMap
    ) -> RelocalizationResult:
        """Attempt to correct *state* using *semantic_map*.

        Returns the original state if confidence is below threshold.
        """
        pos = state.position  # (3,) ENU
        best_offset = np.zeros(2)
        best_score = self._score_alignment(pos[:2], pos[:2], semantic_map)

        # --- Stage 1: coarse grid search ---
        step = self.search_radius_m / max(self.search_steps, 1)
        for dx in np.arange(-self.search_radius_m, self.search_radius_m + step, step):
            for dy in np.arange(-self.search_radius_m, self.search_radius_m + step, step):
                candidate = pos[:2] + np.array([dx, dy])
                score = self._score_alignment(candidate, pos[:2], semantic_map)
                if score > best_score:
                    best_score = score
                    best_offset = np.array([dx, dy])

        # --- Stage 2: Nelder-Mead refinement from best grid point ---
        if self.refine and _HAS_SCIPY and best_score > 0.0:
            x0 = pos[:2] + best_offset

            def _neg_score(xy: np.ndarray) -> float:
                return -self._score_alignment(xy, pos[:2], semantic_map)

            result = _scipy_minimize(
                _neg_score,
                x0,
                method="Nelder-Mead",
                options={"xatol": 0.1, "fatol": 1e-4, "maxiter": 200},
            )
            if result.success and (-result.fun) > best_score:
                refined_offset = result.x - pos[:2]
                best_offset = refined_offset
                best_score = float(-result.fun)

        confidence = float(best_score)
        if confidence < self.confidence_threshold:
            return RelocalizationResult(success=False, corrected_state=None, confidence=confidence)

        corrected_pos = pos.copy()
        corrected_pos[0] += best_offset[0]
        corrected_pos[1] += best_offset[1]
        corrected_state = VIOState(
            pose=SE3(rotation=state.rotation.copy(), translation=corrected_pos),
            velocity=state.velocity.copy(),
            timestamp_s=state.timestamp_s,
            bias_gyro=state.bias_gyro.copy(),
            bias_accel=state.bias_accel.copy(),
            covariance=state.covariance.copy() if state.covariance is not None else None,
            tracked_feature_count=state.tracked_feature_count,
        )
        return RelocalizationResult(
            success=True, corrected_state=corrected_state, confidence=confidence
        )

    def _score_alignment(
        self,
        candidate_xy: np.ndarray,
        original_xy: np.ndarray,
        semantic_map,
    ) -> float:
        """Score a candidate position against the semantic map.

        Higher score = better alignment. Uses label entropy as proxy:
        diverse label distribution = informative region = good anchor.
        """
        r = self.footprint_radius_m
        cell_size = getattr(semantic_map, "cell_size_m", 1.0)
        steps = max(int(r / cell_size), 3)
        label_counts: dict = {}
        total = 0
        for dx in np.linspace(-r, r, steps):
            for dy in np.linspace(-r, r, steps):
                x, y = candidate_xy[0] + dx, candidate_xy[1] + dy
                label = getattr(semantic_map, "label_at", lambda a, b: 0)(x, y)
                label_counts[label] = label_counts.get(label, 0) + 1
                total += 1
        if total == 0:
            return 0.0
        # Shannon entropy normalized to [0,1]
        probs = [c / total for c in label_counts.values()]
        entropy = -sum(p * np.log(p + 1e-12) for p in probs)
        max_entropy = np.log(max(len(label_counts), 1) + 1e-12)
        return float(entropy / (max_entropy + 1e-12))
