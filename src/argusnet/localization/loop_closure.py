"""Spatial loop closure detection for ArgusNet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.indexing.keyframes import Keyframe, KeyframeStore

__all__ = ["LoopClosureCandidate", "LoopClosureDetector"]


@dataclass(frozen=True)
class LoopClosureCandidate:
    query_id: str
    match_id: str
    delta_position: np.ndarray  # (3,) relative translation estimate
    delta_yaw: float
    confidence: float  # [0, 1]
    descriptor_similarity: float | None = None  # cosine similarity if descriptors available


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [−1, 1]; returns 0.0 when either vector is near-zero."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class LoopClosureDetector:
    """Detects loop closures by spatial proximity with optional appearance verification.

    Phase 3: adds descriptor-based cosine-similarity gate on top of the
    spatial proximity + temporal separation criteria from Phase 2.

    When descriptors are absent on either keyframe the detector falls back
    gracefully to the spatial-only result (Phase 2 behaviour), so existing
    workflows without descriptors are unaffected.
    """

    def __init__(
        self,
        spatial_threshold_m: float = 15.0,
        min_temporal_separation_s: float = 30.0,
        max_candidates_per_query: int = 3,
        descriptor_threshold: float = 0.75,
    ) -> None:
        self.spatial_threshold_m = spatial_threshold_m
        self.min_temporal_separation_s = min_temporal_separation_s
        self.max_candidates_per_query = max_candidates_per_query
        self.descriptor_threshold = descriptor_threshold

    def detect(
        self,
        query: Keyframe,
        store: KeyframeStore,
    ) -> list[LoopClosureCandidate]:
        """Find loop closure candidates for *query* in *store*.

        Spatial proximity and temporal separation are applied first.  When
        both keyframes carry descriptors the cosine similarity is computed and
        used as an additional gate (``> descriptor_threshold``).  Passing
        candidates have their base confidence boosted by the similarity score.
        """
        candidates = []
        nearby = store.query_spatial(query.position, self.spatial_threshold_m)
        for kf in nearby:
            if kf.keyframe_id == query.keyframe_id:
                continue
            dt = abs(kf.timestamp_s - query.timestamp_s)
            if dt < self.min_temporal_separation_s:
                continue

            q_pos = np.asarray(query.position, dtype=float)
            m_pos = np.asarray(kf.position, dtype=float)
            delta = m_pos - q_pos
            dist = float(np.linalg.norm(delta))
            spatial_confidence = 1.0 - dist / self.spatial_threshold_m

            q_yaw = query.orientation_rad[2] if query.orientation_rad else 0.0
            m_yaw = kf.orientation_rad[2] if kf.orientation_rad else 0.0
            delta_yaw = float(m_yaw - q_yaw)

            # --- Phase 3: descriptor appearance gate ---
            desc_sim: float | None = None
            if query.descriptor is not None and kf.descriptor is not None:
                sim = _cosine_similarity(
                    np.asarray(query.descriptor, dtype=float),
                    np.asarray(kf.descriptor, dtype=float),
                )
                if sim < self.descriptor_threshold:
                    # Appearance verification failed; skip candidate.
                    continue
                desc_sim = sim
                # Blend spatial confidence with descriptor similarity.
                final_confidence = 0.5 * spatial_confidence + 0.5 * sim
            else:
                # No descriptors available — fall back to spatial-only.
                final_confidence = spatial_confidence

            candidates.append(
                LoopClosureCandidate(
                    query_id=query.keyframe_id,
                    match_id=kf.keyframe_id,
                    delta_position=delta,
                    delta_yaw=delta_yaw,
                    confidence=max(0.0, final_confidence),
                    descriptor_similarity=desc_sim,
                )
            )
        candidates.sort(key=lambda c: -c.confidence)
        return candidates[: self.max_candidates_per_query]
