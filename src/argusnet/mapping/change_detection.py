"""Temporal change detection for ArgusNet semantic maps."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ChangeEvent", "ChangeDetector"]


@dataclass(frozen=True)
class ChangeEvent:
    """A detected semantic change at a map cell."""

    row: int
    col: int
    x_m: float
    y_m: float
    old_label: int
    new_label: int
    confidence: float
    """Confidence of the change detection (0 = uncertain, 1 = certain)."""


class ChangeDetector:
    """Detects label changes between two SemanticMap snapshots.

    Compares dominant labels and confidence thresholds to produce a list
    of ChangeEvent objects representing cells that changed significantly.

    Temporal weighting:
        When *age_s* is passed to :meth:`detect`, the baseline confidence is
        discounted by ``exp(-decay_rate * age_s)`` before being multiplied with
        the current-map confidence.  This prevents stale baselines from
        generating spuriously high-confidence change alerts.

        Default decay_rate of ``1/300`` gives a 5-minute half-life
        (``ln2 / (1/300) ≈ 208 s``).  Set *age_s = 0* (the default) to
        reproduce the original Phase 2 behaviour exactly.
    """

    def __init__(
        self,
        min_confidence_old: float = 0.5,
        min_confidence_new: float = 0.5,
        min_observation_count: int = 3,
        decay_rate: float = 1.0 / 300.0,
    ) -> None:
        self.min_confidence_old = min_confidence_old
        self.min_confidence_new = min_confidence_new
        self.min_observation_count = min_observation_count
        self.decay_rate = decay_rate

    def detect(
        self,
        past_map,
        current_map,
        age_s: float = 0.0,
    ) -> list[ChangeEvent]:
        """Compare *past_map* and *current_map* and return changed cells.

        Both maps must have the same grid dimensions and cell_size_m.
        If they differ, returns an empty list.

        Args:
            past_map: SemanticMap snapshot taken at an earlier time.
            current_map: SemanticMap snapshot taken now.
            age_s: Elapsed time (seconds) between the two snapshots.
                Used to discount the baseline confidence via exponential decay.
                Defaults to 0 (no discounting; identical to Phase 2 behaviour).
        """
        if past_map.rows != current_map.rows or past_map.cols != current_map.cols:
            return []

        # Pre-compute the temporal discount factor once per call.
        age_weight = math.exp(-self.decay_rate * max(0.0, age_s))

        events = []
        for r in range(current_map.rows):
            for c in range(current_map.cols):
                past_cell = past_map._cells[r][c]
                curr_cell = current_map._cells[r][c]

                # Skip cells without enough observations
                if (
                    past_cell.total_observations < self.min_observation_count
                    or curr_cell.total_observations < self.min_observation_count
                ):
                    continue

                past_label = past_cell.dominant_label
                curr_label = curr_cell.dominant_label

                if past_label == curr_label:
                    continue

                past_conf = past_cell.confidence
                curr_conf = curr_cell.confidence

                if past_conf < self.min_confidence_old or curr_conf < self.min_confidence_new:
                    continue

                x, y = current_map.cell_center(r, c)
                # Temporal decay discounts the old-map confidence, so long-stale
                # baselines produce lower change-confidence scores.
                change_confidence = age_weight * past_conf * curr_conf
                events.append(
                    ChangeEvent(
                        row=r,
                        col=c,
                        x_m=x,
                        y_m=y,
                        old_label=past_label,
                        new_label=curr_label,
                        confidence=float(change_confidence),
                    )
                )

        return events

    def events_to_dicts(self, events: list[ChangeEvent]) -> list[dict]:
        """Convert events to JSON-serializable dicts."""
        return [
            {
                "row": e.row,
                "col": e.col,
                "x_m": e.x_m,
                "y_m": e.y_m,
                "old_label": e.old_label,
                "new_label": e.new_label,
                "confidence": e.confidence,
            }
            for e in events
        ]
