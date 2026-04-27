"""Semantic segmentation map for ArgusNet.

Per-cell label grid supporting multi-class labels (vegetation, building,
water, road, etc.) with observation-count-based confidence.  Fuses
labels from multiple observations via a simple voting scheme.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SemanticLabel",
    "SemanticCell",
    "SemanticMap",
]


class SemanticLabel(enum.IntEnum):
    """Predefined semantic classes for aerial mapping."""

    UNKNOWN = 0
    VEGETATION = 1
    BUILDING = 2
    WATER = 3
    ROAD = 4
    BARE_GROUND = 5
    VEHICLE = 6
    PERSON = 7
    INFRASTRUCTURE = 8
    SHADOW = 9


@dataclass
class SemanticCell:
    """A single cell storing label votes and confidence."""

    votes: dict[int, float] = field(default_factory=dict)
    """Maps label (int) → observation count (float to support decay)."""

    @property
    def total_observations(self) -> float:
        return sum(self.votes.values())

    @property
    def dominant_label(self) -> int:
        """Return the label with the most votes, or UNKNOWN."""
        if not self.votes:
            return int(SemanticLabel.UNKNOWN)
        return max(self.votes, key=lambda k: self.votes[k])

    @property
    def confidence(self) -> float:
        """Fraction of votes for the dominant label in [0, 1]."""
        total = self.total_observations
        if total == 0:
            return 0.0
        return self.votes.get(self.dominant_label, 0) / total

    def add_vote(self, label: int, count: float = 1.0) -> None:
        self.votes[label] = self.votes.get(label, 0.0) + count

    def label_probability(self, label: int) -> float:
        """Return the fraction of votes for *label*."""
        total = self.total_observations
        if total == 0:
            return 0.0
        return self.votes.get(label, 0) / total

    def label_distribution(self) -> dict[int, float]:
        """Return normalised probability for each observed label."""
        total = self.total_observations
        if total == 0:
            return {}
        return {label: count / total for label, count in self.votes.items()}

    def reset(self) -> None:
        self.votes.clear()


class SemanticMap:
    """Grid-based semantic label map.

    The map covers a 2-D bounding box with a fixed cell size.  Each cell
    accumulates label votes from observations; the dominant label is the
    cell's current classification.

    Args:
        x_min: Left edge of the map (metres).
        x_max: Right edge of the map (metres).
        y_min: Bottom edge of the map (metres).
        y_max: Top edge of the map (metres).
        cell_size_m: Side length of each square cell (metres).
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        cell_size_m: float = 1.0,
        decay_factor: float = 0.95,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cell_size_m = max(cell_size_m, 0.01)
        self.decay_factor = max(0.0, min(1.0, decay_factor))
        self.cols = max(int(np.ceil((x_max - x_min) / self.cell_size_m)), 1)
        self.rows = max(int(np.ceil((y_max - y_min) / self.cell_size_m)), 1)
        self._cells: list[list[SemanticCell]] = [
            [SemanticCell() for _ in range(self.cols)] for _ in range(self.rows)
        ]

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _cell_index(self, x: float, y: float) -> tuple[int, int] | None:
        """(row, col) for world coordinate, or None if out of bounds."""
        c = int((x - self.x_min) / self.cell_size_m)
        r = int((y - self.y_min) / self.cell_size_m)
        if 0 <= c < self.cols and 0 <= r < self.rows:
            return r, c
        return None

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        """World-space centre of cell (row, col)."""
        x = self.x_min + (col + 0.5) * self.cell_size_m
        y = self.y_min + (row + 0.5) * self.cell_size_m
        return x, y

    # ------------------------------------------------------------------
    # Label operations
    # ------------------------------------------------------------------

    def observe(self, x: float, y: float, label: int, count: float = 1.0) -> bool:
        """Add a label observation at world position (x, y).

        Returns ``True`` if the position is within the map bounds.
        """
        idx = self._cell_index(x, y)
        if idx is None:
            return False
        cell = self._cells[idx[0]][idx[1]]
        if self.decay_factor < 1.0:
            cell.votes = {k: v * self.decay_factor for k, v in cell.votes.items()}
        cell.add_vote(label, count)
        return True

    def observe_region(
        self,
        x_center: float,
        y_center: float,
        radius_m: float,
        label: int,
        count: float = 1.0,
    ) -> int:
        """Add a label observation to all cells within *radius_m*.

        Returns the number of cells updated.
        """
        updated = 0
        cells_r = int(np.ceil(radius_m / self.cell_size_m))
        idx = self._cell_index(x_center, y_center)
        if idx is None:
            return 0
        r0, c0 = idx
        for dr in range(-cells_r, cells_r + 1):
            for dc in range(-cells_r, cells_r + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    cx, cy = self.cell_center(r, c)
                    dist = np.hypot(cx - x_center, cy - y_center)
                    if dist <= radius_m:
                        cell = self._cells[r][c]
                        if self.decay_factor < 1.0:
                            cell.votes = {k: v * self.decay_factor for k, v in cell.votes.items()}
                        cell.add_vote(label, count)
                        updated += 1
        return updated

    def query(self, x: float, y: float) -> SemanticCell | None:
        """Return the cell at (x, y), or None if out of bounds."""
        idx = self._cell_index(x, y)
        if idx is None:
            return None
        return self._cells[idx[0]][idx[1]]

    def label_at(self, x: float, y: float) -> int:
        """Return dominant label at (x, y), or UNKNOWN."""
        cell = self.query(x, y)
        if cell is None:
            return int(SemanticLabel.UNKNOWN)
        return cell.dominant_label

    def confidence_at(self, x: float, y: float) -> float:
        """Return confidence at (x, y), or 0."""
        cell = self.query(x, y)
        if cell is None:
            return 0.0
        return cell.confidence

    # ------------------------------------------------------------------
    # Rasterization
    # ------------------------------------------------------------------

    def to_label_array(self) -> np.ndarray:
        """Return a (rows × cols) int array of dominant labels."""
        arr = np.full((self.rows, self.cols), int(SemanticLabel.UNKNOWN), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                arr[r, c] = self._cells[r][c].dominant_label
        return arr

    def to_confidence_array(self) -> np.ndarray:
        """Return a (rows × cols) float array of confidences."""
        arr = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                arr[r, c] = self._cells[r][c].confidence
        return arr

    def to_observation_count_array(self) -> np.ndarray:
        """Return a (rows × cols) int array of total observation counts."""
        arr = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                arr[r, c] = self._cells[r][c].total_observations
        return arr

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def label_histogram(self) -> dict[int, int]:
        """Count of cells per dominant label (excludes UNKNOWN)."""
        hist: dict[int, int] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                label = self._cells[r][c].dominant_label
                if label != int(SemanticLabel.UNKNOWN):
                    hist[label] = hist.get(label, 0) + 1
        return hist

    def observed_fraction(self) -> float:
        """Fraction of cells that have at least one observation."""
        observed = sum(
            1
            for r in range(self.rows)
            for c in range(self.cols)
            if self._cells[r][c].total_observations > 0
        )
        return observed / max(self.rows * self.cols, 1)

    def clear(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                self._cells[r][c].reset()
