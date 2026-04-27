"""Spatial tile index for ArgusNet.

Provides a grid-based spatial index that maps 2-D bounding boxes to sets
of keyframe IDs.  Uses a simple fixed-resolution grid (no R-tree
dependency) suitable for the expected data volumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from argusnet.core.ids import new_tile_id

__all__ = [
    "TileIndex",
    "TileBounds",
    "TileCell",
]


@dataclass(frozen=True)
class TileBounds:
    """Axis-aligned 2-D bounding box (lon/lat or local x/y)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x_min + self.x_max) * 0.5,
            (self.y_min + self.y_max) * 0.5,
        )

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def contains_point(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def intersects(self, other: TileBounds) -> bool:
        return not (
            self.x_max < other.x_min
            or self.x_min > other.x_max
            or self.y_max < other.y_min
            or self.y_min > other.y_max
        )


@dataclass
class TileCell:
    """A single cell in the tile grid."""

    tile_id: str
    bounds: TileBounds
    col: int
    row: int
    keyframe_ids: set[str] = field(default_factory=set)

    @property
    def count(self) -> int:
        return len(self.keyframe_ids)


class TileIndex:
    """Fixed-resolution 2-D grid index over keyframe positions.

    The index covers a world-space bounding box divided into
    ``cols × rows`` tiles.  Each tile stores a set of keyframe IDs
    whose positions fall within its bounds.

    Args:
        bounds: World-space bounding box for the index area.
        cols: Number of columns (x divisions).
        rows: Number of rows (y divisions).
    """

    def __init__(
        self,
        bounds: TileBounds,
        cols: int = 32,
        rows: int = 32,
    ) -> None:
        self.bounds = bounds
        self.cols = max(cols, 1)
        self.rows = max(rows, 1)
        self._cell_width = bounds.width / self.cols
        self._cell_height = bounds.height / self.rows
        # Pre-create the grid
        self._grid: list[list[TileCell]] = []
        for r in range(self.rows):
            row_cells: list[TileCell] = []
            for c in range(self.cols):
                cell_bounds = TileBounds(
                    x_min=bounds.x_min + c * self._cell_width,
                    x_max=bounds.x_min + (c + 1) * self._cell_width,
                    y_min=bounds.y_min + r * self._cell_height,
                    y_max=bounds.y_min + (r + 1) * self._cell_height,
                )
                row_cells.append(
                    TileCell(
                        tile_id=new_tile_id(0, c, r),
                        bounds=cell_bounds,
                        col=c,
                        row=r,
                    )
                )
            self._grid.append(row_cells)

    # ------------------------------------------------------------------
    # Point → cell mapping
    # ------------------------------------------------------------------

    def _cell_for_point(self, x: float, y: float) -> TileCell | None:
        """Return the cell containing (x, y), or None if out of bounds."""
        if not self.bounds.contains_point(x, y):
            return None
        c = min(int((x - self.bounds.x_min) / self._cell_width), self.cols - 1)
        r = min(int((y - self.bounds.y_min) / self._cell_height), self.rows - 1)
        return self._grid[r][c]

    # ------------------------------------------------------------------
    # Insert / remove
    # ------------------------------------------------------------------

    def insert(self, keyframe_id: str, x: float, y: float) -> bool:
        """Register a keyframe at position (x, y).

        Expands the index bounds if (x, y) falls outside them, then
        always returns ``True``.
        """
        self._expand_to_include(x, y)
        cell = self._cell_for_point(x, y)
        if cell is None:
            return False
        cell.keyframe_ids.add(keyframe_id)
        return True

    def _expand_to_include(self, x: float, y: float) -> None:
        """Expand the grid bounds to include (x, y) if necessary.

        Rebuilds the internal grid while preserving all existing keyframe
        ID registrations.
        """
        if self.bounds.contains_point(x, y):
            return

        # Compute new bounds, adding 10% padding on each expanded side.
        new_x_min = self.bounds.x_min
        new_x_max = self.bounds.x_max
        new_y_min = self.bounds.y_min
        new_y_max = self.bounds.y_max

        if x < new_x_min:
            expansion = new_x_min - x
            new_x_min = x - expansion * 0.1
        if x > new_x_max:
            expansion = x - new_x_max
            new_x_max = x + expansion * 0.1
        if y < new_y_min:
            expansion = new_y_min - y
            new_y_min = y - expansion * 0.1
        if y > new_y_max:
            expansion = y - new_y_max
            new_y_max = y + expansion * 0.1

        new_bounds = TileBounds(
            x_min=new_x_min,
            x_max=new_x_max,
            y_min=new_y_min,
            y_max=new_y_max,
        )

        # Collect all existing (keyframe_id, center_x, center_y) entries
        # by using each cell's center as the representative point.
        existing: list[tuple[str, float, float]] = []
        for row in self._grid:
            for cell in row:
                cx, cy = cell.bounds.center
                for kid in cell.keyframe_ids:
                    existing.append((kid, cx, cy))

        # Rebuild grid with new bounds.
        new_cell_width = new_bounds.width / self.cols
        new_cell_height = new_bounds.height / self.rows
        new_grid: list[list[TileCell]] = []
        for r in range(self.rows):
            row_cells: list[TileCell] = []
            for c in range(self.cols):
                cell_bounds = TileBounds(
                    x_min=new_bounds.x_min + c * new_cell_width,
                    x_max=new_bounds.x_min + (c + 1) * new_cell_width,
                    y_min=new_bounds.y_min + r * new_cell_height,
                    y_max=new_bounds.y_min + (r + 1) * new_cell_height,
                )
                row_cells.append(
                    TileCell(
                        tile_id=new_tile_id(0, c, r),
                        bounds=cell_bounds,
                        col=c,
                        row=r,
                    )
                )
            new_grid.append(row_cells)

        # Swap in new grid and dimensions.
        self.bounds = new_bounds
        self._cell_width = new_cell_width
        self._cell_height = new_cell_height
        self._grid = new_grid

        # Re-map existing keyframe IDs into the new grid.
        for kid, ex, ey in existing:
            cell = self._cell_for_point(ex, ey)
            if cell is not None:
                cell.keyframe_ids.add(kid)

    def remove(self, keyframe_id: str) -> int:
        """Remove a keyframe ID from all cells.  Returns removal count."""
        count = 0
        for row in self._grid:
            for cell in row:
                if keyframe_id in cell.keyframe_ids:
                    cell.keyframe_ids.discard(keyframe_id)
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_point(self, x: float, y: float) -> frozenset[str]:
        """Return keyframe IDs in the cell containing (x, y)."""
        cell = self._cell_for_point(x, y)
        if cell is None:
            return frozenset()
        return frozenset(cell.keyframe_ids)

    def query_box(self, box: TileBounds) -> frozenset[str]:
        """Return all keyframe IDs in cells overlapping *box*."""
        result: set[str] = set()
        # Determine affected column/row range
        c_min = max(int((box.x_min - self.bounds.x_min) / self._cell_width), 0)
        c_max = min(
            int((box.x_max - self.bounds.x_min) / self._cell_width),
            self.cols - 1,
        )
        r_min = max(int((box.y_min - self.bounds.y_min) / self._cell_height), 0)
        r_max = min(
            int((box.y_max - self.bounds.y_min) / self._cell_height),
            self.rows - 1,
        )
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                result.update(self._grid[r][c].keyframe_ids)
        return frozenset(result)

    def query_radius(self, cx: float, cy: float, radius: float) -> frozenset[str]:
        """Return keyframe IDs in cells whose centres are within *radius*
        of ``(cx, cy)``.  This is approximate (cell-level, not point-level).
        """
        result: set[str] = set()
        for row in self._grid:
            for cell in row:
                cell_cx, cell_cy = cell.bounds.center
                dist = math.hypot(cell_cx - cx, cell_cy - cy)
                if dist <= radius + max(self._cell_width, self._cell_height):
                    result.update(cell.keyframe_ids)
        return frozenset(result)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def all_cells(self) -> list[TileCell]:
        """Flat list of every cell in the grid."""
        return [cell for row in self._grid for cell in row]

    def non_empty_cells(self) -> list[TileCell]:
        """Return cells that contain at least one keyframe."""
        return [cell for row in self._grid for cell in row if cell.count > 0]

    @property
    def total_keyframes(self) -> int:
        """Total number of (cell, keyframe) registrations."""
        return sum(cell.count for row in self._grid for cell in row)

    def clear(self) -> None:
        """Remove all keyframe registrations."""
        for row in self._grid:
            for cell in row:
                cell.keyframe_ids.clear()
