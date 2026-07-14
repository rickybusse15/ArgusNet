"""Runtime localization-query interface.

The versioned seam between the localization runtime and its consumers — the
operator/eval replay surface today, precision inspection routing and safety next
(LOCALIZATION.md §13). Consumers ask *where is this platform, and how much can I
trust that* without depending on a particular estimator's internals.

Contract:

* :class:`LocalizationQuery` is the runtime protocol consumers depend on.
* :class:`~argusnet.localization.engine.GridLocalizer` is the default backend.
* Every backend exposes ``source_id`` / ``version`` for provenance/lineage, and
  ``LOCALIZATION_QUERY_CONTRACT_VERSION`` versions the interface shape itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from argusnet.core.types import PoseEstimate

# Version of the LocalizationQuery interface shape (pose/covariance/status
# semantics), distinct from an individual backend's ``version``.
LOCALIZATION_QUERY_CONTRACT_VERSION = "1.0"

__all__ = [
    "LOCALIZATION_QUERY_CONTRACT_VERSION",
    "LocalizationQuery",
]


@runtime_checkable
class LocalizationQuery(Protocol):
    """Read-only pose/covariance/status interface a localization backend exposes."""

    source_id: str
    version: str

    def current_pose(self, platform_id: str) -> PoseEstimate | None:
        """Map-relative pose estimate with covariance and status, or None."""
        ...

    def current_covariance(self, platform_id: str) -> tuple[float, ...]:
        """Flattened row-major 3x3 position covariance (m^2); empty if unknown."""
        ...

    def localization_status(self, platform_id: str) -> str:
        """Current :class:`~argusnet.core.types.LocalizationStatus` value."""
        ...

    def confidence(self, platform_id: str) -> float:
        """Scalar localization confidence in [0, 1]."""
        ...

    def is_localized(self, platform_id: str, threshold: float | None = None) -> bool:
        """Whether the platform has a trustworthy fix for map-relative action."""
        ...

    def pose_estimates(self) -> tuple[PoseEstimate, ...]:
        """All current per-platform poses (stable order)."""
        ...
