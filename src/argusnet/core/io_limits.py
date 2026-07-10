"""Size guards for untrusted file inputs (replay JSON, recordings, DEM rasters).

Bounds memory/CPU spent on a single file before it is parsed or decoded,
protecting against decompression-bomb-style resource exhaustion from a
shared/untrusted replay file, scene package, or GeoTIFF DEM.
"""

from __future__ import annotations

from pathlib import Path

from argusnet.core.errors import ArgusNetError

__all__ = ["DEFAULT_MAX_FILE_BYTES", "FileSizeLimitError", "check_file_size", "read_text_capped"]

DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024  # 256 MB


class FileSizeLimitError(ArgusNetError):
    """Raised when an untrusted input file exceeds its configured size cap."""


def check_file_size(path: str | Path, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> int:
    size = Path(path).stat().st_size
    if size > max_bytes:
        raise FileSizeLimitError(f"{path}: {size} bytes exceeds the {max_bytes} byte limit.")
    return size


def read_text_capped(
    path: str | Path, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES, encoding: str = "utf-8"
) -> str:
    """Read a text file, refusing to load anything larger than ``max_bytes``."""
    check_file_size(path, max_bytes=max_bytes)
    return Path(path).read_text(encoding=encoding)
