"""Size and path guards for untrusted file inputs.

Bounds memory spent on a single file before it is parsed, and refuses a
manifest path that leaves its package — including through a symlink — so a
shared replay, scene package, environment bundle, or GeoTIFF cannot read
arbitrary files or expand without a cap.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

from argusnet.core.errors import ArgusNetError

__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "FileSizeLimitError",
    "check_file_size",
    "read_text_capped",
    "reject_unsafe_relative_path",
    "resolve_contained_path",
    "sha256_file",
]

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


def sha256_file(path: str | Path, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> str:
    """Hash ``path`` in chunks, after refusing a file over ``max_bytes``."""
    check_file_size(path, max_bytes=max_bytes)
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_unsafe_relative_path(relative_path: str, *, field_name: str) -> None:
    """Reject an empty, absolute, or ``..`` path before it is joined onto a root.

    The check is lexical and runs on both POSIX and Windows spellings, so a
    manifest written on one platform cannot escape when it is opened on the
    other.
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if "\x00" in relative_path:
        raise ValueError(f"{field_name} must not contain NUL bytes: {relative_path!r}")
    if PurePosixPath(relative_path).is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise ValueError(f"{field_name} must be a relative path: {relative_path!r}")
    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"{field_name} must not contain '..': {relative_path!r}")


def resolve_contained_path(root: str | Path, relative_path: str, *, field_name: str) -> Path:
    """Join ``relative_path`` onto ``root`` and refuse any path that leaves it.

    ``Path.resolve`` follows symlinks, so a package entry that is a symlink to
    a file outside the package fails ``relative_to`` instead of being read.
    """
    reject_unsafe_relative_path(relative_path, field_name=field_name)
    root_resolved = Path(root).resolve()
    resolved = (root_resolved / relative_path).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"{field_name} escapes the root: {relative_path!r}") from None
    return resolved
