from __future__ import annotations


class ArgusNetError(Exception):
    """Base exception for typed ArgusNet runtime errors."""


class ValidationError(ArgusNetError):
    """Raised when runtime data violates a strict validation contract."""

