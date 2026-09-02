"""Guards on the lazy public API in ``argusnet/__init__.py``.

The package advertises its surface through ``__all__`` and resolves each name on
demand through ``_EXPORTS``. Nothing previously checked that the two agree with
reality, so ``argusnet.LaunchEvent`` survived in both tables long after the
launch system it referred to was deleted -- ``from argusnet import LaunchEvent``
raised ``AttributeError`` at runtime while the name still looked supported.
"""

from __future__ import annotations

import pytest

import argusnet

_EXPORTS: dict[str, tuple[str, str]] = argusnet._EXPORTS


@pytest.mark.parametrize("name", sorted(argusnet.__all__))
def test_every_advertised_name_resolves(name: str) -> None:
    """Each ``__all__`` entry must actually be importable.

    ``__getattr__`` raises ``AttributeError`` for a dangling ``_EXPORTS`` target,
    so simply resolving the attribute is the assertion.
    """
    getattr(argusnet, name)


def test_all_and_exports_cover_the_same_names() -> None:
    """``__all__`` and ``_EXPORTS`` must not drift apart."""
    advertised = set(argusnet.__all__)
    resolvable = set(_EXPORTS)
    assert advertised - resolvable == set(), "names in __all__ with no _EXPORTS entry"
    assert resolvable - advertised == set(), "names in _EXPORTS not advertised in __all__"


def test_unknown_attribute_still_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        argusnet.ThisNameDoesNotExist  # noqa: B018
