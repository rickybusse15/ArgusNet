"""Analysis imports without the native library (P6.1).

Evaluation, reporting, offline import and configuration are usable from a
checkout or an install where Rust was never built: the native library is loaded
when something calls into it, not when a module is imported. Every module is
imported first, on its own, so no import order hides a circle.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import importlib, sys

failed = []
for name in sys.argv[1:]:
    for loaded in [m for m in sys.modules if m == "argusnet" or m.startswith("argusnet.")]:
        del sys.modules[loaded]
    try:
        importlib.import_module(name)
    except BaseException as error:
        failed.append(f"{name}: {type(error).__name__}: {error}")
print("\\n".join(failed))

# The library really is out of reach: the first call into it says so.
from argusnet._native_world import call
try:
    call("visibility_profile")
except ImportError:
    pass
else:
    print("the native library was loaded")
"""


def _probe(modules, tmp_path):
    """Import each module first, with no native library anywhere to load.

    The loader takes the library named here and looks nowhere else, so a build in
    the checkout or beside the package is not found either.
    """
    source = ROOT / "src"
    return subprocess.run(
        [sys.executable, "-c", _PROBE, *modules],
        cwd=tmp_path,
        env={
            "PYTHONPATH": str(source),
            "PATH": "/usr/bin:/bin",
            "ARGUSNET_NATIVE_LIBRARY": str(tmp_path / "no-such-library"),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _modules(package):
    source = ROOT / "src"
    return sorted(
        ".".join(path.relative_to(source).with_suffix("").parts).removesuffix(".__init__")
        for path in (source / "argusnet" / package).rglob("*.py")
        if path.name != "__main__.py"
    )


PACKAGES = sorted(
    path.name
    for path in (ROOT / "src" / "argusnet").iterdir()
    # ``v1`` is the generated gRPC stubs: they need grpc, not the library.
    if (path / "__init__.py").is_file() and path.name != "v1"
)


@pytest.mark.parametrize("package", PACKAGES)
def test_every_module_imports_first_and_without_the_native_library(package, tmp_path):
    modules = _modules(package)
    assert modules
    result = _probe(modules, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_the_probe_finds_a_module_that_needs_the_library(tmp_path):
    needs = tmp_path / "needs_native.py"
    needs.write_text(
        "from argusnet._native_world import call\nPROFILE = call('visibility_profile')\n"
    )
    result = _probe(["needs_native"], tmp_path)
    assert result.stdout.startswith("needs_native: "), (result.stdout, result.stderr)


def test_defaults_asked_for_late_are_rusts_own():
    from argusnet._native_world import call
    from argusnet.safety.native import DronePhysicalLimits
    from argusnet.world.visibility import SensorVisibilityModel

    limits = call("limits", profile="tracker")
    made = DronePhysicalLimits()
    assert {name: getattr(made, name) for name in made.__dataclass_fields__} == {
        name: limits[name] for name in made.__dataclass_fields__
    }
    profile = call("visibility_profile")
    model = SensorVisibilityModel()
    assert model.hard_blockers == frozenset(profile["hard_blockers"])
    assert model.vegetation_hard_block_threshold == profile["vegetation_hard_block_threshold"]
