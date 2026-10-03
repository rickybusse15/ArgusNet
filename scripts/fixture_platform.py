"""Where a parity fixture was captured, named as the Rust tests name it.

Python's maths calls go to the platform's libm, so a fixture of reference
values is exact only on the platform that captured it. Every capture script
records `captured_on()` as the fixture's `_captured_on`; the Rust parity
tests (`rust/argusnet-sim/tests/common/parity.rs`) compare bit for bit when
it names the platform they run on, and within a documented bound otherwise.
"""

from __future__ import annotations

import platform

# platform.system() / platform.machine() -> Rust's std::env::consts::{OS, ARCH}.
_OS = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}
_ARCH = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64", "AMD64": "x86_64"}


def captured_on() -> str:
    system, machine = platform.system(), platform.machine()
    return f"{_OS.get(system, system.lower())}-{_ARCH.get(machine, machine.lower())}"
