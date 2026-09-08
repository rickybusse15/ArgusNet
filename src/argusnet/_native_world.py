"""Owned bindings to the Rust world and safety authorities (ABI version 1)."""

from __future__ import annotations

import ctypes
import importlib.machinery
import json
import os
import sys
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


@lru_cache(maxsize=1)
def _library():
    package = Path(__file__).resolve().parent
    override = os.environ.get("ARGUSNET_NATIVE_LIBRARY")
    candidates = (
        [Path(override)]
        if override
        else [package / ("_native" + suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    )
    # Source-checkout development uses an explicitly built Cargo artifact.
    root = package.parents[1]
    name = (
        "argusnet_native.dll"
        if sys.platform == "win32"
        else ("libargusnet_native.dylib" if sys.platform == "darwin" else "libargusnet_native.so")
    )
    if not override and (root / "Cargo.toml").exists():
        candidates += [root / "target" / profile / name for profile in ("release", "debug")]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ImportError(
            "ArgusNet's Rust world library is required. Install the package with pip install -e . "
            "or build it with cargo build -p argusnet-native."
        )
    lib = ctypes.CDLL(str(path))
    lib.argusnet_call.argtypes = [ctypes.c_char_p]
    lib.argusnet_call.restype = ctypes.c_void_p
    lib.argusnet_free.argtypes = [ctypes.c_void_p]
    lib.argusnet_free.restype = None
    lib.argusnet_sample.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.argusnet_sample.restype = ctypes.c_int
    return lib


def call(op: str, **kwargs):
    lib = _library()
    payload = json.dumps({"op": op, **kwargs}, default=_json_default, allow_nan=False).encode()
    pointer = lib.argusnet_call(payload)
    if not pointer:
        raise RuntimeError("Rust world returned no response")
    try:
        result = json.loads(ctypes.string_at(pointer))
    finally:
        lib.argusnet_free(pointer)
    if "error" in result:
        raise ValueError(result["error"])
    return result["ok"]


class NativeHandle:
    """An owned native object; integer IDs prevent use-after-free at the ABI."""

    def __init__(self, op: str, **kwargs):
        self.id = call(op, **kwargs)

    def __copy__(self):
        return NativeHandle("clone", handle=self.id)

    def __deepcopy__(self, memo):
        copied = self.__copy__()
        memo[id(self)] = copied
        return copied

    def __getstate__(self):
        raise TypeError("Native world handles cannot be pickled; serialize world metadata instead")

    def __del__(self):
        if getattr(self, "id", None) is not None:
            with suppress(Exception):
                call("drop", handle=self.id)
            self.id = None


def sample(handle: NativeHandle, points, *, analytic=False, method="heights", delta=0.5):
    points = np.asarray(points, dtype=float)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("xy_m must have shape (..., 2).")
    shape = points.shape[:-1] + ((2,) if method == "gradients" else ())
    points = np.ascontiguousarray(points.reshape(-1, 2), dtype=np.float64)
    values = np.empty(shape, dtype=np.float64)
    mode = 2 if method == "gradients" else 3 if method == "curvature" else int(analytic)
    code = _library().argusnet_sample(
        handle.id,
        points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(points),
        mode,
        delta,
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code:
        raise ValueError("Invalid native terrain query or expired handle")
    return values if shape else float(values)
