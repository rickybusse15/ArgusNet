"""Backward-compatibility shim — transparent alias for argusnet.simulation.sim.

Using sys.modules aliasing so that patches on ``smart_tracker.sim.X`` also
affect ``argusnet.simulation.sim.X`` (same module object).
"""
import sys as _sys
import argusnet.simulation.sim as _argusnet_sim

_sys.modules[__name__] = _argusnet_sim
