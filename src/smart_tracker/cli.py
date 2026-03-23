"""Backward-compatibility shim — transparent alias for argusnet.cli.main.

Using sys.modules aliasing so that patches on ``smart_tracker.cli.X`` also
affect ``argusnet.cli.main.X`` (same module object).
"""
import sys as _sys
import argusnet.cli.main as _argusnet_cli_main

_sys.modules[__name__] = _argusnet_cli_main
