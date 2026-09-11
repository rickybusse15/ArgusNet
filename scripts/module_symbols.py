#!/usr/bin/env python3
"""Top-level symbols of a module, classified by where they are referenced.

Three buckets, and the third is the one that needs care:

  referenced outside the module  -- the module's real surface.
  private, used only within      -- a helper its own module calls. LIVE. An
                                    earlier version of this script lacked this
                                    bucket and reported deconfliction.py's
                                    CorridorWindowAllocator (145 lines) as
                                    unreachable; DeconflictionLayer
                                    instantiates it at line 243.
  referenced nowhere             -- genuinely dead.

The external search is word-bounded, so `generate_mission` does not match
sim.py's unrelated `generate_mission_zones`.

Usage:
    python3 scripts/module_symbols.py src/argusnet/planning/inspection.py
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
tree = ast.parse(path.read_text())
lines = path.read_text().splitlines()

syms = []
for node in tree.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        end = node.end_lineno or node.lineno
        syms.append((node.name, node.lineno, end, end - node.lineno + 1, type(node).__name__))


def refs(name):
    out = subprocess.run(
        ["grep", "-rln", r"\b" + name + r"\b", "src/", "tests/", "--include=*.py"],
        capture_output=True,
        text=True,
    ).stdout.split()
    return [f for f in out if Path(f) != path]


def internal_uses(name, decl_lineno):
    """References to `name` inside this module, excluding its own definition
    line. A private helper called only by its own module is LIVE, not dead --
    counting only external references would report it as unreachable."""
    n = 0
    for i, line in enumerate(lines, start=1):
        if i == decl_lineno:
            continue
        if re.search(r"\b" + re.escape(name) + r"\b", line):
            n += 1
    return n


used, unused, internal = [], [], []
used_lines = unused_lines = internal_lines = 0
for name, a, _b, n, _kind in syms:
    r = refs(name)
    inner = internal_uses(name, a)
    if r:
        used.append((name, n, r))
        used_lines += n
    elif inner:
        internal.append((name, n, inner))
        internal_lines += n
    else:
        unused.append((name, n))
        unused_lines += n

print(f"=== {path} ({len(lines)} lines) ===")
print(f"-- referenced outside the module ({used_lines} lines) --")
for name, n, r in sorted(used, key=lambda x: -x[1]):
    print(f"  {n:5}  {name:38} {', '.join(sorted(set(r)))}")
print(f"-- private, used only within this module ({internal_lines} lines) --")
for name, n, c in sorted(internal, key=lambda x: -x[1]):
    print(f"  {n:5}  {name:38} {c} internal reference(s)")
print(f"-- referenced NOWHERE, inside or out ({unused_lines} lines) --")
for name, n in sorted(unused, key=lambda x: -x[1]):
    print(f"  {n:5}  {name}")
