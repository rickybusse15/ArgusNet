#!/usr/bin/env python3
"""Transitive import reachability over src/argusnet, from the entry points.

Classifies each module of a package as RUNTIME (reachable by following imports
from a [project.scripts] entry point), test-only (reachable only from tests/),
or ORPHAN (neither).

Follows imports with `ast`. Counting name references is NOT the same thing --
in P4.4 the two disagreed by 759 lines, because grepping sim.py for `state`
found local variables while sim.py never imports localization.state at all.
Function-level imports count: `ast.walk` reaches them, and
mission/profiles.py imports from planning.inspection inside a function body.

Usage:
    python3 scripts/module_reachability.py argusnet.planning
"""

import ast
import sys
from collections import deque
from pathlib import Path

SRC = Path("src")
PKG = "argusnet"

ENTRY = [
    "argusnet.cli.main",
    "argusnet.simulation.sim",
    "argusnet.evaluation.export",
    "argusnet.cli.scene",
]


def mod_path(mod):
    p = SRC / (mod.replace(".", "/") + ".py")
    if p.exists():
        return p
    p = SRC / mod.replace(".", "/") / "__init__.py"
    return p if p.exists() else None


def imports_of(path, mod):
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    out = set()
    pkg = mod.rsplit(".", 1)[0] if "." in mod else mod
    if path.name == "__init__.py":
        pkg = mod
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(PKG):
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - node.level + 1]
                root = ".".join(base + ([node.module] if node.module else []))
            else:
                root = node.module or ""
            if not root.startswith(PKG):
                continue
            out.add(root)
            for a in node.names:
                out.add(f"{root}.{a.name}")
    return out


def close(seeds):
    seen, q = set(), deque(seeds)
    while q:
        m = q.popleft()
        if m in seen:
            continue
        p = mod_path(m)
        if p is None:
            continue
        seen.add(m)
        for imp in imports_of(p, m):
            if imp not in seen:
                q.append(imp)
    return seen


runtime = close(ENTRY)

# Modules imported by the test suite (exercised, but not by the running product).
test_seeds = set()
for t in Path("tests").rglob("*.py"):
    test_seeds |= imports_of(t, "tests")
tested = close(test_seeds)

target = sys.argv[1] if len(sys.argv) > 1 else "argusnet.planning"
print(f"=== {target} ===")
rows = []
for p in sorted((SRC / target.replace(".", "/")).glob("*.py")):
    mod = f"{target}.{p.stem}" if p.stem != "__init__" else target
    n = len(p.read_text().splitlines())
    if mod in runtime:
        state = "RUNTIME"
    elif mod in tested:
        state = "test-only"
    else:
        state = "ORPHAN"
    rows.append((state, n, p.name))
    print(f"  {state:10} {n:5}  {p.name}")
for s in ("RUNTIME", "test-only", "ORPHAN"):
    tot = sum(n for st, n, _ in rows if st == s)
    print(f"  -> {s}: {tot} lines")
