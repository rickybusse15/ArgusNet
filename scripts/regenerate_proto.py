#!/usr/bin/env python3
"""Regenerate the committed Python protobuf bindings from proto/.

Rust needs no equivalent: `argusnet-proto` runs `tonic-prost-build` from its
own build.rs, so its bindings are produced at compile time and never committed.
Python's are committed because the package ships without a protoc toolchain,
which is exactly why they can drift silently -- hence this script and the CI
check that runs it with --check.

The generator version is pinned in pyproject.toml's `dev` extra. protoc stamps
its own version into the output, so an unpinned generator produces spurious
diffs that have nothing to do with the schema:

    -# Protobuf Python Version: 6.31.0        (grpcio-tools 1.73.1)
    +# Protobuf Python Version: 6.31.1        (grpcio-tools 1.78.0)

Output is written verbatim. Do not run a formatter or linter over it: ruff's
`extend-exclude` already skips `*_pb2.py` / `*_pb2_grpc.py` for that reason.
The committed `world_model_pb2_grpc.py` was at some point passed through
`ruff check --fix`, which rewrote `class Stub(object):` to `class Stub:` and
dropped an "unused" `import warnings`. That made the file unreproducible from
protoc alone, which is the drift this script exists to prevent.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO_ROOT / "proto"
PROTO_FILE = "argusnet/v1/world_model.proto"
# Where the generated modules are committed, mirroring the proto package path.
OUTPUT_DIR = REPO_ROOT / "src" / "argusnet" / "v1"
GENERATED = ("world_model_pb2.py", "world_model_pb2_grpc.py")


def generate(destination: Path) -> None:
    """Run protoc into `destination`, which must already exist."""
    try:
        from grpc_tools import protoc
    except ImportError:  # pragma: no cover - depends on the environment
        sys.exit(
            "grpcio-tools is not installed. It is pinned in the 'dev' extra:\n"
            "    pip install -e '.[dev]'"
        )

    # protoc's argv[0] is ignored but must be present.
    exit_code = protoc.main(
        [
            "protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={destination}",
            f"--grpc_python_out={destination}",
            PROTO_FILE,
        ]
    )
    if exit_code != 0:
        sys.exit(f"protoc failed with exit code {exit_code}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed bindings differ from freshly generated ones. "
        "Used by CI; writes nothing.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        fresh_root = Path(tmp)
        generate(fresh_root)
        fresh_dir = fresh_root / "argusnet" / "v1"

        if args.check:
            drifted = []
            for name in GENERATED:
                committed = OUTPUT_DIR / name
                if not committed.exists():
                    drifted.append(f"{name}: missing from {OUTPUT_DIR}")
                    continue
                expected = (fresh_dir / name).read_text()
                actual = committed.read_text()
                if expected != actual:
                    diff = "".join(
                        difflib.unified_diff(
                            expected.splitlines(keepends=True),
                            actual.splitlines(keepends=True),
                            fromfile=f"{name} (regenerated)",
                            tofile=f"{name} (committed)",
                        )
                    )
                    drifted.append(f"{name} differs:\n{diff}")
            if drifted:
                print("\n".join(drifted), file=sys.stderr)
                print(
                    "\nCommitted protobuf bindings are out of date. Regenerate with:\n"
                    "    python3 scripts/regenerate_proto.py",
                    file=sys.stderr,
                )
                return 1
            print("Committed protobuf bindings match the proto definitions.")
            return 0

        for name in GENERATED:
            shutil.copyfile(fresh_dir / name, OUTPUT_DIR / name)
            print(f"wrote {(OUTPUT_DIR / name).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
