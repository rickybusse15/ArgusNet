from __future__ import annotations

import sys
from typing import Optional, Sequence

from argusnet.cli.main import main as cli_main
from argusnet.world.scene_loader import build_scene_package, load_scene_manifest  # noqa: F401


def main(argv: Optional[Sequence[str]] = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    cli_main(["build-scene", *forwarded])


if __name__ == "__main__":
    main()
