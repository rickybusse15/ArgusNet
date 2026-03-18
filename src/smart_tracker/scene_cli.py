from __future__ import annotations

import sys
from typing import Optional, Sequence

from .cli import main as cli_main


def main(argv: Optional[Sequence[str]] = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    cli_main(["build-scene", *forwarded])


if __name__ == "__main__":
    main()
