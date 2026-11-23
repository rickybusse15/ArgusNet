"""The Python reference world.

``environment`` re-exports what was moved out of it into ``obstacles``,
``visibility`` and ``environment_io``, which import it back. Loaded first, the
circle closes whichever of them is imported first.
"""

from . import environment as environment
