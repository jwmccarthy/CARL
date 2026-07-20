import os
from pathlib import Path


os.environ.setdefault(
    "CARL_ARENA_OBJ",
    str(Path(__file__).with_name("assets") / "arena.obj"),
)

from ._carl import Env

__all__ = ["Env"]
