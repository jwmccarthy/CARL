import os
from pathlib import Path


os.environ.setdefault(
    "CARL_ARENA_OBJ",
    str(Path(__file__).with_name("assets") / "arena.obj"),
)

from . import _carl

# Re-export the native layout and physics constants without copying their values
__all__ = [name for name in vars(_carl) if name == "Env" or name.isupper()]
globals().update({name: getattr(_carl, name) for name in __all__})
