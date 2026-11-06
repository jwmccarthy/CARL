import os
from pathlib import Path

_arena_obj = Path(__file__).with_name("assets") / "arena.obj"
if not _arena_obj.is_file():
    _arena_obj = Path(__file__).resolve().parents[2] / "assets" / "arena.obj"
os.environ.setdefault("CARL_ARENA_OBJ", str(_arena_obj))

import torch as _torch

try:
    from . import _carl
except ImportError as exc:
    raise ImportError(
        f"CARL requires a native GPU extension compatible with PyTorch {_torch.__version__}. "
        "Build with CUDA 12.8, or ROCm with CARL_GPU_BACKEND=HIP."
    ) from exc

__all__ = [name for name in vars(_carl) if name == "Env" or name.isupper()]
globals().update({name: getattr(_carl, name) for name in __all__})
