from dataclasses import dataclass, field

import torch as th

from .. import _carl
from .enums import ContactSurface


@dataclass
class BallTouchBatch:
    surfaces:          th.Tensor
    _native:           _carl.EventState | None = field(default=None, repr=False, compare=False)
    tick_surfaces:     th.Tensor        | None = None
    tick_speed:        th.Tensor        | None = None
    tick_speed_before: th.Tensor        | None = None

    def wait(self):
        if self._native is not None:
            self._native.acquire(th.cuda.current_stream().cuda_stream)
        return self

    def snapshot(self):
        self.wait()
        return BallTouchBatch(
            self.surfaces.clone(),
            tick_surfaces=None if self.tick_surfaces is None else self.tick_surfaces.clone(),
            tick_speed=None if self.tick_speed is None else self.tick_speed.clone(),
            tick_speed_before=None if self.tick_speed_before is None else self.tick_speed_before.clone(),
        )

    @property
    def touched(self) -> th.Tensor:
        self.wait()
        return self.surfaces != 0

    def on_surface(self, surface: ContactSurface) -> th.Tensor:
        self.wait()
        return (self.surfaces & (1 << int(surface))) != 0
