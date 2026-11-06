from dataclasses import dataclass, field, fields, replace

import torch as th

from .. import _carl
from .enums import Event


@dataclass
class EntityEvents:

    flags:   th.Tensor
    _native: _carl.EventState | None = field(default=None, repr=False, compare=False)

    def wait(self):
        if self._native is not None:
            self._native.acquire(th.cuda.current_stream().cuda_stream)
        return self

    def _has(self, event: Event) -> th.Tensor:
        self.wait()
        return (self.flags & int(event)) != 0

    def snapshot(self):
        self.wait()
        snapshot = replace(self, _native=None)

        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, th.Tensor):
                setattr(snapshot, item.name, value.clone())

        return snapshot


@dataclass
class CarEvents(EntityEvents):

    surface_contacted: th.Tensor | None = None
    surface_left:      th.Tensor | None = None

    @property
    def ball_touched(self) -> th.Tensor:
        return self._has(Event.BALL_TOUCH)

    @property
    def jumped(self) -> th.Tensor:
        return self._has(Event.JUMP)

    @property
    def double_jumped(self) -> th.Tensor:
        return self._has(Event.DOUBLE_JUMP)

    @property
    def dodged(self) -> th.Tensor:
        return self._has(Event.DODGE)

    @property
    def contacted_surface(self) -> th.Tensor:
        return self._has(Event.CONTACT_SURFACE)

    @property
    def flip_reset(self) -> th.Tensor:
        return self._has(Event.FLIP_RESET)

    @property
    def demoed(self) -> th.Tensor:
        return self._has(Event.DEMOED)

    @property
    def left_surface(self) -> th.Tensor:
        return self._has(Event.LEAVE_SURFACE)


@dataclass
class BallEvents(EntityEvents):

    surface_entered: th.Tensor | None = None
    surface_left:    th.Tensor | None = None
    goal_side:       th.Tensor | None = None
    speed_change:    th.Tensor | None = None

    @property
    def touched(self) -> th.Tensor:
        return self._has(Event.BALL_TOUCH)

    @property
    def scored(self) -> th.Tensor:
        return self._has(Event.GOAL)

    @property
    def accelerated(self) -> th.Tensor:
        return self._has(Event.BALL_ACCELERATES)
