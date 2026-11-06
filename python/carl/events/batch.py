from dataclasses import dataclass, field, fields, replace

import torch as th

from .. import _carl
from .flags import BallEvents, CarEvents, EntityEvents
from .touches import BallTouchBatch


@dataclass(kw_only=True)
class CarlEvents:

    score_delta:  th.Tensor | None      = None
    done:         th.Tensor | None      = None
    terminated:   th.Tensor | None      = None
    truncated:    th.Tensor | None      = None
    cars:         CarEvents | None      = None
    ball:         BallEvents | None     = None
    ball_touches: BallTouchBatch | None = None
    _native:      _carl.EventState | None = field(default=None, repr=False, compare=False)

    @property
    def ego(self) -> CarEvents:
        if self.cars is None:
            raise RuntimeError("ego events require capture_events=True")
        if self.cars.flags.shape[1] != 1:
            raise ValueError("ego events require exactly one car per simulation")
        return CarEvents(
            self.cars.flags[:, 0], self.cars._native,
            self.cars.surface_contacted[:, 0], self.cars.surface_left[:, 0],
        )

    @classmethod
    def from_native(cls, native: _carl.EventState, **kwargs):
        native.acquire(th.cuda.current_stream().cuda_stream)
        batch = cls(_native=native, **kwargs)

        if native.enabled:
            batch.cars = CarEvents(
                th.from_dlpack(native.get_car_flags()), native,
                th.from_dlpack(native.get_car_surface_contacted()),
                th.from_dlpack(native.get_car_surface_left()),
            )
            batch.ball = BallEvents(
                flags           = th.from_dlpack(native.get_ball_flags()),
                surface_entered = th.from_dlpack(native.get_ball_surface_entered()),
                surface_left    = th.from_dlpack(native.get_ball_surface_left()),
                goal_side       = th.from_dlpack(native.get_goal_side()),
                speed_change    = th.from_dlpack(native.get_ball_speed_change()),
                _native         = native,
            )

        if native.touches_enabled:
            batch.ball_touches = BallTouchBatch(
                surfaces = th.from_dlpack(native.get_ball_touch_surfaces()),
                tick_surfaces = th.from_dlpack(native.get_ball_touch_tick_surfaces()),
                tick_speed = th.from_dlpack(native.get_ball_touch_tick_speed()),
                tick_speed_before = th.from_dlpack(native.get_ball_touch_tick_speed_before()),
                _native  = native,
            )

        return batch

    def wait(self):
        if self._native is not None:
            self._native.acquire(th.cuda.current_stream().cuda_stream)
        return self

    def snapshot(self):
        self.wait()
        snapshot = replace(self, _native=None)

        for item in fields(self):
            value = getattr(self, item.name)
            
            if isinstance(value, th.Tensor):
                setattr(snapshot, item.name, value.clone())
            elif isinstance(value, (EntityEvents, BallTouchBatch)):
                setattr(snapshot, item.name, value.snapshot())

        return snapshot
