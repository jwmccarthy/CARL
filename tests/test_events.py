import unittest

import torch as th

from carl import _carl
from carl.events import (
    ArenaSurface, SurfaceContact, BallTouch, BallTouchBatch, ContactSurface,
    CarEvents, Event, Jump,
)


class EventDefinitionTests(unittest.TestCase):

    def test_native_soccar_dimensions_are_exported(self) -> None:
        from carl import SOCCAR_BACK_WALL, SOCCAR_CEILING, SOCCAR_SIDE_WALL
        from carl.training import reset

        self.assertEqual((SOCCAR_SIDE_WALL, SOCCAR_BACK_WALL, SOCCAR_CEILING), (4096, 5120, 2048))
        self.assertEqual(reset.SOCCAR_SIDE_WALL, _carl.SOCCAR_SIDE_WALL)
        self.assertEqual(reset.SOCCAR_BACK_WALL, _carl.SOCCAR_BACK_WALL)
        self.assertEqual(reset.SOCCAR_CEILING, _carl.SOCCAR_CEILING)

    def test_integer_identifiers(self) -> None:
        events = th.tensor(list(Event), dtype=th.int32)

        self.assertEqual(events.unique().numel(), len(Event))
        self.assertEqual(Event.BALL_TOUCH, _carl.EVENT_BALL_TOUCH)
        self.assertEqual(ContactSurface.UNDERSIDE, _carl.TOUCH_UNDERSIDE)

    def test_standalone_conditions(self) -> None:
        touch = BallTouch(surface=ContactSurface.FRONT, min_speed_gain=500)
        contact = SurfaceContact(surface=ArenaSurface.FLOOR, entity="ball")

        self.assertEqual(touch.surface, ContactSurface.FRONT)
        self.assertEqual(contact.surface, ArenaSurface.FLOOR)
        self.assertEqual(contact.entity, "ball")
        self.assertEqual(Jump(), Jump())

    def test_named_flags_are_elementwise(self) -> None:
        events = CarEvents(th.tensor([
            [Event.JUMP | Event.BALL_TOUCH, Event.DODGE],
            [Event.FLIP_RESET, Event.JUMP],
        ], dtype=th.int32))

        self.assertEqual(events.jumped.tolist(), [[True, False], [False, True]])
        self.assertEqual(events.ball_touched.tolist(), [[True, False], [False, False]])
        self.assertEqual(events.flip_reset.tolist(), [[False, False], [True, False]])

    def test_leave_surface_flag(self) -> None:
        events = CarEvents(th.tensor([[Event.LEAVE_SURFACE]], dtype=th.int32))
        self.assertEqual(events.left_surface.tolist(), [[True]])

    def test_touch_batch(self) -> None:
        batch = BallTouchBatch(
            surfaces = th.tensor([[1 << int(ContactSurface.FRONT), 0],
                                  [(1 << int(ContactSurface.ROOF)) |
                                   (1 << int(ContactSurface.FRONT)), 0]], dtype=th.int32),
        )

        self.assertEqual(batch.touched.tolist(), [[True, False], [True, False]])
        self.assertEqual(batch.on_surface(ContactSurface.ROOF).tolist(), [[False, False], [True, False]])
        self.assertEqual(batch.on_surface(ContactSurface.FRONT).tolist(), [[True, False], [True, False]])
