import unittest

import torch as th

from carl import _carl
from carl.events import (
    ArenaSurface, SurfaceContact, BallTouch, BallTouchBatch, ContactSurface,
    CarEvents, Event, Jump,
)


class EventDefinitionTests(unittest.TestCase):

    def test_integer_identifiers(self):
        events = th.tensor(list(Event), dtype=th.int32)

        self.assertEqual(events.unique().numel(), len(Event))
        self.assertEqual(Event.BALL_TOUCH, _carl.EVENT_BALL_TOUCH)
        self.assertEqual(ContactSurface.UNDERSIDE, _carl.TOUCH_UNDERSIDE)

    def test_standalone_conditions(self):
        touch = BallTouch(surface=ContactSurface.FRONT, min_speed_gain=500)
        contact = SurfaceContact(surface=ArenaSurface.FLOOR, entity="ball")

        self.assertEqual(touch.surface, ContactSurface.FRONT)
        self.assertEqual(contact.surface, ArenaSurface.FLOOR)
        self.assertEqual(contact.entity, "ball")
        self.assertEqual(Jump(), Jump())

    def test_named_flags_are_elementwise(self):
        events = CarEvents(th.tensor([
            [Event.JUMP | Event.BALL_TOUCH, Event.DODGE],
            [Event.FLIP_RESET, Event.JUMP],
        ], dtype=th.int32))

        self.assertEqual(events.jumped.tolist(), [[True, False], [False, True]])
        self.assertEqual(events.ball_touched.tolist(), [[True, False], [False, False]])
        self.assertEqual(events.flip_reset.tolist(), [[False, False], [True, False]])

    def test_leave_surface_flag(self):
        events = CarEvents(th.tensor([[Event.LEAVE_SURFACE]], dtype=th.int32))
        self.assertEqual(events.left_surface.tolist(), [[True]])

    def test_touch_batch(self):
        batch = BallTouchBatch(
            surfaces = th.tensor([[1 << int(ContactSurface.FRONT), 0],
                                  [(1 << int(ContactSurface.ROOF)) |
                                   (1 << int(ContactSurface.FRONT)), 0]], dtype=th.int32),
        )

        self.assertEqual(batch.touched.tolist(), [[True, False], [True, False]])
        self.assertEqual(batch.on_surface(ContactSurface.ROOF).tolist(), [[False, False], [True, False]])
        self.assertEqual(batch.on_surface(ContactSurface.FRONT).tolist(), [[True, False], [True, False]])
