import unittest
from types import SimpleNamespace

import torch as th

from carl.events import (
    ArenaSurface, BallEvents, BallTouch, BallTouchBatch, CarlEvents, CarEvents,
    ContactSurface, Event, Goal, Jump, LeaveSurface, StateCondition, SurfaceContact,
)
from carl.training import (
    BallState, CarState, CompiledTrainingPack, Exercise, InitialState, OnSurface,
    Stage, TargetKind, TrainingPack, TrainingPackRuntime,
)


def make_events(
    device: str,
    flags:  list[int],
    left:   list[int] | None = None,
) -> CarlEvents:
    count = len(flags)
    flags = th.tensor(flags, dtype=th.int32, device=device).reshape(count, 1)
    car_surfaces = th.zeros_like(flags)
    if left is not None:
        car_surfaces[:, 0] = th.tensor(left, dtype=th.int32, device=device)
    ball = th.zeros(count, dtype=th.int32, device=device)

    return CarlEvents(
        cars = CarEvents(flags, surface_contacted=th.zeros_like(flags), surface_left=car_surfaces),
        ball = BallEvents(ball, surface_entered=ball, surface_left=ball, goal_side=ball),
    )


class TrainingRuntimeTests(unittest.TestCase):

    def make_pack(self) -> TrainingPack:
        initial = InitialState(ego=CarState(placement=OnSurface()), ball=BallState((0, 0, 100)))

        return TrainingPack(name="test", exercises=[
            Exercise(name="sequence", initial=initial, stages=[
                Stage(Jump(), within_ticks=8),
                Stage(LeaveSurface(surface=ArenaSurface.SIDE_WALL), within_ticks=4,
                      fail_on=[Jump()]),
            ]),
            Exercise(name="simple", initial=initial, stages=[Stage(Jump())]),
        ])

    def test_compilation_and_vectorized_progress(self) -> None:
        for device in (["cpu", "cuda"] if th.cuda.is_available() else ["cpu"]):
            with self.subTest(device=device):
                pack = CompiledTrainingPack.compile(self.make_pack(), device=device)
                self.assertEqual(pack.valid.tolist(), [[True, True], [True, False]])
                runtime = TrainingPackRuntime(pack, 3)
                runtime.reset(th.ones(3, dtype=th.bool, device=device),
                              th.tensor([0, 0, 1], device=device))
                first = runtime.step(make_events(device, [Event.JUMP, 0, Event.JUMP]), 4)
                self.assertEqual(first.advanced.tolist(), [True, False, True])
                self.assertEqual(first.succeeded.tolist(), [False, False, True])
                self.assertEqual(runtime.ticks_remaining.tolist(), [4, 4, 0])

                second = runtime.step(make_events(device, [Event.JUMP, Event.JUMP, Event.JUMP],
                                                  [ArenaSurface.SIDE_WALL, 0, 0]), 4)
                self.assertEqual(second.failed.tolist(), [True, False, False])
                self.assertEqual(second.advanced.tolist(), [False, True, False])
                self.assertEqual(runtime.stage_index.tolist(), [1, 1, 0])

                third = runtime.step(make_events(device, [0, 0, 0], [0, ArenaSurface.SIDE_WALL, 0]), 4)
                self.assertEqual(third.succeeded.tolist(), [False, True, False])

    def test_timeout_and_partial_reset(self) -> None:
        pack = CompiledTrainingPack.compile(self.make_pack(), device="cpu")
        runtime = TrainingPackRuntime(pack, 2)
        runtime.reset(th.tensor([True, True]), th.tensor([0, 0]))
        runtime.step(make_events("cpu", [0, 0]), 4)
        result = runtime.step(make_events("cpu", [0, Event.JUMP]), 4)
        self.assertEqual(result.timed_out.tolist(), [True, False])
        runtime.reset(th.tensor([True, False]), th.tensor([1, 0]))
        self.assertEqual(runtime.exercise_index.tolist(), [1, 0])
        self.assertEqual(runtime.stage_index.tolist(), [0, 1])

    def test_minimum_ticks_stage_and_exercise_time(self) -> None:
        for device in (["cpu", "cuda"] if th.cuda.is_available() else ["cpu"]):
            pack = self.make_pack()
            pack.exercises[0].stages = [
                Stage(Jump(), min_ticks=4, within_ticks=20, from_exercise_start=True),
                Stage(Jump(), min_ticks=4),
            ]
            compiled = CompiledTrainingPack.compile(pack, device=device)
            runtime = TrainingPackRuntime(compiled, 1)
            runtime.reset(th.tensor([True], device=device), th.tensor([0], device=device))
            event = make_events(device, [Event.JUMP])
            self.assertEqual(runtime.target_sequence().min_ticks[0, :2].tolist(), [4, 4])
            self.assertFalse(runtime.step(event, 3).advanced.item())
            self.assertFalse(runtime.step(event, 2).advanced.item())
            self.assertTrue(runtime.step(event, 1).advanced.item())
            self.assertFalse(runtime.step(event, 4).advanced.item())
            self.assertTrue(runtime.step(event, 1).succeeded.item())
            runtime.reset(th.tensor([True], device=device), th.tensor([0], device=device))
            self.assertEqual(runtime.stage_elapsed_ticks.item(), 0)

    def test_invalid_minimum_ticks(self) -> None:
        for minimum in (-1, 1.5, True, 10):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                Stage(Jump(), min_ticks=minimum, within_ticks=10)
        pack = self.make_pack()
        pack.exercises[0].stages[0].min_ticks = -1
        with self.assertRaises(ValueError):
            CompiledTrainingPack.compile(pack, device="cpu")

    def test_unsupported_touch_height_filter_is_not_exposed(self) -> None:
        with self.assertRaises(TypeError):
            BallTouch(min_height=300)

    def test_touch_capture_and_goal_direction(self) -> None:
        pack = self.make_pack()
        pack.exercises[0].stages = [
            Stage(BallTouch(surface=ContactSurface.ROOF)), Stage(Goal()),
        ]
        compiled = CompiledTrainingPack.compile(pack, device="cpu")
        runtime = TrainingPackRuntime(compiled, 2)
        runtime.reset(th.tensor([True, True]), th.tensor([0, 0]))
        events = make_events("cpu", [0, 0])

        with self.assertRaisesRegex(ValueError, "touch capture"):
            runtime.step(events, 4)

        events.ball_touches = BallTouchBatch(th.tensor([[1 << int(ContactSurface.ROOF)], [0]]))
        self.assertEqual(runtime.step(events, 4).advanced.tolist(), [True, False])
        events.ball.flags[:] = int(Event.GOAL)
        events.ball.goal_side = th.tensor([-1, 1])
        self.assertEqual(runtime.step(events, 4).succeeded.tolist(), [False, False])
        events.ball.goal_side[0] = 1
        self.assertEqual(runtime.step(events, 4).succeeded.tolist(), [True, False])

    def test_touch_speed_change_uses_matching_tick_and_surface(self) -> None:
        for device in (["cpu", "cuda"] if th.cuda.is_available() else ["cpu"]):
            pack = self.make_pack()
            pack.exercises[0].stages = [Stage(BallTouch(surface=ContactSurface.FRONT, min_speed_gain=-100, max_speed_gain=200))]
            compiled = CompiledTrainingPack.compile(pack, device=device)
            runtime = TrainingPackRuntime(compiled, 4)
            runtime.reset(th.ones(4, dtype=th.bool, device=device), th.zeros(4, dtype=th.long, device=device))
            events = make_events(device, [Event.BALL_TOUCH] * 4)
            front = 1 << int(ContactSurface.FRONT)
            roof = 1 << int(ContactSurface.ROOF)
            surfaces = th.tensor([[[front], [front], [front], [front]],
                                  [[0], [0], [0], [roof]]], device=device)
            before = th.tensor([[[600.], [600.], [600.], [600.]],
                                [[0.], [0.], [0.], [600.]]], device=device)
            speeds = th.tensor([[[500.], [800.], [801.], [400.]],
                                [[0.], [0.], [0.], [600.]]], device=device)
            events.ball_touches = BallTouchBatch(surfaces=surfaces.sum(dim=0),
                                                tick_surfaces=surfaces, tick_speed=speeds,
                                                tick_speed_before=before)
            target = runtime.target_sequence()
            self.assertEqual(target.stages.gain_min[0, 0].item(), -100)
            self.assertEqual(target.stages.gain_max[0, 0].item(), 200)
            self.assertEqual(runtime.step(events, 4).succeeded.tolist(), [True, True, False, False])

    def test_multiple_touch_parts_match_any_allowed_face_at_the_qualified_tick(self) -> None:
        pack = self.make_pack()
        pack.exercises[0].stages = [Stage(BallTouch(
            (ContactSurface.FRONT, ContactSurface.ROOF), min_speed_gain=50, max_speed_gain=200,
        ))]
        compiled = CompiledTrainingPack.compile(pack, device="cpu")
        runtime = TrainingPackRuntime(compiled, 5)
        runtime.reset(th.ones(5, dtype=th.bool), th.zeros(5, dtype=th.long))
        events = make_events("cpu", [Event.BALL_TOUCH] * 5)
        front, roof, rear = (1 << int(part) for part in (ContactSurface.FRONT, ContactSurface.ROOF, ContactSurface.REAR))
        surfaces = th.tensor([[[front], [roof], [rear], [front], [front]],
                              [[0], [0], [0], [roof], [rear]]])
        before = th.full((2, 5, 1), 500.)
        speed = th.tensor([[[550.], [700.], [600.], [520.], [520.]],
                           [[500.], [500.], [500.], [600.], [600.]]])
        events.ball_touches = BallTouchBatch(surfaces=surfaces.sum(dim=0), tick_surfaces=surfaces,
                                             tick_speed=speed, tick_speed_before=before)
        self.assertEqual(compiled.car_parts[0, 0], front | roof)
        # The fifth row must not borrow the rear contact's valid gain to qualify
        # its earlier underpowered front touch.
        self.assertEqual(runtime.step(events, 4).succeeded.tolist(), [True, True, False, True, False])

    def test_all_parts_tuple_uses_wildcard_flags_and_invalid_parts_are_rejected(self) -> None:
        pack = self.make_pack()
        pack.exercises[0].stages = [Stage(BallTouch(tuple(ContactSurface)))]
        compiled = CompiledTrainingPack.compile(pack, device="cpu")
        self.assertFalse(compiled.needs_touch)
        runtime = TrainingPackRuntime(compiled, 1)
        runtime.reset(th.ones(1, dtype=th.bool), th.zeros(1, dtype=th.long))
        self.assertTrue(runtime.step(make_events("cpu", [Event.BALL_TOUCH]), 1).succeeded.item())
        for surface in ((), (99,)):
            pack.exercises[0].stages[0].event = BallTouch(surface)
            with self.subTest(surface=surface), self.assertRaises(ValueError):
                CompiledTrainingPack.compile(pack, device="cpu")

    def test_invalid_touch_speed_bounds(self) -> None:
        for condition in (BallTouch(min_speed_gain=float("nan")), BallTouch(max_speed_gain=float("inf")),
                          BallTouch(min_speed_gain=10, max_speed_gain=9)):
            pack = self.make_pack()
            pack.exercises[0].stages[0].event = condition
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                CompiledTrainingPack.compile(pack, device="cpu")

    def test_state_conditions_on_gpu_and_cpu(self) -> None:
        for device in (["cpu", "cuda"] if th.cuda.is_available() else ["cpu"]):
            with self.subTest(device=device):
                pack = self.make_pack()
                pack.exercises[0].stages = [Stage(
                    StateCondition("car_speed", minimum=500, maximum=500),
                    fail_on=[StateCondition("ball_height", maximum=300)],
                )]
                pack.exercises[0].fail_on = [StateCondition("car_boost", maximum=10)]
                compiled = CompiledTrainingPack.compile(pack, device=device)
                runtime = TrainingPackRuntime(compiled, 3)
                runtime.reset(th.ones(3, dtype=th.bool, device=device), th.zeros(3, dtype=th.int64, device=device))
                events = make_events(device, [0, 0, 0])

                with self.assertRaisesRegex(ValueError, "transition state"):
                    runtime.step(events, 4)

                state = SimpleNamespace(
                    ball_position = th.tensor([[0, 0, 500], [0, 0, 300], [0, 0, 500]], device=device),
                    ball_velocity = th.zeros((3, 3), device=device),
                    car_position  = th.zeros((3, 1, 3), device=device),
                    car_velocity  = th.tensor([[[300, 400, 0]]] * 3, device=device, dtype=th.float32),
                    car_boost     = th.tensor([[50], [50], [10]], device=device),
                )
                if device == "cuda":
                    previous = th.cuda.get_sync_debug_mode()

                    try:
                        th.cuda.set_sync_debug_mode("error")
                        result = runtime.step(events, 4, state=state)
                    finally:
                        th.cuda.set_sync_debug_mode(previous)
                else:
                    result = runtime.step(events, 4, state=state)

                self.assertEqual(result.succeeded.tolist(), [True, False, False])
                self.assertEqual(result.failed.tolist(), [False, True, True])

    def test_invalid_state_condition_fails_compilation(self) -> None:
        for condition in (StateCondition("invalid", minimum=0), StateCondition("ball_height"),
                          StateCondition("ball_height", minimum=2, maximum=1),
                          StateCondition("car_boost", minimum=float("nan"))):
            pack = self.make_pack()
            pack.exercises[0].stages[0].event = condition

            with self.subTest(condition=condition), self.assertRaises(ValueError):
                CompiledTrainingPack.compile(pack, device="cpu")

    def test_unspecified_surface_includes_other_car_surfaces(self) -> None:
        pack = self.make_pack()
        pack.exercises[0].stages = [Stage(SurfaceContact())]
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack, device="cpu"), 1)
        runtime.reset(th.tensor([True]), th.tensor([0]))
        events = make_events("cpu", [0])
        events.cars.surface_contacted[0, 0] = int(ArenaSurface.OTHER)

        self.assertEqual(runtime.step(events, 4).succeeded.tolist(), [True])

    def test_wildcards_use_flags_without_touch_capture(self) -> None:
        for device in (["cpu", "cuda"] if th.cuda.is_available() else ["cpu"]):
            for condition, flag in ((BallTouch(), Event.BALL_TOUCH), (LeaveSurface(), Event.LEAVE_SURFACE)):
                with self.subTest(device=device, condition=condition):
                    pack = self.make_pack()
                    pack.exercises[0].stages = [Stage(condition)]
                    compiled = CompiledTrainingPack.compile(pack, device=device)
                    self.assertFalse(compiled.needs_touch)
                    self.assertFalse(compiled.needs_touch_speed)
                    runtime = TrainingPackRuntime(compiled, 2)
                    runtime.reset(th.ones(2, dtype=th.bool, device=device), th.zeros(2, dtype=th.long, device=device))
                    events = make_events(device, [flag, 0], [0, ArenaSurface.SIDE_WALL])
                    self.assertEqual(runtime.step(events, 1).succeeded.tolist(), [True, False])
                    runtime.reset(th.ones(2, dtype=th.bool, device=device), th.zeros(2, dtype=th.long, device=device))
                    events.ball_touches = BallTouchBatch(th.tensor([[0], [63]], device=device))
                    self.assertEqual(runtime.step(events, 1).succeeded.tolist(), [True, False])

    def test_explicit_all_surfaces_car_leave_is_not_a_flag_wildcard(self) -> None:
        pack = self.make_pack()
        pack.exercises[0].stages = [Stage(LeaveSurface(ArenaSurface(31)))]
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack, device="cpu"), 2)
        runtime.reset(th.ones(2, dtype=th.bool), th.zeros(2, dtype=th.long))
        result = runtime.step(make_events("cpu", [Event.LEAVE_SURFACE, 0], [0, ArenaSurface.OTHER]), 1)
        self.assertEqual(result.succeeded.tolist(), [False, True])

    def test_touch_capture_requirements_for_every_group(self) -> None:
        for condition, touch, speed in ((BallTouch(), False, False),
                                        (BallTouch(ContactSurface.FRONT), True, False),
                                        (BallTouch(min_speed_gain=0), True, True),
                                        (BallTouch(max_speed_gain=0), True, True)):
            for group in ("event", "repeat", "failure"):
                with self.subTest(condition=condition, group=group):
                    pack = self.make_pack()
                    stage = Stage(condition) if group == "event" else Stage(Jump())
                    if group == "repeat":
                        stage.repeat_on = condition
                    if group == "failure":
                        stage.fail_on = [condition]
                    pack.exercises[0].stages = [stage]
                    compiled = CompiledTrainingPack.compile(pack, device="cpu")
                    self.assertEqual(compiled.needs_touch, touch)
                    self.assertEqual(compiled.needs_touch_speed, speed)

    def test_no_sentinels_and_distinct_state_and_gain_bounds(self) -> None:
        pack = self.make_pack()
        pack.exercises[0].stages = [Stage(
            StateCondition("car_speed", minimum=0),
            repeat_on=BallTouch(max_speed_gain=0), max_repeats=0, max_total_matches=1,
            fail_on=[StateCondition("ball_height", maximum=100), BallTouch(min_speed_gain=-50)],
        )]
        compiled = CompiledTrainingPack.compile(pack, device="cpu")
        self.assertEqual(compiled.kind[0, 0].item(), TargetKind.STATE)
        self.assertEqual(compiled.repeat_kind[0, 0].item(), TargetKind.TOUCH)
        for prefix in ("", "repeat_", "failure_"):
            for bound in ("state_min", "state_max", "gain_min", "gain_max"):
                values = getattr(compiled, prefix + bound)
                has = getattr(compiled, prefix + "has_" + bound)
                self.assertTrue(th.isfinite(values).all().item())
                self.assertFalse(values[~has].any().item())
        self.assertTrue(compiled.has_state_min[0, 0].item())
        self.assertTrue(compiled.repeat_has_gain_max[0, 0].item())
        self.assertTrue(compiled.has_max_repeats[0, 0].item())
        self.assertFalse(compiled.has_max_repeats[1].any().item())
        for value, has in ((compiled.deadline, compiled.has_deadline),
                           (compiled.max_repeats, compiled.has_max_repeats),
                           (compiled.max_total_matches, compiled.has_max_total_matches)):
            self.assertFalse(value[~has].any().item())

    def test_gain_qualified_repeats_count_matching_ticks_and_limits(self) -> None:
        for device in (["cpu", "cuda"] if th.cuda.is_available() else ["cpu"]):
            pack = self.make_pack()
            pack.exercises[0].stages = [Stage(
                Goal(), within_ticks=8, repeat_on=BallTouch(ContactSurface.FRONT, min_speed_gain=100),
                max_repeats=2, max_total_matches=3,
            )]
            runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack, device=device), 1)
            runtime.reset(th.ones(1, dtype=th.bool, device=device), th.zeros(1, dtype=th.long, device=device))
            events = make_events(device, [Event.BALL_TOUCH])
            surfaces = th.full((2, 1, 1), 1 << int(ContactSurface.FRONT), device=device)
            events.ball_touches = BallTouchBatch(
                surfaces=surfaces.sum(dim=0), tick_surfaces=surfaces,
                tick_speed=th.full((2, 1, 1), 100.0, device=device), tick_speed_before=th.zeros(2, 1, 1, device=device),
            )
            first = runtime.step(events, 4)
            self.assertTrue(first.repeated.item())
            self.assertEqual(runtime.repeat_count.item(), 2)
            self.assertEqual(runtime.total_matches.item(), 2)
            self.assertEqual(runtime.ticks_remaining.item(), 8)
            self.assertTrue(runtime.step(events, 4).failed.item())

    def test_total_match_limit_requires_a_touch_gain_bound(self) -> None:
        for condition in (BallTouch(), BallTouch(ContactSurface.FRONT), Jump()):
            pack = self.make_pack()
            pack.exercises[0].stages = [Stage(condition, max_total_matches=1)]
            with self.subTest(condition=condition), self.assertRaisesRegex(ValueError, "speed-qualified"):
                CompiledTrainingPack.compile(pack, device="cpu")


if __name__ == "__main__":
    unittest.main()
