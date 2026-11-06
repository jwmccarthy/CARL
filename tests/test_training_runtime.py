import unittest
from types import SimpleNamespace

import torch as th

from carl.events import (
    ArenaSurface, BallEvents, BallTouch, BallTouchBatch, CarlEvents, CarEvents,
    ContactSurface, Event, Goal, Jump, LeaveSurface, StateCondition, SurfaceContact,
)
from carl.training import (
    BallState, CarState, CompiledTrainingPack, Exercise, InitialState, OnSurface,
    Stage, TrainingPack, TrainingPackRuntime,
)


def make_events(device, flags, left=None):
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

    def make_pack(self):
        initial = InitialState(ego=CarState(placement=OnSurface()), ball=BallState((0, 0, 100)))
        return TrainingPack(name="test", exercises=[
            Exercise(name="sequence", initial=initial, stages=[
                Stage(Jump(), within_ticks=8),
                Stage(LeaveSurface(surface=ArenaSurface.SIDE_WALL), within_ticks=4,
                      fail_on=[Jump()]),
            ]),
            Exercise(name="simple", initial=initial, stages=[Stage(Jump())]),
        ])

    def test_compilation_and_vectorized_progress(self):
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
                self.assertEqual(runtime.ticks_remaining.tolist(), [4, 4, -1])

                second = runtime.step(make_events(device, [Event.JUMP, Event.JUMP, Event.JUMP],
                                                  [ArenaSurface.SIDE_WALL, 0, 0]), 4)
                self.assertEqual(second.failed.tolist(), [True, False, False])
                self.assertEqual(second.advanced.tolist(), [False, True, False])
                self.assertEqual(runtime.stage_index.tolist(), [1, 1, 0])

                third = runtime.step(make_events(device, [0, 0, 0], [0, ArenaSurface.SIDE_WALL, 0]), 4)
                self.assertEqual(third.succeeded.tolist(), [False, True, False])

    def test_timeout_and_partial_reset(self):
        pack = CompiledTrainingPack.compile(self.make_pack(), device="cpu")
        runtime = TrainingPackRuntime(pack, 2)
        runtime.reset(th.tensor([True, True]), th.tensor([0, 0]))
        runtime.step(make_events("cpu", [0, 0]), 4)
        result = runtime.step(make_events("cpu", [0, Event.JUMP]), 4)
        self.assertEqual(result.timed_out.tolist(), [True, False])
        runtime.reset(th.tensor([True, False]), th.tensor([1, 0]))
        self.assertEqual(runtime.exercise_index.tolist(), [1, 0])
        self.assertEqual(runtime.stage_index.tolist(), [0, 1])

    def test_minimum_ticks_stage_and_exercise_time(self):
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

    def test_invalid_minimum_ticks(self):
        for minimum in (-1, 1.5, True, 10):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                Stage(Jump(), min_ticks=minimum, within_ticks=10)
        pack = self.make_pack()
        pack.exercises[0].stages[0].min_ticks = -1
        with self.assertRaises(ValueError):
            CompiledTrainingPack.compile(pack, device="cpu")

    def test_unsupported_touch_filter_fails_during_compilation(self):
        pack = self.make_pack()
        pack.exercises[0].stages[0].event = BallTouch(min_height=300)
        with self.assertRaisesRegex(ValueError, "not supported"):
            CompiledTrainingPack.compile(pack, device="cpu")

    def test_touch_capture_and_goal_direction(self):
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

    def test_touch_speed_change_uses_matching_tick_and_surface(self):
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
            self.assertEqual(target.stages.min_touch_speed_gain[0, 0].item(), -100)
            self.assertEqual(target.stages.max_touch_speed_gain[0, 0].item(), 200)
            self.assertEqual(runtime.step(events, 4).succeeded.tolist(), [True, True, False, False])

    def test_invalid_touch_speed_bounds(self):
        for condition in (BallTouch(min_speed_gain=float("nan")), BallTouch(max_speed_gain=float("inf")),
                          BallTouch(min_speed_gain=10, max_speed_gain=9)):
            pack = self.make_pack()
            pack.exercises[0].stages[0].event = condition
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                CompiledTrainingPack.compile(pack, device="cpu")

    def test_state_conditions_on_gpu_and_cpu(self):
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

    def test_invalid_state_condition_fails_compilation(self):
        for condition in (StateCondition("invalid", minimum=0), StateCondition("ball_height"),
                          StateCondition("ball_height", minimum=2, maximum=1),
                          StateCondition("car_boost", minimum=float("nan"))):
            pack = self.make_pack()
            pack.exercises[0].stages[0].event = condition

            with self.subTest(condition=condition), self.assertRaises(ValueError):
                CompiledTrainingPack.compile(pack, device="cpu")

    def test_unspecified_surface_includes_other_car_surfaces(self):
        pack = self.make_pack()
        pack.exercises[0].stages = [Stage(SurfaceContact())]
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack, device="cpu"), 1)
        runtime.reset(th.tensor([True]), th.tensor([0]))
        events = make_events("cpu", [0])
        events.cars.surface_contacted[0, 0] = int(ArenaSurface.OTHER)

        self.assertEqual(runtime.step(events, 4).succeeded.tolist(), [True])


if __name__ == "__main__":
    unittest.main()
