import unittest

import torch as th

from carl.events import (
    ArenaSurface, BallEvents, BallTouch, BallTouchBatch, CarlEvents, CarEvents,
    ContactSurface, Event, Jump,
)
from carl.training import (
    BallState, CarState, CompiledTrainingPack, Exercise, InitialState, LeaveSurface,
    OnSurface, Stage, StateCondition, StateMetric, TargetEntity, TargetKind,
    TrainingPack, TrainingPackRuntime,
)


class TrainingTargetTests(unittest.TestCase):

    def make_runtime(self, device):
        initial = InitialState(ego=CarState(placement=OnSurface()), ball=BallState((0, 0, 500)))
        pack = TrainingPack(name="targets", exercises=[
            Exercise(
                name    = "mechanic",
                initial = initial,
                fail_on = [StateCondition("ball_height", maximum=300)],
                stages  = [
                    Stage(Jump(), within_ticks=8, fail_on=[LeaveSurface(surface=ArenaSurface.SIDE_WALL)]),
                    Stage(BallTouch(surface=ContactSurface.ROOF)),
                ],
            ),
            Exercise(
                name    = "speed",
                initial = initial,
                stages  = [Stage(StateCondition("car_speed", minimum=1500, maximum=2300))],
            ),
        ])
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack, device=device), n_sim=2)
        runtime.reset(th.ones(2, dtype=th.bool, device=device), th.tensor([0, 1], device=device))

        return runtime

    def test_full_sequence_and_failure_metadata(self):
        for device in (["cpu", "cuda"] if th.cuda.is_available() else ["cpu"]):
            with self.subTest(device=device):
                runtime = self.make_runtime(device)

                if device == "cuda":
                    previous = th.cuda.get_sync_debug_mode()

                    try:
                        th.cuda.set_sync_debug_mode("error")
                        target = runtime.target_sequence()
                    finally:
                        th.cuda.set_sync_debug_mode(previous)
                else:
                    target = runtime.target_sequence()

                self.assertEqual(target.stages.valid.tolist(), [[True, True], [True, False]])
                self.assertEqual(target.stages.kind.tolist(), [
                    [TargetKind.CAR_EVENT, TargetKind.TOUCH_SURFACE],
                    [TargetKind.STATE_VALUE, TargetKind.PADDING],
                ])
                self.assertEqual(target.stages.entity.tolist(), [
                    [TargetEntity.CAR, TargetEntity.CAR], [TargetEntity.CAR, TargetEntity.NONE],
                ])
                self.assertEqual(target.stages.event_mask[0, 0].item(), Event.JUMP)
                self.assertEqual(target.stages.touch_surface_mask[0, 1].item(), 1 << int(ContactSurface.ROOF))
                self.assertEqual(target.stages.state_metric[1, 0].item(), StateMetric.CAR_SPEED)
                self.assertTrue(target.stages.has_state_metric[1, 0].item())
                self.assertEqual(target.stages.minimum[1, 0].item(), 1500)
                self.assertEqual(target.stages.maximum[1, 0].item(), 2300)
                self.assertEqual(target.deadline_ticks.tolist(), [[8, 0], [0, 0]])
                self.assertEqual(target.has_deadline.tolist(), [[True, False], [False, False]])
                self.assertEqual(target.ticks_remaining.tolist(), [8, 0])
                self.assertEqual(target.has_remaining.tolist(), [True, False])

                self.assertEqual(target.failures.valid[0, 0].tolist(), [True, True])
                self.assertEqual(target.failure_exercise_wide[0, 0].tolist(), [True, False])
                self.assertEqual(target.failures.state_metric[0, 0, 0].item(), StateMetric.BALL_HEIGHT)
                self.assertEqual(target.failures.maximum[0, 0, 0].item(), 300)
                self.assertFalse(target.failures.has_minimum[0, 0, 0].item())
                self.assertEqual(target.failures.minimum[0, 0, 0].item(), 0)
                self.assertEqual(target.failures.arena_surface_mask[0, 0, 1].item(), ArenaSurface.SIDE_WALL)
                self.assertFalse(target.failures.valid[1].any().item())
                self.assertTrue(th.isfinite(target.failures.minimum).all().item())
                self.assertTrue(th.isfinite(target.failures.maximum).all().item())

    def test_progress_and_partial_reassignment(self):
        runtime = self.make_runtime("cpu")
        zeros = th.zeros((2, 1), dtype=th.int32)
        ball = th.zeros(2, dtype=th.int32)
        events = CarlEvents(
            cars         = CarEvents(th.tensor([[Event.JUMP], [0]], dtype=th.int32),
                                     surface_contacted=zeros, surface_left=zeros),
            ball         = BallEvents(ball, surface_entered=ball, surface_left=ball, goal_side=ball),
            ball_touches = BallTouchBatch(zeros),
        )
        runtime.step(events, 4, state=type("State", (), {
            "ball_position": th.tensor([[0, 0, 500]] * 2), "ball_velocity": th.zeros(2, 3),
            "car_position": th.zeros(2, 1, 3), "car_velocity": th.zeros(2, 1, 3),
            "car_boost": th.zeros(2, 1),
        })())
        before = runtime.target_sequence()

        self.assertEqual(before.stage_index.tolist(), [1, 0])
        self.assertEqual(before.stages.kind[0, 0].item(), TargetKind.CAR_EVENT)
        self.assertEqual(before.stages.kind[0, 1].item(), TargetKind.TOUCH_SURFACE)

        runtime.reset(th.tensor([True, False]), th.tensor([1, 0]))
        after = runtime.target_sequence()

        self.assertEqual(after.exercise_index.tolist(), [1, 1])
        self.assertEqual(after.stage_index.tolist(), [0, 0])
        self.assertEqual(before.exercise_index.tolist(), [0, 1])
        self.assertEqual(before.stage_index.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
