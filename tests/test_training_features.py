import math
import unittest

import torch as th

import carl
from carl.events import (
    ArenaSurface, BallTouch, Condition, ContactSurface, Demoed, Dodge, DoubleJump,
    FlipReset, Goal, Jump, LeaveSurface, StateCondition, SurfaceContact,
)
from carl.training import (
    POLICY_FEATURE_DIM, POLICY_FEATURE_SLICES, BallState, CarState,
    CompiledTrainingPack, Exercise, InitialState, OnSurface, Stage, TargetEntity,
    TargetKind, TrainingPack, TrainingPackRuntime, policy_features,
)
from carl.training.targets import FEATURE_SCALES
from test_training_runtime import make_events


DEVICES = ["cpu", "cuda"] if th.cuda.is_available() else ["cpu"]


class PolicyFeatureTests(unittest.TestCase):

    def make_runtime(
        self,
        stages:    list[list[Stage | Condition]],
        device:    str = "cpu",
        *,
        fail_on:   list[Condition] | None = None,
        team_sign: int | th.Tensor = 1,
    ) -> TrainingPackRuntime:
        initial = InitialState(CarState(OnSurface()), BallState((0, 0, 500)))
        pack = TrainingPack("features", [Exercise(
            name=str(index), initial=initial, fail_on=list(fail_on or []),
            stages=[stage if isinstance(stage, Stage) else Stage(stage) for stage in sequence],
        ) for index, sequence in enumerate(stages)])
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack, device), len(stages), team_sign=team_sign)
        runtime.reset(th.ones(len(stages), dtype=th.bool, device=device), th.arange(len(stages), device=device))

        return runtime

    def test_touch_and_leave_variants_differ_only_in_qualifiers(self) -> None:
        for device in DEVICES:
            with self.subTest(device=device):
                runtime = self.make_runtime([
                    [BallTouch()], [BallTouch(ContactSurface.FRONT)],
                    [BallTouch(ContactSurface.FRONT, min_speed_gain=500)],
                    [LeaveSurface()], [LeaveSurface(ArenaSurface.SIDE_WALL)],
                ], device)
                target = runtime.target_sequence().stages
                self.assertEqual(target.kind[:, 0].tolist(), [TargetKind.TOUCH] * 3 + [TargetKind.LEAVE] * 2)
                self.assertEqual(target.car_parts[:, 0].tolist(), [63, 1 << int(ContactSurface.FRONT),
                                                                  1 << int(ContactSurface.FRONT), 0, 0])
                self.assertEqual(target.arena_surfaces[:, 0].tolist(), [0, 0, 0, 31, int(ArenaSurface.SIDE_WALL)])
                vectors = policy_features(runtime).window[:, 0]
                allowed = th.zeros(POLICY_FEATURE_DIM, dtype=th.bool, device=device)
                for field in ("car_parts", "gain_min", "gain_max"):
                    allowed[POLICY_FEATURE_SLICES[field]] = True
                th.testing.assert_close(vectors[0, ~allowed], vectors[1, ~allowed])
                th.testing.assert_close(vectors[1, ~allowed], vectors[2, ~allowed])
                allowed[:] = False
                allowed[POLICY_FEATURE_SLICES["arena_surfaces"]] = True
                th.testing.assert_close(vectors[3, ~allowed], vectors[4, ~allowed])
                th.testing.assert_close(vectors[2, POLICY_FEATURE_SLICES["gain_min"]],
                                        vectors.new_tensor([500 / carl.BALL_MAX_SPEED, 1]))
                self.assertFalse(vectors[:, POLICY_FEATURE_SLICES["event_flags"]].any().item())

    def test_small_edits_are_closer_than_a_different_kind(self) -> None:
        for device in DEVICES:
            runtime = self.make_runtime([
                [StateCondition("car_speed", minimum=500)],
                [StateCondition("car_speed", minimum=501)],
                [Jump()],
                [SurfaceContact(ArenaSurface.SIDE_WALL)],
                [SurfaceContact(ArenaSurface.SIDE_WALL | ArenaSurface.BACK_WALL)],
                [LeaveSurface(ArenaSurface.SIDE_WALL)],
            ], device)
            vectors = policy_features(runtime).window[:, 0]
            self.assertLess((vectors[0] - vectors[1]).norm().item(), (vectors[0] - vectors[2]).norm().item())
            self.assertLess((vectors[3] - vectors[4]).norm().item(), (vectors[3] - vectors[5]).norm().item())

    def test_multi_part_touch_is_encoded_as_a_multi_hot_mask(self) -> None:
        runtime = self.make_runtime([[BallTouch((ContactSurface.FRONT, ContactSurface.ROOF))]])
        parts = policy_features(runtime).window[0, 0, POLICY_FEATURE_SLICES["car_parts"]]
        expected = th.zeros(len(ContactSurface))
        expected[ContactSurface.FRONT] = expected[ContactSurface.ROOF] = 1
        th.testing.assert_close(parts, expected)

    def test_window_shifts_and_current_failures_are_not_duplicated(self) -> None:
        for device in DEVICES:
            runtime = self.make_runtime([
                [Stage(Jump(), fail_on=[LeaveSurface(ArenaSurface.SIDE_WALL)]),
                 Stage(Dodge(), fail_on=[SurfaceContact(ArenaSurface.CEILING)]),
                 FlipReset(), Goal()],
                [Goal()],
            ], device, fail_on=[Demoed()])
            before = policy_features(runtime)
            self.assertEqual(before.window.shape, (2, 3, POLICY_FEATURE_DIM))
            self.assertEqual(before.failures.shape, (2, 2, POLICY_FEATURE_DIM))
            self.assertEqual(before.window_valid.tolist(), [[1, 1, 1], [1, 0, 0]])
            self.assertEqual(before.failure_valid.tolist(), [[1, 1], [1, 0]])
            self.assertFalse(before.window[1, 1:].any().item())
            self.assertFalse(before.failures[1, 1:].any().item())

            runtime.step(make_events(device, [carl.EVENT_JUMP, 0]), 1)
            after = policy_features(runtime)
            th.testing.assert_close(after.window[0, :2], before.window[0, 1:])
            th.testing.assert_close(after.failures[:, 0], before.failures[:, 0])
            self.assertFalse(th.equal(after.failures[0, 1], before.failures[0, 1]))
            runtime.step(make_events(device, [carl.EVENT_DODGE, 0]), 1)
            runtime.step(make_events(device, [carl.EVENT_FLIP_RESET, 0]), 1)
            last = policy_features(runtime, K=6)
            self.assertEqual(last.window_valid[0].tolist(), [1, 0, 0, 0, 0, 0])
            self.assertFalse(last.window[:, 1:].any().item())

    def test_bound_scales_entities_and_presence_are_explicit(self) -> None:
        metrics = list(FEATURE_SCALES)[:9]
        for device in DEVICES:
            runtime = self.make_runtime([
                [StateCondition(metric, minimum=-FEATURE_SCALES[metric], maximum=0)] for metric in metrics
            ] + [[BallTouch(max_speed_gain=-carl.BALL_MAX_SPEED)]], device)
            vectors = policy_features(runtime).window[:, 0]
            self.assertEqual(runtime.pack.entity[:, 0].tolist(), [TargetEntity.BALL] * 4 + [TargetEntity.CAR] * 6)
            th.testing.assert_close(vectors[:9, POLICY_FEATURE_SLICES["state_metric"]], th.eye(9, device=device))
            th.testing.assert_close(vectors[:9, POLICY_FEATURE_SLICES["state_min"]],
                                    vectors.new_tensor([[-1, 1]] * 9))
            th.testing.assert_close(vectors[:9, POLICY_FEATURE_SLICES["state_max"]],
                                    vectors.new_tensor([[0, 1]] * 9))
            th.testing.assert_close(vectors[9, POLICY_FEATURE_SLICES["gain_max"]], vectors.new_tensor([-1, 1]))
            self.assertFalse(vectors[:9, POLICY_FEATURE_SLICES["gain_max"]].any().item())
            self.assertFalse(vectors[9, POLICY_FEATURE_SLICES["state_metric"]].any().item())

    def test_deadline_minimum_ticks_and_slot_zero_progress(self) -> None:
        for device in DEVICES:
            stage = Stage(Jump(), within_ticks=120, min_ticks=4, repeat_on=BallTouch(min_speed_gain=0),
                          max_repeats=4, max_total_matches=8)
            runtime = self.make_runtime([[stage, stage], [Jump()]], device)
            runtime.ticks_remaining[0] = 60
            runtime.repeat_count[:] = 2
            runtime.total_matches[:] = 3
            features = policy_features(runtime)
            vectors = features.window
            th.testing.assert_close(vectors[0, :2, POLICY_FEATURE_SLICES["deadline"]],
                                    vectors.new_tensor([[math.log1p(120) / math.log1p(carl.REGULATION_TICKS), 1]] * 2))
            th.testing.assert_close(vectors[0, :2, POLICY_FEATURE_SLICES["min_ticks"]],
                                    vectors.new_tensor([[math.log1p(4) / math.log1p(carl.REGULATION_TICKS), 1]] * 2))
            th.testing.assert_close(vectors[0, 0, POLICY_FEATURE_SLICES["progress"]],
                                    vectors.new_tensor([0.5, 0.5, 3 / 8]))
            self.assertFalse(vectors[:, 1:, POLICY_FEATURE_SLICES["progress"]].any().item())
            self.assertFalse(vectors[1, :, POLICY_FEATURE_SLICES["progress"]].any().item())
            self.assertEqual(vectors.dtype, th.float32)
            self.assertEqual(features.failures.dtype, th.float32)
            self.assertTrue(th.isfinite(vectors).all().item())

    def test_goal_side_is_relative_to_team_and_identity_is_not_encoded(self) -> None:
        for device in DEVICES:
            runtime = self.make_runtime([[Goal()], [Goal()], [Jump()], [Jump()]], device,
                                        team_sign=th.tensor([1, -1, 1, 1], device=device))
            features = policy_features(runtime)
            self.assertEqual(features.window[:, 0, POLICY_FEATURE_SLICES["goal_side"]].flatten().tolist(),
                             [1, -1, 0, 0])
            th.testing.assert_close(features.window[2], features.window[3])
            th.testing.assert_close(features.failures[2], features.failures[3])

    def test_event_flags_are_multi_hot_only_for_event_conditions(self) -> None:
        runtime = self.make_runtime([[Jump(), DoubleJump(), Dodge(), FlipReset(), Demoed(), Goal(), BallTouch()]])
        features = policy_features(runtime, K=7).window[0]
        flags = features[:, POLICY_FEATURE_SLICES["event_flags"]]
        th.testing.assert_close(flags[:5], th.eye(5))
        self.assertFalse(flags[5:].any().item())

    @unittest.skipUnless(th.cuda.is_available(), "requires a CUDA GPU")
    def test_features_do_not_synchronize_with_host_or_mutate_progress(self) -> None:
        runtime = self.make_runtime([[Stage(Jump(), within_ticks=8), Goal()]], "cuda")
        before = {name: value.clone() for name, value in vars(runtime).items() if isinstance(value, th.Tensor)}
        previous = th.cuda.get_sync_debug_mode()

        try:
            th.cuda.set_sync_debug_mode("error")
            features = policy_features(runtime)
        finally:
            th.cuda.set_sync_debug_mode(previous)

        for name, value in before.items():
            th.testing.assert_close(getattr(runtime, name), value)
        self.assertEqual(features.window.device, runtime.pack.kind.device)

    def test_invalid_feature_options(self) -> None:
        runtime = self.make_runtime([[Jump()]])
        for K in (0, -1, True, 1.5):
            with self.subTest(K=K), self.assertRaises(ValueError):
                policy_features(runtime, K=K)
        for cap in (0, -1, float("inf"), float("nan")):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                policy_features(runtime, tick_cap=cap)


if __name__ == "__main__":
    unittest.main()
