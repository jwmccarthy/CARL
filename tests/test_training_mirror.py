from copy import deepcopy
import math
import unittest

import torch as th

from carl.events import (
    ArenaSurface, BallTouch, Condition, ContactSurface, Goal, Jump, LeaveSurface,
    StateCondition, SurfaceContact,
)
from carl.training import (
    BallState, Box2D, Box3D, CarState, CompiledTrainingPack, Euler, Exercise,
    InitialState, InAir, OnSurface, ResetSampler, Stage, TrainingPack,
    TrainingPackResetProvider, Uniform,
)


DEVICES = ["cpu", "cuda"] if th.cuda.is_available() else ["cpu"]


class TrainingMirrorTests(unittest.TestCase):

    def exercise(self, placement: OnSurface | InAir | None = None) -> Exercise:
        return Exercise(
            name="mirror",
            initial=InitialState(
                ego=CarState(
                    placement=InAir((250, 500, 800), Euler(roll=0.3, pitch=-0.4, yaw=0.5))
                    if placement is None else placement,
                    velocity=(50, 100, 150), angular_velocity=(1, 2, 3), boost=70,
                ),
                ball=BallState((123, 456, 789), velocity=(10, 20, 30), angular_velocity=(4, 5, 6)),
            ),
            stages=[Stage(Jump(), within_ticks=120), Stage(Goal(), within_ticks=240)],
            target_goal="negative_y",
        )

    def test_reflection_of_physical_states_for_every_placement(self) -> None:
        placements = [None]
        for surface in (ArenaSurface.FLOOR, ArenaSurface.CEILING, ArenaSurface.SIDE_WALL, ArenaSurface.BACK_WALL):
            for side in (-1, 1):
                placements.append(OnSurface(surface, position=(250, 700), heading=0.65, side=side))
        for device in DEVICES:
            selected = th.ones(1, dtype=th.bool, device=device)
            polar = th.tensor([-1, 1, 1], device=device)
            axial = th.tensor([1, -1, -1], device=device)
            for placement in placements:
                with self.subTest(device=device, placement=placement):
                    exercise = self.exercise(placement)
                    before = deepcopy(exercise)
                    mirrored = exercise.mirrored()
                    original_state = ResetSampler(1, device=device).select(selected).sample(exercise.initial)
                    mirrored_state = ResetSampler(1, device=device).select(selected).sample(mirrored.initial)
                    for name in ("position", "velocity"):
                        th.testing.assert_close(getattr(mirrored_state.ball, name), getattr(original_state.ball, name) * polar)
                    for name in ("position", "velocity", "forward", "up"):
                        th.testing.assert_close(getattr(mirrored_state.cars, name), getattr(original_state.cars, name) * polar)
                    th.testing.assert_close(mirrored_state.ball.angular_velocity, original_state.ball.angular_velocity * axial)
                    th.testing.assert_close(mirrored_state.cars.angular_velocity, original_state.cars.angular_velocity * axial)
                    self.assertEqual(mirrored.target_goal, exercise.target_goal)
                    self.assertEqual(exercise, before)

    def test_distributions_preserve_mirrored_intervals(self) -> None:
        exercise = self.exercise(InAir(
            Box3D((-500, -100, 500), (100, 50, 1000)),
            Euler(roll=Uniform(-0.4, 0.2), pitch=Uniform(-0.2, 0.3), yaw=Uniform(0.5, 1.0)),
        ))
        exercise.initial.ball.position = (Uniform(-200, 100), 500, Uniform(700, 900))
        exercise.initial.ball.angular_velocity = (Uniform(1, 2), Uniform(2, 4), Uniform(-3, 5))
        mirrored = exercise.mirrored()
        self.assertEqual(mirrored.initial.ego.placement.position, Box3D((-100, -100, 500), (500, 50, 1000)))
        self.assertEqual(mirrored.initial.ego.placement.orientation.roll, Uniform(-0.2, 0.4))
        self.assertEqual(mirrored.initial.ego.placement.orientation.pitch, Uniform(-0.2, 0.3))
        self.assertEqual(mirrored.initial.ego.placement.orientation.yaw, Uniform(math.pi - 1, math.pi - 0.5))
        self.assertEqual(mirrored.initial.ball.position[0], Uniform(-100, 200))
        self.assertEqual(mirrored.initial.ball.angular_velocity, (Uniform(1, 2), Uniform(-4, -2), Uniform(-5, 3)))

        exercise.initial.ego.placement = OnSurface(position=Box2D((-500, -300), (-200, 100)))
        self.assertEqual(exercise.mirrored().initial.ego.placement.position, Box2D((200, -300), (500, 100)))

    def test_conditions_are_mirrored_in_every_group(self) -> None:
        exercise = self.exercise()
        exercise.fail_on = [StateCondition("ball_x", maximum=-500)]
        exercise.stages = [Stage(
            StateCondition("car_x", minimum=-300, maximum=100), within_ticks=120,
            repeat_on=BallTouch(ContactSurface.LEFT, min_speed_gain=50), max_repeats=3, max_total_matches=4,
            fail_on=[BallTouch(ContactSurface.RIGHT, max_speed_gain=100),
                     StateCondition("car_y", minimum=100), LeaveSurface(ArenaSurface.SIDE_WALL)],
        )]
        mirrored = exercise.mirrored()
        stage = mirrored.stages[0]
        self.assertEqual(stage.event, StateCondition("car_x", minimum=-100, maximum=300))
        self.assertEqual(stage.repeat_on, BallTouch(ContactSurface.RIGHT, min_speed_gain=50))
        self.assertEqual(stage.fail_on[0], BallTouch(ContactSurface.LEFT, max_speed_gain=100))
        self.assertEqual(stage.fail_on[1:], exercise.stages[0].fail_on[1:])
        self.assertEqual(mirrored.fail_on, [StateCondition("ball_x", minimum=500)])
        self.assertEqual((stage.within_ticks, stage.max_repeats, stage.max_total_matches), (120, 3, 4))
        compiled = CompiledTrainingPack.compile(TrainingPack("pair", [exercise, mirrored]), device="cpu")
        self.assertEqual(compiled.state_min[:, 0].tolist(), [-300, -100])
        self.assertEqual(compiled.repeat_car_parts[:, 0].tolist(), [1 << ContactSurface.LEFT, 1 << ContactSurface.RIGHT])

    def test_mirror_twice_restores_the_definition(self) -> None:
        exercise = self.exercise()
        exercise.stages = [Stage(StateCondition("car_x", minimum=0), fail_on=[SurfaceContact()])]
        restored = exercise.mirrored().mirrored(name=exercise.name)
        self.assertEqual(restored.stages, exercise.stages)
        self.assertEqual(restored.initial.ball, exercise.initial.ball)
        self.assertEqual(restored.initial.ego.velocity, exercise.initial.ego.velocity)
        self.assertEqual(restored.initial.ego.angular_velocity, exercise.initial.ego.angular_velocity)
        self.assertAlmostEqual(restored.initial.ego.placement.orientation.yaw, exercise.initial.ego.placement.orientation.yaw)

    def test_touch_part_sets_mirror_left_right_without_changing_roof_or_nose(self) -> None:
        exercise = self.exercise()
        exercise.stages[0].event = BallTouch((ContactSurface.LEFT, ContactSurface.ROOF, ContactSurface.FRONT))
        mirrored = exercise.mirrored()
        self.assertEqual(mirrored.stages[0].event.surface,
                         (ContactSurface.RIGHT, ContactSurface.ROOF, ContactSurface.FRONT))
        self.assertEqual(mirrored.mirrored().stages, exercise.stages)

    def test_pack_pairs_are_independent_of_each_other_and_the_source(self) -> None:
        exercise = self.exercise()
        pack = TrainingPack("pack", [exercise])
        augmented = pack.with_mirrors()
        self.assertEqual([e.name for e in augmented.exercises], ["mirror", "mirror (mirrored)"])
        augmented.exercises[0].stages[0].fail_on.append(SurfaceContact())
        augmented.exercises[1].initial.ego.boost = 0
        self.assertEqual(exercise.stages[0].fail_on, [])
        self.assertEqual(augmented.exercises[1].stages[0].fail_on, [])
        self.assertEqual(exercise.initial.ego.boost, 70)
        self.assertEqual(len(pack.exercises), 1)

    def test_balanced_selection_visits_every_variant_with_partial_resets(self) -> None:
        pack = TrainingPack("pack", [self.exercise(), self.exercise()]).with_mirrors()
        for device in DEVICES:
            provider = TrainingPackResetProvider(pack, 2, device=device, balanced=True)
            all_selected = th.ones(2, dtype=th.bool, device=device)
            second_selected = th.tensor([False, True], device=device)
            provider(all_selected)
            self.assertEqual(provider.exercise_index.tolist(), [0, 1])
            provider(second_selected)
            self.assertEqual(provider.exercise_index.tolist(), [0, 2])
            provider(second_selected)
            self.assertEqual(provider.exercise_index.tolist(), [0, 3])
            provider(second_selected)
            self.assertEqual(provider.exercise_index.tolist(), [0, 0])
            self.assertEqual(provider.sampler.reset_counts.tolist(), [1, 4])

    @unittest.skipUnless(th.cuda.is_available(), "requires a CUDA GPU")
    def test_balanced_reset_does_not_synchronize_with_host(self) -> None:
        provider = TrainingPackResetProvider(TrainingPack("pack", [self.exercise()]).with_mirrors(),
                                            2, balanced=True)
        selected = th.ones(2, dtype=th.bool, device="cuda")
        previous = th.cuda.get_sync_debug_mode()

        try:
            th.cuda.set_sync_debug_mode("error")
            provider(selected)
        finally:
            th.cuda.set_sync_debug_mode(previous)

    def test_unknown_directional_conditions_are_not_silently_copied(self) -> None:
        exercise = self.exercise()
        exercise.stages = [Stage(Condition())]
        with self.assertRaisesRegex(ValueError, "mirrored condition"):
            exercise.mirrored()


if __name__ == "__main__":
    unittest.main()
