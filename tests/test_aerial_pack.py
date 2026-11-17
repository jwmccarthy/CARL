import unittest

import torch as th

from carl.events import ArenaSurface, BallTouch, ContactSurface, Dodge, DoubleJump, Jump, LeaveSurface, StateCondition, SurfaceContact
from carl.training import CompiledTrainingPack, TrainingPackResetProvider
from examples.training_packs.pack import aerial_pack


class AerialPackTests(unittest.TestCase):

    def test_specific_sequences_and_original_pack_remain_available(self) -> None:
        pack = aerial_pack()
        exercises = {exercise.name: exercise for exercise in pack.exercises[::2]}
        self.assertEqual([len(exercise.stages) for exercise in exercises.values()], [4, 4, 4, 5, 4, 5])
        self.assertEqual(len(aerial_pack(1).exercises[0].stages), 3)
        CompiledTrainingPack.compile(pack, device="cpu")
        for exercise in exercises.values():
            self.assertTrue(all(stage.within_ticks is not None for stage in exercise.stages))
            touches = [stage.event for stage in exercise.stages if isinstance(stage.event, BallTouch)]
            self.assertTrue(touches)
            setup = (ContactSurface.FRONT, ContactSurface.ROOF)
            expected = ([setup, ContactSurface.FRONT, ContactSurface.FRONT]
                        if exercise.name == "Air dribble" else [setup] * len(touches))
            self.assertEqual([touch.surface for touch in touches], expected)
            self.assertTrue(all(touch.min_speed_gain is not None for touch in touches))
            for stage in exercise.stages:
                if isinstance(stage.event, BallTouch):
                    wrong = [condition for condition in stage.fail_on if isinstance(condition, BallTouch)]
                    self.assertEqual(len(wrong), 1)
                    self.assertEqual(set(wrong[0].surface), {ContactSurface.REAR, ContactSurface.LEFT,
                                                            ContactSurface.RIGHT, ContactSurface.UNDERSIDE})

        fast = exercises["Fast aerial"]
        self.assertIsInstance(fast.stages[0].event, Jump)
        self.assertIsInstance(fast.stages[1].event, DoubleJump)
        self.assertGreater(fast.stages[1].min_ticks, 0)
        self.assertTrue(fast.stages[2].from_exercise_start)
        self.assertIn(Dodge(), fast.fail_on)
        wall = exercises["Wall aerial"]
        self.assertEqual(wall.stages[0].event, LeaveSurface(ArenaSurface.SIDE_WALL))
        self.assertEqual(wall.stages[1].event, StateCondition("car_x", minimum=-3500))
        self.assertEqual(pack.exercises[7].stages[1].event, StateCondition("car_x", maximum=3500))
        ceiling = exercises["Ceiling aerial"]
        self.assertIn(Jump(), ceiling.stages[0].fail_on)
        self.assertEqual(ceiling.stages[1].event, StateCondition("car_height", maximum=1600))
        dribble = exercises["Air dribble"]
        for stage in dribble.stages[1:3]:
            self.assertGreaterEqual(stage.min_ticks, 24)
            self.assertLessEqual(stage.event.max_speed_gain, 250)
            self.assertIn(SurfaceContact(entity="car"), stage.fail_on)
        # A shot may finish after the shooter lands; a dribble may not.
        self.assertNotIn(SurfaceContact(entity="car"), exercises["Aerial shot"].stages[-1].fail_on)
        self.assertIn(SurfaceContact(entity="car"), dribble.stages[-1].fail_on)

    def test_ball_launches_are_modestly_higher_and_more_upward_than_v1(self) -> None:
        old, new = aerial_pack(1), aerial_pack(2)
        selected = th.ones(12, dtype=th.bool)
        before = TrainingPackResetProvider(old, 12, device="cpu", balanced=True)(selected)
        after = TrainingPackResetProvider(new, 12, device="cpu", balanced=True)(selected)
        # Identical sampler/noise layout isolates the actual distribution changes.
        lift = after.ball.position[:, 2] - before.ball.position[:, 2]
        upward = after.ball.velocity[:, 2] - before.ball.velocity[:, 2]
        self.assertTrue(((lift >= 99) & (lift <= 151)).all())
        self.assertTrue(((upward >= 149) & (upward <= 201)).all())
        self.assertTrue((after.ball.velocity[:, 2] > 0).all())

    def test_v3_raises_every_ball_start_another_250_without_other_changes(self) -> None:
        previous, current = aerial_pack(2), aerial_pack()
        selected = th.ones(12, dtype=th.bool)
        before = TrainingPackResetProvider(previous, 12, device="cpu", balanced=True)(selected)
        after = TrainingPackResetProvider(current, 12, device="cpu", balanced=True)(selected)
        th.testing.assert_close(after.ball.position, before.ball.position + th.tensor([0, 0, 250]))
        th.testing.assert_close(after.ball.velocity, before.ball.velocity)
        th.testing.assert_close(after.ball.angular_velocity, before.ball.angular_velocity)
        th.testing.assert_close(after.cars, before.cars)
        for old, new in zip(previous.exercises, current.exercises):
            self.assertEqual(new.stages, old.stages)
            self.assertEqual(new.fail_on, old.fail_on)
            self.assertEqual(new.target_goal, old.target_goal)
        self.assertEqual(current, aerial_pack(3))


if __name__ == "__main__":
    unittest.main()
