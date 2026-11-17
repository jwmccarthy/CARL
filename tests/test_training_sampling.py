import math
import unittest
from unittest import mock

import torch as th

from carl.events import ArenaSurface, Jump
from carl.training import (
    BallState, Box2D, Box3D, CarState, Euler, Exercise, InitialState, InAir,
    OnSurface, ResetBatch, ResetSampler, Stage, TrainingPack, TrainingPackResetProvider, Uniform,
)


class TrainingSamplingTests(unittest.TestCase):

    def test_batched_initial_states_match_all_single_exercise_placements(self) -> None:
        placements = [InAir((250, 500, 800), Euler(roll=0.3, pitch=-0.4, yaw=0.5))]
        for surface in (ArenaSurface.FLOOR, ArenaSurface.CEILING, ArenaSurface.SIDE_WALL, ArenaSurface.BACK_WALL):
            for side in (-1, 1):
                placements.append(OnSurface(surface, position=(250, 700), heading=0.65, side=side))
        exercises = [Exercise(str(index), InitialState(
            CarState(placement, velocity=(50, 100, 150), angular_velocity=(1, 2, 3), boost=70),
            BallState((123, 456, 789), velocity=(10, 20, 30), angular_velocity=(4, 5, 6)),
        ), [Stage(Jump())]) for index, placement in enumerate(placements)]
        pack = TrainingPack("placements", exercises).with_mirrors()
        provider = TrainingPackResetProvider(pack, len(pack.exercises), device="cpu", balanced=True)
        selected = th.ones(len(pack.exercises), dtype=th.bool)
        # Sampling must no longer loop over ResetBatch.sample for every exercise.
        with mock.patch.object(ResetBatch, "sample", side_effect=AssertionError("unbatched sampling")):
            actual = provider(selected)
        for index, exercise in enumerate(pack.exercises):
            expected = ResetSampler(1, device="cpu").select(th.ones(1, dtype=th.bool)).sample(exercise.initial)
            th.testing.assert_close(actual.ball[index:index + 1], expected.ball)
            th.testing.assert_close(actual.cars[index:index + 1], expected.cars)
            th.testing.assert_close(actual.car_internal_state[index:index + 1], expected.car_internal_state)

    def test_uniform_bounds_seeded_sampling_and_masked_resets(self) -> None:
        initial = InitialState(
            CarState(InAir(Box3D((-200, -100, 500), (100, 200, 800)),
                           Euler(roll=Uniform(-0.2, 0.3), yaw=math.pi / 2)),
                     velocity=(Uniform(100, 200), 300, 400),
                     angular_velocity=(1, Uniform(-2, 2), 3), boost=Uniform(50, 100)),
            BallState((Uniform(-300, 300), 500, Uniform(600, 900)),
                      velocity=(0, Uniform(200, 400), Uniform(250, 350)), angular_velocity=(1, 2, 3)),
        )
        wall = InitialState(CarState(OnSurface(ArenaSurface.SIDE_WALL, Box2D((-500, 400), (500, 700)),
                                               Uniform(0.2, 0.4), side=-1)),
                            BallState((100, 200, 300)))
        pack = TrainingPack("random", [Exercise("air", initial, [Stage(Jump())]),
                                      Exercise("wall", wall, [Stage(Jump())])])
        first = TrainingPackResetProvider(pack, 128, seed=17, device="cpu", balanced=True)
        second = TrainingPackResetProvider(pack, 128, seed=17, device="cpu", balanced=True)
        selected = th.ones(128, dtype=th.bool)
        for _ in range(3):
            before = first.exercise_index.clone()
            counts = first.sampler.reset_counts.clone()
            state, repeated = first(selected), second(selected)
            th.testing.assert_close(state.ball, repeated.ball)
            th.testing.assert_close(state.cars, repeated.cars)
            th.testing.assert_close(first.exercise_index, second.exercise_index)
            th.testing.assert_close(first.sampler.reset_counts, counts + selected.long())
            th.testing.assert_close(first.exercise_index[~selected], before[~selected])
            self.assertIsNone(state.simulation_indices)
            th.testing.assert_close(state.reset_mask, selected)
            air = first.exercise_index == 0
            self.assertTrue(((state.cars.ego.boost[air] >= 50) & (state.cars.ego.boost[air] < 100)).all())
            self.assertTrue(((state.ball.position[air, 2] >= 600) & (state.ball.position[air, 2] < 900)).all())
            self.assertTrue(((state.ball.velocity[air, 2] >= 250) & (state.ball.velocity[air, 2] < 350)).all())
            forward, up = state.cars.ego.forward, state.cars.ego.up
            th.testing.assert_close(forward.norm(dim=-1), th.ones(128))
            th.testing.assert_close(up.norm(dim=-1), th.ones(128))
            th.testing.assert_close((forward * up).sum(dim=-1), th.zeros(128), atol=1e-6, rtol=0)
            selected[::2] = False
        empty = first(th.zeros(128, dtype=th.bool))
        self.assertFalse(empty.reset_mask.any())


if __name__ == "__main__":
    unittest.main()
