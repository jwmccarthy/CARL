import math
import unittest

import torch as th

from carl import _carl
from carl.events import ArenaSurface, Jump
from carl.gymnasium import CARLTorchVectorEnv
from carl.training import (
    BallState, Box2D, Box3D, CarState, Euler, Exercise, ExerciseResetProvider,
    InitialState, InAir, OnSurface, ResetSampler, Stage, TrainingPack,
    TrainingPackResetProvider, Uniform,
)


@unittest.skipUnless(th.cuda.is_available(), "requires a CUDA GPU")
class ResetSamplingTests(unittest.TestCase):

    def indices(self, values):
        return th.tensor(values, device="cuda", dtype=th.int64)

    def exercise(self):
        return Exercise(
            name = "random aerial",
            initial = InitialState(
                ego = CarState(
                    placement = InAir(position=(Uniform(-200, 200), 0, 800)),
                    boost     = Uniform(50, 100),
                ),
                ball = BallState(position=(0, 500, Uniform(500, 700))),
            ),
            stages = [Stage(Jump())],
        )

    def test_exercise_provider_partial_and_seeded_resets(self):
        exercise = self.exercise()
        first = ExerciseResetProvider(exercise, 4, seed=17)
        second = ExerciseResetProvider(exercise, 4, seed=17)
        mask = th.tensor([False, True, False, True], device="cuda")

        state = first(mask)
        repeated = second(mask)

        self.assertEqual(state.simulation_indices.tolist(), [-1, 1, -1, 3])
        self.assertEqual(first.sampler.reset_counts.tolist(), [0, 1, 0, 1])
        th.testing.assert_close(state.cars, repeated.cars)
        th.testing.assert_close(state.ball, repeated.ball)
        self.assertTrue(state.ball.is_cuda)
        self.assertTrue(((state.cars.boost >= 50) & (state.cars.boost < 100)).all().item())
        self.assertFalse(th.equal(state.ball, first(mask).ball))

        empty = first(th.zeros(4, dtype=th.bool, device="cuda"))
        self.assertEqual(empty.simulation_indices.tolist(), [-1, -1, -1, -1])
        self.assertEqual(empty.cars.shape[0], 4)

        with self.assertRaisesRegex(ValueError, "boolean mask"):
            first(th.zeros(4, dtype=th.int64, device="cuda"))

    def test_training_pack_provider_selects_per_simulation(self):
        exercises = [self.exercise(), self.exercise()]
        exercises[0].initial.ball.position = (0, 0, 300)
        exercises[1].initial.ball.position = (0, 0, 700)
        pack = TrainingPack(name="two starts", exercises=exercises)
        first = TrainingPackResetProvider(pack, 64, seed=17)
        second = TrainingPackResetProvider(pack, 64, seed=17)
        selected = th.ones(64, dtype=th.bool, device="cuda")

        state = first(selected)
        repeated = second(selected)

        th.testing.assert_close(state.ball, repeated.ball)
        th.testing.assert_close(first.exercise_index, second.exercise_index)
        self.assertEqual(first.exercise_index.unique().numel(), 2)
        expected_height = th.where(first.exercise_index == 0, 300, 700)
        th.testing.assert_close(state.ball.position[:, 2], expected_height.float())

        previous = first.exercise_index.clone()
        selected[::2] = False
        state = first(selected)

        self.assertTrue(th.equal(first.exercise_index[::2], previous[::2]))
        self.assertTrue((state.simulation_indices[::2] == -1).all().item())
        self.assertEqual(first.sampler.reset_counts.tolist(), [1, 2] * 32)

    def test_training_pack_provider_avoids_torch_synchronization(self):
        pack = TrainingPack(name="two starts", exercises=[self.exercise(), self.exercise()])
        provider = TrainingPackResetProvider(pack, 8, seed=1)
        mask = th.ones(8, dtype=th.bool, device="cuda")
        previous = th.cuda.get_sync_debug_mode()

        try:
            th.cuda.set_sync_debug_mode("error")
            provider(mask)
        finally:
            th.cuda.set_sync_debug_mode(previous)

    def test_exercise_provider_environment_reset(self):
        provider = ExerciseResetProvider(self.exercise(), 2, seed=5)
        env = CARLTorchVectorEnv(n_sim=2, n_blue=1, n_orange=0, reset_state_provider=provider)
        self.addCleanup(env.close)

        observation = env.reset()

        self.assertEqual(provider.sampler.reset_counts.tolist(), [1, 1])
        self.assertTrue(th.isfinite(observation).all().item())
        self.assertTrue((observation.car_position[:, 0, 2] == 800).all().item())
        self.assertTrue(((observation.ball_position[:, 2] >= 500) &
                         (observation.ball_position[:, 2] < 700)).all().item())

        next_observation, _, _, _, _ = env.step(th.zeros((2, 7), device="cuda"))
        self.assertTrue(th.isfinite(next_observation).all().item())

        env.set_ball(
            position           = th.tensor([[0, 5300, 200]], device="cuda", dtype=th.float32),
            velocity           = th.zeros((1, 3), device="cuda"),
            angular_velocity   = th.zeros((1, 3), device="cuda"),
            simulation_indices = self.indices([0]),
        )
        reset_observation, _, terminated, _, _ = env.step(th.zeros((2, 7), device="cuda"))

        self.assertEqual(terminated.tolist(), [True, False])
        self.assertEqual(provider.sampler.reset_counts.tolist(), [2, 1])
        self.assertEqual(reset_observation.car_position[0, 0, 2].item(), 800)

    def test_exercise_reset_path_does_not_synchronize_torch(self):
        provider = ExerciseResetProvider(self.exercise(), 2, seed=5)
        env = CARLTorchVectorEnv(n_sim=2, n_blue=1, n_orange=0, reset_state_provider=provider)
        self.addCleanup(env.close)
        mask = th.tensor([True, False], device="cuda")
        previous = th.cuda.get_sync_debug_mode()

        try:
            th.cuda.set_sync_debug_mode("error")
            env._apply_reset_state(mask)
        finally:
            th.cuda.set_sync_debug_mode(previous)

    def test_native_setters_do_not_wait_for_input_stream(self):
        env = _carl.Env(1, 1, 0, 7)
        producer = th.cuda.Stream()
        ready = th.cuda.Event()
        position = th.empty((1, 3), device="cuda")
        velocity = th.zeros((1, 3), device="cuda")
        car_position = th.zeros((1, 1, 3), device="cuda")
        rotation = th.zeros((1, 1, 4), device="cuda")
        car_velocity = th.zeros((1, 1, 3), device="cuda")
        demoed = th.zeros((1, 1), device="cuda", dtype=th.bool)
        indices = th.zeros(1, device="cuda", dtype=th.int64)
        scores = th.zeros(1, device="cuda", dtype=th.int32)
        producer.wait_stream(th.cuda.current_stream())

        with th.cuda.stream(producer):
            th.cuda._sleep(200_000_000)
            position[:, 0].fill_(100)
            position[:, 1].fill_(200)
            position[:, 2].fill_(500)
            car_position[:, :, 2].fill_(800)
            rotation[:, :, 3].fill_(1)
            ready.record()
            env.set_ball(position, velocity, velocity, simulation_indices=indices)
            env.set_car(car_position, rotation, car_velocity, car_velocity, demoed,
                        simulation_indices=indices)
            env.set_match_state(scores, scores, scores, simulation_indices=indices)
            env.reset_boost_pads(simulation_indices=indices)

        self.assertFalse(ready.query())
        del position, velocity, car_position, rotation, car_velocity, demoed, indices, scores
        env.events.acquire(th.cuda.current_stream().cuda_stream)
        state = th.from_dlpack(env.get_state()).clone()

        th.testing.assert_close(state[0, :3], th.tensor([100, 200, 500], device="cuda", dtype=th.float32))

    def test_seeded_stream_is_reproducible(self):
        first = ResetSampler(3, seed=123)
        second = ResetSampler(3, seed=123)
        different = ResetSampler(3, seed=124)

        def draw(sampler):
            return sampler.select(self.indices([2, 0])).uniform(shape=(4,))

        values = draw(first)
        repeated = draw(second)
        other = draw(different)

        th.testing.assert_close(values, repeated, rtol=0, atol=0)
        self.assertFalse(th.equal(values, other))
        self.assertFalse(th.equal(repeated, draw(second)))
        self.assertTrue(values.is_cuda)
        self.assertTrue(((values >= 0) & (values < 1)).all().item())

    def test_uniform_choice_and_partial_counters(self):
        sampler = ResetSampler(4, seed=17)
        mask = th.tensor([False, True, False, True], device="cuda")
        batch = sampler.select(mask)
        values = batch.uniform(-5, 8, shape=(3,))
        choices = batch.choice((-1, 1))

        self.assertTrue(values.is_cuda)
        self.assertEqual(values.shape, (2, 3))
        self.assertTrue(((values >= -5) & (values < 8)).all().item())
        self.assertTrue(((choices == -1) | (choices == 1)).all().item())
        self.assertEqual(sampler.reset_counts.tolist(), [0, 1, 0, 1])
        self.assertEqual(batch.simulation_indices.tolist(), [1, 3])

    def test_floor_placement_and_correlated_ball(self):
        batch = ResetSampler(3).select(self.indices([0, 2]))
        batch.ego.on_floor(
            position=Box2D((-200, -1500), (200, -1000)),
            heading=math.pi / 2,
            velocity=(0, Uniform(500, 700), 0),
            boost=Uniform(60, 80),
        )
        ball_position = batch.ego.local_point(forward=300, up=200)
        batch.ball.set(position=ball_position, velocity=batch.ego.local_vector(forward=400))
        state = batch.build()

        self.assertTrue(state.ball.is_cuda)
        self.assertEqual(state.cars.shape, (2, 1, _carl.OBS_PER_CAR))
        self.assertTrue((state.cars.position[..., 2] == _carl.CAR_REST_Z).all().item())
        self.assertTrue(state.cars.on_ground.all().item())
        th.testing.assert_close(state.cars.ego.forward, th.tensor([[0, 1, 0]] * 2, device="cuda", dtype=th.float32))
        th.testing.assert_close(state.ball.position, ball_position)
        th.testing.assert_close(
            state.ball.velocity,
            th.tensor([[0, 400, 0]] * 2, device="cuda", dtype=th.float32),
            rtol=0, atol=1e-4,
        )
        self.assertTrue(((state.cars.boost >= 60) & (state.cars.boost < 80)).all().item())

    def test_sample_initial_state_in_air(self):
        initial = InitialState(
            ego = CarState(
                placement = InAir(
                    position    = Box3D((-200, -1000, 500), (200, -500, 900)),
                    orientation = Euler(roll=Uniform(-0.5, 0.5), pitch=0.4, yaw=1.2),
                ),
                velocity         = (Uniform(100, 200), 300, -100),
                angular_velocity = (0.5, -0.2, Uniform(-1, 1)),
                boost            = 75,
            ),
            ball = BallState(
                position         = (1000, 1000, Uniform(300, 500)),
                velocity         = (100, -200, 300),
                angular_velocity = (1, 2, 3),
            ),
        )
        state = ResetSampler(4).select(self.indices([1, 3])).sample(initial)

        self.assertFalse(state.cars.on_ground.any().item())
        self.assertTrue((state.cars.position[..., 2] >= 500).all().item())
        th.testing.assert_close(state.cars.forward.norm(dim=-1), th.ones((2, 1), device="cuda"))
        th.testing.assert_close(state.cars.up.norm(dim=-1), th.ones((2, 1), device="cuda"))
        th.testing.assert_close((state.cars.up * state.cars.forward).sum(dim=-1), th.zeros((2, 1), device="cuda"), atol=1e-6, rtol=0)
        self.assertTrue((state.cars.boost == 75).all().item())
        th.testing.assert_close(state.ball.angular_velocity, th.tensor([[1, 2, 3]] * 2, device="cuda", dtype=th.float32))
        self.assertFalse(state.car_internal_state.any().item())

    def test_empty_selection(self):
        batch = ResetSampler(4).select(self.indices([]))
        state = batch.build()

        self.assertEqual(batch.uniform(shape=(3,)).shape, (0, 3))
        self.assertEqual(state.ball.shape, (0, _carl.OBS_BALL))
        self.assertEqual(state.cars.shape, (0, 1, _carl.OBS_PER_CAR))

    def test_ceiling_and_side_wall_placement(self):
        batch = ResetSampler(2).select(self.indices([0, 1]))
        ceiling_z = 2048 - _carl.CAR_REST_Z

        batch.ego.on_ceiling(position=(100, -200), heading=math.pi / 2)
        state = batch.build()
        th.testing.assert_close(
            state.cars.position,
            th.tensor([[[100, -200, ceiling_z]]] * 2, device="cuda"),
        )
        th.testing.assert_close(
            state.cars.up,
            th.tensor([[[0, 0, -1]]] * 2, device="cuda", dtype=th.float32),
        )
        th.testing.assert_close(
            state.cars.forward,
            th.tensor([[[0, -1, 0]]] * 2, device="cuda", dtype=th.float32),
            rtol=0, atol=1e-6,
        )
        self.assertFalse(state.cars.on_ground.any().item())

        wall_x = 4096 - _carl.CAR_REST_Z
        batch.ego.on_side_wall(position=(300, 500), side=1)
        state = batch.build()
        th.testing.assert_close(
            state.cars.position,
            th.tensor([[[wall_x, 300, 500]]] * 2, device="cuda"),
        )
        th.testing.assert_close(
            state.cars.up,
            th.tensor([[[-1, 0, 0]]] * 2, device="cuda", dtype=th.float32),
        )
        th.testing.assert_close(
            state.cars.forward,
            th.tensor([[[0, 1, 0]]] * 2, device="cuda", dtype=th.float32),
        )
        self.assertFalse(state.cars.on_ground.any().item())

        batch.ego.on_side_wall(position=(300, 500), side=-1)
        state = batch.build()
        th.testing.assert_close(
            state.cars.position,
            th.tensor([[[-wall_x, 300, 500]]] * 2, device="cuda"),
        )
        th.testing.assert_close(
            state.cars.up,
            th.tensor([[[1, 0, 0]]] * 2, device="cuda", dtype=th.float32),
        )

    def test_back_wall_and_ball_surface_placement(self):
        batch = ResetSampler(2).select(self.indices([0, 1]))
        wall_y = 5120 - _carl.CAR_REST_Z

        batch.ego.on_back_wall(position=(400, 700), side=1)
        state = batch.build()
        th.testing.assert_close(
            state.cars.position,
            th.tensor([[[400, wall_y, 700]]] * 2, device="cuda"),
        )
        th.testing.assert_close(
            state.cars.up,
            th.tensor([[[0, -1, 0]]] * 2, device="cuda", dtype=th.float32),
        )
        th.testing.assert_close(
            state.cars.forward,
            th.tensor([[[1, 0, 0]]] * 2, device="cuda", dtype=th.float32),
        )

        batch.ego.on_back_wall(position=(400, 700), side=-1)
        state = batch.build()
        th.testing.assert_close(
            state.cars.position,
            th.tensor([[[400, -wall_y, 700]]] * 2, device="cuda"),
        )
        th.testing.assert_close(
            state.cars.up,
            th.tensor([[[0, 1, 0]]] * 2, device="cuda", dtype=th.float32),
        )

        ceiling_z = 2048 - _carl.BALL_REST_Z
        batch.ball.on_ceiling(position=(200, 300))
        th.testing.assert_close(
            batch.build().ball.position,
            th.tensor([[200, 300, ceiling_z]] * 2, device="cuda"),
        )

        ball_x = 4096 - _carl.BALL_REST_Z
        batch.ball.on_side_wall(position=(400, 600), side=-1)
        th.testing.assert_close(
            batch.build().ball.position,
            th.tensor([[-ball_x, 400, 600]] * 2, device="cuda"),
        )

        ball_y = 5120 - _carl.BALL_REST_Z
        batch.ball.on_back_wall(position=(400, 600), side=1)
        th.testing.assert_close(
            batch.build().ball.position,
            th.tensor([[400, ball_y, 600]] * 2, device="cuda"),
        )

    def test_surface_sampling_and_side_validation(self):
        initial = InitialState(
            ego = CarState(
                placement = OnSurface(
                    surface=ArenaSurface.SIDE_WALL, position=(600, 800),
                    heading=0.3, side=-1,
                ),
                velocity  = (0, 100, 0),
            ),
            ball = BallState(position=(0, 0, 300)),
        )
        state = ResetSampler(2).select(self.indices([0, 1])).sample(initial)
        wall_x = -(4096 - _carl.CAR_REST_Z)

        self.assertTrue((state.cars.position[..., 0] == wall_x).all().item())
        th.testing.assert_close(
            state.cars.position[:, 0, 1:],
            th.tensor([[600, 800]] * 2, device="cuda", dtype=th.float32),
        )
        self.assertFalse(state.cars.on_ground.any().item())
        self.assertFalse(state.car_internal_state.any().item())

        with self.assertRaises(ValueError):
            OnSurface(side=0)

        batch = ResetSampler(1).select(self.indices([0]))
        with self.assertRaises(ValueError):
            batch.ego.on_side_wall(side=2)
        with self.assertRaises(ValueError):
            batch.ball.on_back_wall(side=0)
        with self.assertRaises(ValueError):
            batch.ego.on_surface(ArenaSurface.FLOOR | ArenaSurface.CEILING)

    def test_surface_reset_steps_without_divergence(self):
        sampler = ResetSampler(2, seed=5)
        initial = InitialState(
            ego  = CarState(OnSurface(surface=ArenaSurface.CEILING)),
            ball = BallState(position=(0, 0, 500)),
        )

        def provider(mask):
            return sampler.select(mask).sample(initial)

        env = CARLTorchVectorEnv(
            n_sim=2, n_blue=1, n_orange=0, frameskip=1,
            reset_state_provider=provider,
        )
        self.addCleanup(env.close)
        observation = env.reset()
        self.assertTrue(th.isfinite(observation).all().item())

        actions = th.zeros((2, 7), device="cuda")
        observation, *_ = env.step(actions)
        self.assertTrue(th.isfinite(observation).all().item())
        th.testing.assert_close(
            observation.car_position[..., 2],
            th.full((2, 1), 2048 - _carl.CAR_REST_Z, device="cuda"),
            rtol=0, atol=100,
        )

    def test_wall_and_ceiling_car_starts_do_not_launch(self):
        cases = [
            (ArenaSurface.SIDE_WALL, 1, (0, 700), 0),
            (ArenaSurface.SIDE_WALL, -1, (0, 700), 0),
            (ArenaSurface.BACK_WALL, 1, (2000, 700), 1),
            (ArenaSurface.BACK_WALL, -1, (2000, 700), 1),
            (ArenaSurface.CEILING, 1, (0, 0), 2),
        ]

        for surface, side, position, axis in cases:
            with self.subTest(surface=surface, side=side):
                initial = InitialState(
                    ego  = CarState(OnSurface(surface=surface, position=position, side=side)),
                    ball = BallState(position=(0, 0, 500)),
                )
                sampler = ResetSampler(1, seed=5)

                def provider(mask):
                    return sampler.select(mask).sample(initial)

                env = CARLTorchVectorEnv(
                    n_sim=1, n_blue=1, n_orange=0, frameskip=1, reset_state_provider=provider,
                )

                try:
                    env.reset()
                    observation, _, _, _, _ = env.step(th.zeros((1, 7), device="cuda"))

                    self.assertTrue(observation.cars.ego.on_ground[0].item())
                    self.assertLess(abs(observation.cars.ego.velocity[0, axis].item()), 15)
                finally:
                    env.close()

    def test_provider_round_trip_and_neutral_partial_reset(self):
        sampler = ResetSampler(2, seed=19)
        initial = InitialState(
            ego = CarState(
                placement        = InAir((300, -1000, 800), Euler(yaw=0.4)),
                velocity         = (100, 200, 300),
                angular_velocity = (0.1, 0.2, 0.3),
                boost            = 70,
            ),
            ball = BallState(position=(1000, 1000, 500), velocity=(300, 0, 0)),
        )

        def provider(mask):
            return sampler.select(mask).sample(initial)

        env = CARLTorchVectorEnv(
            n_sim=2, n_blue=1, n_orange=0, frameskip=1,
            reset_state_provider=provider,
        )
        self.addCleanup(env.close)
        observation = env.reset()

        expected_position = th.tensor([[[300, -1000, 800]]] * 2, device="cuda", dtype=th.float32)
        th.testing.assert_close(observation.car_position, expected_position)
        th.testing.assert_close(observation.car_velocity, th.tensor([[[100, 200, 300]]] * 2, device="cuda", dtype=th.float32))
        self.assertTrue((observation.cars.boost == 70).all().item())

        actions = th.zeros((2, 7), device="cuda")
        actions[:, -1] = 1
        observation, _, _, _, _ = env.step(actions)
        self.assertTrue(observation.cars.has_double_jumped.all().item())
        before = observation.clone()

        mask = th.tensor([False, True], device="cuda")
        env._apply_reset_state(mask)
        observation = env._observe()

        self.assertEqual(observation.cars.has_double_jumped[:, 0].tolist(), [True, False])
        self.assertEqual(observation.ego_has_flip_or_jump.tolist(), [False, True])
        th.testing.assert_close(observation[0], before[0])
        th.testing.assert_close(observation.car_position[1], expected_position[1])
