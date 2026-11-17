import unittest
from unittest import mock

import torch as th

from carl import _carl
from carl.events import CarlEvents
from carl.gymnasium import CARLTorchVectorEnv, StepContext


@unittest.skipUnless(th.cuda.is_available(), "requires a CUDA GPU")
class EventViewTests(unittest.TestCase):

    def setUp(self) -> None:
        self.env = _carl.Env(
            2, 1, 0, 7, frameskip=8,
            capture_events=True, capture_ball_touches=True,
        )
        self.actions = th.zeros((2, 1, 7), device="cuda")
        self.actions[..., -1] = 1

    def test_views_share_native_storage(self) -> None:
        self.env.step_continuous(self.actions)

        with mock.patch("torch.cuda.synchronize", side_effect=AssertionError("blocking sync")):
            batch = CarlEvents.from_native(self.env.events)

        flags = th.from_dlpack(self.env.events.get_car_flags())
        touches = th.from_dlpack(self.env.events.get_ball_touch_surfaces())

        self.assertEqual(batch.cars.flags.data_ptr(), flags.data_ptr())
        self.assertEqual(batch.ball_touches.surfaces.data_ptr(), touches.data_ptr())
        self.assertTrue(batch.cars.jumped.all().item())
        self.assertEqual(batch.ego.jumped.shape, (2,))
        self.assertEqual(batch.ego.flags.data_ptr(), batch.cars.flags.data_ptr())

    def test_snapshot_survives_step_reset_and_resize(self) -> None:
        self.env.step_continuous(self.actions)
        views = CarlEvents.from_native(self.env.events)
        snapshot = views.snapshot()

        self.assertNotEqual(snapshot.cars.flags.data_ptr(), views.cars.flags.data_ptr())
        self.assertNotEqual(snapshot.ball_touches.surfaces.data_ptr(), views.ball_touches.surfaces.data_ptr())

        self.env.step_continuous(self.actions)
        self.env.reset()
        self.env.frameskip = 4

        self.assertEqual(snapshot.cars.flags.shape, (2, 1))
        self.assertTrue(snapshot.cars.jumped.all().item())

    def test_cross_stream_reads_complete_before_buffer_reuse(self) -> None:
        consumer = th.cuda.Stream()
        consumer.wait_stream(th.cuda.current_stream())
        finished = th.cuda.Event()

        self.env.step_continuous(self.actions)

        with th.cuda.stream(consumer):
            views = CarlEvents.from_native(self.env.events)
            th.cuda._sleep(200_000_000)
            snapshot = views.snapshot()
            finished.record()

        self.env.step_continuous(self.actions)
        self.assertFalse(finished.query())

        latest = CarlEvents.from_native(self.env.events)
        th.cuda.current_stream().wait_event(finished)

        self.assertTrue(snapshot.cars.jumped.all().item())
        self.assertFalse(latest.cars.jumped.any().item())

    def test_touch_only_capture(self) -> None:
        env = _carl.Env(1, 1, 0, 7, capture_ball_touches=True)
        batch = CarlEvents.from_native(env.events)

        self.assertIsNone(batch.cars)
        self.assertIsNone(batch.ball)
        self.assertEqual(batch.ball_touches.surfaces.tolist(), [[0]])

    def test_capture_preserves_observations_actions_and_transitions(self) -> None:
        for team in (1, 2, 3):
            for previous_controls in (False, True):
                with self.subTest(team=team, previous_controls=previous_controls):
                    self.check_capture_equivalence(team, previous_controls)

    def check_capture_equivalence(self, team, previous_controls):
        envs = [CARLTorchVectorEnv(
            4, team, team, seed=17, frameskip=4, discrete_actions=True,
            previous_controls=previous_controls, copy_outputs=True,
            capture_events=events, capture_ball_touches=touches,
            no_touch_timeout_ticks=24,
        ) for events, touches in ((False, False), (True, False), (False, True), (True, True))]
        for env in envs:
            self.addCleanup(env.close)

        observations = [env.reset() for env in envs]
        generator = th.Generator(device="cuda").manual_seed(29)
        for step in range(16):
            for env, observation in zip(envs[1:], observations[1:]):
                th.testing.assert_close(observation, observations[0], rtol=0, atol=0)
                th.testing.assert_close(env.action_mask(observation), envs[0].action_mask(observations[0]))

            actions = th.stack([th.randint(size, (envs[0].n_envs,), device="cuda", generator=generator)
                                for size in _carl.ACTION_NVECS], dim=-1)
            if step == 3:
                for env in envs:
                    position = th.tensor([[0., 5300., 200.]], device="cuda")
                    env.set_ball(position, th.zeros_like(position), th.zeros_like(position),
                                 simulation_indices=th.tensor([0], device="cuda"))

            outputs = [env.step(actions) for env in envs]
            observations = [output[0] for output in outputs]
            for output in outputs[1:]:
                for actual, expected in zip(output[:4], outputs[0][:4]):
                    th.testing.assert_close(actual, expected, rtol=0, atol=0)
                if "final_obs" in output[4]:
                    th.testing.assert_close(output[4]["final_obs"], outputs[0][4]["final_obs"], rtol=0, atol=0)

    def test_reward_context_uses_environment_event_batch(self) -> None:
        received = []

        def reward(context: StepContext) -> th.Tensor:
            self.assertIsInstance(context, StepContext)
            received.append(context.events)
            return context.events.cars.jumped.float()

        env = CARLTorchVectorEnv(
            n_sim=2, n_blue=1, n_orange=0, frameskip=8,
            capture_events=True, capture_ball_touches=True,
            reward_funcs=[reward],
        )
        self.addCleanup(env.close)
        env.reset()
        actions = self.actions.reshape(2, 7)
        _, reward, _, _, _ = env.step(actions)

        self.assertIs(received[-1], env.events)
        self.assertIs(type(env.events), CarlEvents)
        self.assertIs(env.get_ball_touches(), env.events.ball_touches)
        self.assertEqual(reward.tolist(), [1, 1])

        snapshot = env.events.snapshot()
        env._env.frameskip = 4
        env.step(actions)

        self.assertIs(type(snapshot), CarlEvents)
        self.assertEqual(snapshot.score_delta.tolist(), [0, 0])
        self.assertTrue(snapshot.cars.jumped.all().item())

    def test_ego_requires_one_car(self) -> None:
        env = _carl.Env(1, 1, 1, 7, capture_events=True)
        batch = CarlEvents.from_native(env.events)

        with self.assertRaises(ValueError):
            _ = batch.ego

    def test_context_observation_is_before_autoreset(self) -> None:
        received = []

        def reward(context: StepContext) -> th.Tensor:
            received.append(context)
            return th.zeros((2, 1), device="cuda")

        env = CARLTorchVectorEnv(
            n_sim=2, n_blue=1, n_orange=0, frameskip=8,
            capture_events=True, reward_funcs=[reward],
        )
        self.addCleanup(env.close)
        env.reset()
        env.set_car(
            position=th.tensor([[[300, 500, 800]]] * 2, device="cuda", dtype=th.float32),
            rotation=th.tensor([[[0, 0, 0, 1]]] * 2, device="cuda", dtype=th.float32),
            velocity=th.zeros((2, 1, 3), device="cuda"),
            angular_velocity=th.zeros((2, 1, 3), device="cuda"),
            demoed=th.zeros((2, 1), device="cuda", dtype=th.bool),
        )
        env.set_ball(
            position=th.tensor([[0, 5300, 200], [1000, 0, 1000]], device="cuda", dtype=th.float32),
            velocity=th.zeros((2, 3), device="cuda"),
            angular_velocity=th.zeros((2, 3), device="cuda"),
        )
        observation, _, terminated, _, info = env.step(th.zeros((2, 7), device="cuda"))
        context = received[-1]

        self.assertEqual(terminated.tolist(), [True, False])
        self.assertEqual(context.events.ball.scored.tolist(), [True, False])
        th.testing.assert_close(context.current_observation.car_position, context.current.car_position)
        th.testing.assert_close(context.current_observation[1], observation[1])
        self.assertFalse(th.equal(context.current_observation[0], observation[0]))
        th.testing.assert_close(context.current_observation, info["final_obs"])
