import unittest
from unittest import mock

import torch as th

from carl import _carl
from carl.events import CarlEvents
from carl.gymnasium import CARLTorchVectorEnv, StepContext


@unittest.skipUnless(th.cuda.is_available(), "requires a CUDA GPU")
class EventViewTests(unittest.TestCase):

    def setUp(self):
        self.env = _carl.Env(
            2, 1, 0, 7, frameskip=8,
            capture_events=True, capture_ball_touches=True,
        )
        self.actions = th.zeros((2, 1, 7), device="cuda")
        self.actions[..., -1] = 1

    def test_views_share_native_storage(self):
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

    def test_snapshot_survives_step_reset_and_resize(self):
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

    def test_cross_stream_reads_complete_before_buffer_reuse(self):
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

    def test_touch_only_capture(self):
        env = _carl.Env(1, 1, 0, 7, capture_ball_touches=True)
        batch = CarlEvents.from_native(env.events)

        self.assertIsNone(batch.cars)
        self.assertIsNone(batch.ball)
        self.assertEqual(batch.ball_touches.surfaces.tolist(), [[0]])

    def test_reward_context_uses_environment_event_batch(self):
        received = []

        def reward(context):
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

    def test_ego_requires_one_car(self):
        env = _carl.Env(1, 1, 1, 7, capture_events=True)
        batch = CarlEvents.from_native(env.events)

        with self.assertRaises(ValueError):
            _ = batch.ego

    def test_context_observation_is_before_autoreset(self):
        received = []

        def reward(context):
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
