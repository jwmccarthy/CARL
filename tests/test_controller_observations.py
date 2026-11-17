"""Controller schema compatibility and immediate replay-reset observations."""
import math
import unittest

import carl
import torch as th
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import CARLActionCodec
from carl.gymnasium.state import CARLObservation


class ControllerViewTests(unittest.TestCase):
    def test_base_and_extended_views_and_action_masks(self):
        for cars in (2, 4, 6):
            base_size = 85 + 27 * cars
            base = th.zeros((3, base_size))
            base[:, -2] = 1
            base[:, -1] = .5
            extended = th.cat((base, th.zeros((3, 27))), -1)
            # New trailing fields must not be mistaken for the dodge window.
            extended[:, -2:] = -1
            old = CARLObservation.from_tensor(base, cars)
            new = CARLObservation.from_tensor(extended, cars)
            th.testing.assert_close(new.ego_has_flip_or_jump, old.ego_has_flip_or_jump)
            th.testing.assert_close(new.ego_flip_window_remaining, old.ego_flip_window_remaining)
            th.testing.assert_close(CARLActionCodec(True).mask(extended), CARLActionCodec().mask(base))
            self.assertEqual(new.ego_controller.shape, (3, 27))
            new.ego_previous_controls.fill_(.25)
            th.testing.assert_close(extended[:, base_size:base_size + 8], th.full((3, 8), .25))
            with self.assertRaisesRegex(ValueError, "ego controller v1"):
                _ = old.ego_controller


@unittest.skipUnless(th.cuda.is_available(), "requires CUDA or ROCm")
class ControllerResetTests(unittest.TestCase):
    def make_env(self, simulations=6, team=1, **options):
        env = CARLTorchVectorEnv(simulations, team, team, frameskip=1,
                                ego_controller=True, discrete_actions=True, **options)
        self.addCleanup(env.close)
        env.reset()
        return env

    def restore(self, env, positions, rotations=None, internal=None, indices=None, demo=None):
        count, cars = positions.shape[:2]
        if rotations is None:
            rotations = th.zeros((count, cars, 4), device="cuda")
            rotations[..., 3] = 1
        if internal is None:
            internal = th.zeros((count, cars, carl.CAR_INTERNAL_SIZE), device="cuda")
        if demo is None:
            demo = th.zeros((count, cars), device="cuda", dtype=th.bool)
        env._env.set_car(positions, rotations, th.zeros_like(positions), th.zeros_like(positions),
                         demo, boost=th.full((count, cars), 50., device="cuda"),
                         internal_state=internal, simulation_indices=indices)
        return env._observe().as_subclass(th.Tensor).clone().reshape(env.n_sim, env.n_cars, -1)

    def test_floor_wall_ceiling_air_and_demo_without_stepping(self):
        env = self.make_env()
        positions = th.tensor([[[0., 0., 17.], [2000., 0., 500.]]] * 6, device="cuda")
        positions[1, 0, 2] = 500
        positions[2, 0] = th.tensor([4096. - 17, 0., 500.], device="cuda")
        positions[3, 0] = th.tensor([0., 0., 2048. - 17], device="cuda")
        positions[4, 0, 2] = 500
        rotations = th.zeros((6, 2, 4), device="cuda")
        rotations[..., 3] = 1
        rotations[2, 0] = th.tensor([0., -math.sqrt(.5), 0., math.sqrt(.5)], device="cuda")
        rotations[3, 0] = th.tensor([1., 0., 0., 0.], device="cuda")
        internal = th.zeros((6, 2, carl.CAR_INTERNAL_SIZE), device="cuda")
        internal[1, 0, 0] = 1  # Stale grounded flag must not survive an aerial reset.
        internal[4, 0, [3, 4, 5, 6]] = th.tensor([1., 1., 1., .07], device="cuda")
        demo = th.zeros((6, 2), device="cuda", dtype=th.bool)
        demo[5, 0] = True
        ticks = env._tensor(env._env.get_episode_ticks()).clone()
        result = self.restore(env, positions, rotations, internal, demo=demo)
        expected = th.tensor([1, 0, 1, 1, 0, 0], device="cuda", dtype=th.bool)
        th.testing.assert_close(result[:, 0, 25].bool(), expected)
        th.testing.assert_close(result[:, 0, -8:-4].sum(-1) >= 3, expected)
        self.assertEqual(result[4, 0, -22].item(), 1)  # previous jump input
        self.assertEqual(result[4, 0, -19].item(), 1)  # lastJump
        self.assertAlmostEqual(result[4, 0, -16].item(), .07, places=6)
        th.testing.assert_close(result[:, 0, 9:12], positions[:, 0], rtol=0, atol=0)
        th.testing.assert_close(result[:, 0, 12:18], th.zeros((6, 6), device="cuda"), rtol=0, atol=0)
        th.testing.assert_close(env._tensor(env._env.get_episode_ticks()), ticks, rtol=0, atol=0)
        th.testing.assert_close(env._observe().as_subclass(th.Tensor).reshape_as(result), result)

    def test_selected_simulations_sentinel_and_all_team_views(self):
        for team in (1, 2, 3):
            env = self.make_env(simulations=3, team=team)
            before = env._observe().as_subclass(th.Tensor).clone().reshape(3, team * 2, -1)
            positions = th.zeros((2, team * 2, 3), device="cuda")
            positions[..., 2] = 17
            positions[..., 0] = th.arange(team * 2, device="cuda") * 200 - 600
            internal = th.zeros((2, team * 2, carl.CAR_INTERNAL_SIZE), device="cuda")
            internal[0, :, 5] = 1
            indices = th.tensor([2, -1], device="cuda", dtype=th.long)
            result = self.restore(env, positions, internal=internal, indices=indices)
            th.testing.assert_close(result[:2], before[:2], rtol=0, atol=0)
            self.assertTrue((result[2, :, -8:-4] == 1).all())
            self.assertTrue((result[2, :, -22] == 1).all())
            self.assertTrue((result[2, :, -19] == 1).all())
            action = th.zeros((env.n_envs, 7), device="cuda", dtype=th.int32)
            action[2 * env.n_cars:, 6] = 1
            next_obs = env.step(action)[0].as_subclass(th.Tensor).reshape_as(result)
            # Holding the restored jump latch must not produce a fresh jump.
            self.assertFalse(next_obs[2, :, -18].bool().any())
            self.assertFalse(next_obs[2, :, -17].bool().any())

    def test_contacts_with_other_restored_car_are_order_independent(self):
        env = self.make_env(simulations=2)
        positions = th.tensor([[[0., 0., 17.], [0., 0., 65.]],
                               [[0., 0., 65.], [0., 0., 17.]]], device="cuda")
        result = self.restore(env, positions)
        self.assertGreaterEqual(result[0, 1, -8:-4].sum().item(), 3)
        th.testing.assert_close(result[0, 1, -8:-4], result[1, 0, -8:-4])

    def test_ball_contacts_do_not_advance_jump_or_flip_state(self):
        env = self.make_env(simulations=2)
        ball = th.tensor([[0., 0., 500.], [1000., 0., 500.]], device="cuda")
        env.set_ball(ball, th.zeros_like(ball), th.zeros_like(ball))
        positions = th.tensor([[[0., 0., 590.], [2000., 0., 500.]]] * 2, device="cuda")
        result = self.restore(env, positions)
        self.assertGreaterEqual(result[0, 0, -8:-4].sum().item(), 3)
        self.assertEqual(result[1, 0, -8:-4].sum().item(), 0)
        th.testing.assert_close(result[..., -18:-15], th.zeros((2, 2, 3), device="cuda"))

    def test_opt_in_preserves_prefix_and_controller_is_car_local(self):
        for team in (1, 2, 3):
            for normalize in (False, True):
                extended = self.make_env(simulations=1, team=team, normalize=normalize)
                base = CARLTorchVectorEnv(1, team, team, frameskip=1, normalize=normalize,
                                         discrete_actions=True)
                self.addCleanup(base.close)
                base.reset()
                actions = th.tensor([[.3, -.4, .5, 1., 1., -.6, 0.]], device="cuda")
                actions = actions.expand(team * 2, -1).contiguous()
                # Exercise the native continuous path to verify the actual
                # throttle/steer/yaw/pitch/roll/jump/boost/slide ordering.
                base._env.step_continuous(actions.reshape(1, team * 2, 7))
                extended._env.step_continuous(actions.reshape(1, team * 2, 7))
                old = base._observe().as_subclass(th.Tensor)
                new = extended._observe()
                th.testing.assert_close(new.as_subclass(th.Tensor)[..., :-27], old, rtol=0, atol=0)
                expected = th.tensor([[.5, .3, .3, -.4, -.6, 0., 1., 1.]], device="cuda")
                th.testing.assert_close(new.ego_previous_controls.as_subclass(th.Tensor),
                                        expected.expand(team * 2, -1))
                self.assertEqual(new.shape[-1], 112 + 27 * team * 2)

    def test_timeout_final_observation_keeps_controller_before_autoreset(self):
        env = self.make_env(simulations=1, no_touch_timeout_ticks=1)
        action = th.ones((2, 7), device="cuda", dtype=th.int32)
        observation, _, terminated, truncated, info = env.step(action)
        self.assertTrue(truncated.all())
        self.assertFalse(terminated.any())
        self.assertEqual(info["final_obs"].shape, observation.shape)
        self.assertTrue(info["final_obs"].ego_previous_controls[..., 5].bool().all())
        self.assertFalse(observation.ego_previous_controls.bool().any())


if __name__ == "__main__":
    th.set_num_threads(1)
    unittest.main()
