"""Restored controller/contact observations must be valid before the first step."""
import math
import unittest

import torch as th

from carl.gymnasium import CARLTorchVectorEnv


@unittest.skipUnless(th.cuda.is_available(), "requires CUDA")
class ResetObservationTests(unittest.TestCase):
    def make_env(self, simulations=6, team=1):
        env = CARLTorchVectorEnv(simulations, team, team, frameskip=1, normalize=False,
                                ego_controller=True, discrete_actions=True, capture_events=True)
        self.addCleanup(env.close)
        env.reset()
        return env

    def restore(self, env, positions, rotations=None, internal=None, indices=None, demo=None):
        n, cars = positions.shape[:2]
        if rotations is None:
            rotations = th.zeros((n, cars, 4), device="cuda")
            rotations[..., 3] = 1
        if internal is None:
            internal = th.zeros((n, cars, 19), device="cuda")
        if demo is None:
            demo = th.zeros((n, cars), device="cuda", dtype=th.bool)
        env._env.set_car(positions, rotations, th.zeros_like(positions), th.zeros_like(positions),
                         demo, boost=th.full((n, cars), 50., device="cuda"),
                         internal_state=internal, simulation_indices=indices)
        return env._observe().as_subclass(th.Tensor).clone().reshape(env.n_sim, env.n_cars, -1)

    def test_floor_wall_ceiling_air_and_demo_without_stepping(self):
        env = self.make_env()
        positions = th.tensor([[[0., 0., 17.], [2000., 0., 500.]]] * 6, device="cuda")
        positions[1, 0] = th.tensor([0., 0., 500.], device="cuda")
        positions[2, 0] = th.tensor([4096. - 17, 0., 500.], device="cuda")
        positions[3, 0] = th.tensor([0., 0., 2048. - 17], device="cuda")
        positions[4, 0, 2] = 500
        rotation = th.zeros((6, 2, 4), device="cuda")
        rotation[..., 3] = 1
        rotation[2, 0] = th.tensor([0., -math.sqrt(.5), 0., math.sqrt(.5)], device="cuda")
        rotation[3, 0] = th.tensor([1., 0., 0., 0.], device="cuda")
        internal = th.zeros((6, 2, 19), device="cuda")
        internal[1, 0, 0] = 1  # Deliberately stale ground flag in open air.
        internal[4, 0, [3, 4, 5, 6]] = th.tensor([1., 1., 1., .07], device="cuda")
        demo = th.zeros((6, 2), device="cuda", dtype=th.bool)
        demo[5, 0] = True
        ticks = env._tensor(env._env.get_episode_ticks()).clone()
        result = self.restore(env, positions, rotation, internal, demo=demo)
        expected = th.tensor([1, 0, 1, 1, 0, 0], device="cuda", dtype=th.bool)
        th.testing.assert_close(result[:, 0, 25].bool(), expected)
        th.testing.assert_close(result[:, 0, -8:-4].sum(-1) >= 3, expected)
        self.assertEqual(result[4, 0, -22].item(), 1)  # previous controller jump
        self.assertEqual(result[4, 0, -19].item(), 1)  # lastJump
        self.assertAlmostEqual(result[4, 0, -16].item(), .07, places=6)
        th.testing.assert_close(result[:, 0, 9:12], positions[:, 0], rtol=0, atol=0)
        th.testing.assert_close(env._tensor(env._env.get_episode_ticks()), ticks, rtol=0, atol=0)
        self.assertFalse(env.events.cars.jumped.any())
        self.assertFalse(env.events.cars.double_jumped.any())
        # Queries stay stable: no settle tick and no repeated hidden-state update.
        th.testing.assert_close(env._observe().as_subclass(th.Tensor).reshape_as(result), result)

    def test_selected_simulations_sentinel_and_all_team_views(self):
        for team in (1, 2, 3):
            env = self.make_env(simulations=3, team=team)
            before = env._observe().as_subclass(th.Tensor).clone().reshape(3, team * 2, -1)
            pos = th.zeros((2, team * 2, 3), device="cuda")
            pos[..., 2] = 17
            pos[..., 0] = th.arange(team * 2, device="cuda") * 200 - 600
            internal = th.zeros((2, team * 2, 19), device="cuda")
            internal[0, :, 5] = 1
            indices = th.tensor([2, -1], device="cuda", dtype=th.long)
            result = self.restore(env, pos, internal=internal, indices=indices)
            th.testing.assert_close(result[:2], before[:2], rtol=0, atol=0)
            self.assertTrue((result[2, :, -8:-4] == 1).all())
            self.assertTrue((result[2, :, -22] == 1).all())
            self.assertTrue((result[2, :, -19] == 1).all())
            # A held jump is not a new press; the next step preserves that latch.
            action = th.zeros((env.n_envs, 7), device="cuda", dtype=th.int32)
            action[2 * env.n_cars:, 6] = 1
            env.step(action)
            self.assertFalse(env.events.cars.jumped.reshape(3, -1)[2].any())

    def test_contacts_with_other_restored_car_are_order_independent(self):
        env = self.make_env(simulations=2)
        # Upper car's downward wheel rays hit the lower car's roof. Swapping
        # actor order must preserve the result, requiring a post-restore kernel.
        pos = th.tensor([[[0., 0., 17.], [0., 0., 65.]],
                         [[0., 0., 65.], [0., 0., 17.]]], device="cuda")
        result = self.restore(env, pos)
        self.assertGreaterEqual(result[0, 1, -8:-4].sum().item(), 3)
        th.testing.assert_close(result[0, 1, -8:-4], result[1, 0, -8:-4])

    def test_ball_contact_query_does_not_emit_jump_or_flip_reset(self):
        env = self.make_env(simulations=2)
        ball = th.tensor([[0., 0., 500.], [1000., 0., 500.]], device="cuda")
        env._env.set_ball(ball, th.zeros_like(ball), th.zeros_like(ball))
        pos = th.tensor([[[0., 0., 590.], [2000., 0., 500.]]] * 2, device="cuda")
        result = self.restore(env, pos)
        self.assertGreaterEqual(result[0, 0, -8:-4].sum().item(), 3)
        self.assertEqual(result[1, 0, -8:-4].sum().item(), 0)
        self.assertFalse(env.events.cars.jumped.any())
        self.assertFalse(env.events.cars.double_jumped.any())
        self.assertFalse(env.events.cars.dodged.any())


if __name__ == "__main__":
    th.set_num_threads(1)
    unittest.main()
