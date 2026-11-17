"""Controller schema compatibility and immediate replay-reset observations."""
import math
import unittest

import carl
import torch as th
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import CARLActionCodec
from carl.gymnasium.state import CARLObservation


class ControllerViewTests(unittest.TestCase):
    def test_v3_views_and_action_masks(self):
        self.assertEqual(carl.OBS_CONTROLLER_VERSION, 3)
        self.assertEqual(carl.OBS_EGO_CONTROLLER, 5)
        self.assertEqual(carl.OBS_PREVIOUS_CONTROLS, 8)
        for cars in (2, 4, 6):
            base_size = 85 + 27 * cars
            base = th.zeros((3, base_size))
            base[:, -2] = 1
            base[:, -1] = th.tensor([.5, 0., .5])
            base[2, 25] = 1
            default = th.cat((base, th.zeros((3, 5))), -1)
            extended = th.cat((default, th.zeros((3, 8))), -1)
            # New trailing fields must not be mistaken for the dodge window.
            extended[:, -2:] = -1
            new = CARLObservation.from_tensor(extended, cars)
            th.testing.assert_close(new.ego_has_flip_or_jump, th.ones(3, dtype=th.bool))
            th.testing.assert_close(new.ego_flip_window_remaining, base[:, -1])
            th.testing.assert_close(CARLActionCodec(True).mask(extended)[:, 17],
                                    th.tensor([True, False, True]))
            th.testing.assert_close(CARLActionCodec(True).mask(extended),
                                    CARLActionCodec().mask(default))
            default_view = CARLObservation.from_tensor(default, cars)
            th.testing.assert_close(new.ego_controller, default_view.ego_controller)
            with self.assertRaisesRegex(ValueError, "does not include previous controls"):
                _ = default_view.ego_previous_controls
            self.assertEqual(new.ego_controller.shape, (3, 5))
            new.ego_previous_controls.fill_(.25)
            th.testing.assert_close(extended[:, -8:], th.full((3, 8), .25))
            for old_size in (base_size, base_size + 27):
                old = CARLObservation.from_tensor(th.zeros((3, old_size)), cars)
                with self.assertRaisesRegex(ValueError, "ego controller v3"):
                    _ = old.ego_controller
            for name in ("ego_last_jump", "ego_jump_time", "ego_boost_min_time_remaining",
                         "ego_flip_relative_torque", "ego_handbrake", "ego_wheel_contacts",
                         "ego_is_auto_flipping", "ego_auto_flip_time", "ego_auto_flip_torque_scale"):
                self.assertFalse(hasattr(new, name), name)

    def test_controller_state_has_no_opt_out(self):
        with self.assertRaises(TypeError):
            CARLTorchVectorEnv(1, 1, 1, ego_controller=False)
        with self.assertRaises(TypeError):
            carl.Env(1, 1, 1, 0, ego_controller=False)


@unittest.skipUnless(th.cuda.is_available(), "requires CUDA or ROCm")
class ControllerResetTests(unittest.TestCase):
    def make_env(self, simulations=6, team=1, **options):
        env = CARLTorchVectorEnv(simulations, team, team, frameskip=1,
                                discrete_actions=True, **options)
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
        controller = result[..., -5:]
        self.assertEqual(controller[4, 0, 0].item(), 1)  # hasJumped
        self.assertEqual(controller[4, 0, 1].item(), 1)  # isJumping
        self.assertGreater(controller[5, 0, 4].item(), 0)  # demo timer
        th.testing.assert_close(result[:, 0, 9:12], positions[:, 0], rtol=0, atol=0)
        th.testing.assert_close(result[:, 0, 12:18], th.zeros((6, 6), device="cuda"), rtol=0, atol=0)
        th.testing.assert_close(env._tensor(env._env.get_episode_ticks()), ticks, rtol=0, atol=0)
        th.testing.assert_close(env._observe().as_subclass(th.Tensor).reshape_as(result), result)

    def test_selected_simulations_sentinel_and_all_team_views(self):
        for team in (1, 2, 3):
            env = self.make_env(simulations=3, team=team, previous_controls=True)
            before = env._observe().as_subclass(th.Tensor).clone().reshape(3, team * 2, -1)
            positions = th.zeros((2, team * 2, 3), device="cuda")
            positions[..., 2] = 17
            positions[..., 0] = th.arange(team * 2, device="cuda") * 200 - 600
            internal = th.zeros((2, team * 2, carl.CAR_INTERNAL_SIZE), device="cuda")
            internal[0, :, 5] = 1
            indices = th.tensor([2, -1], device="cuda", dtype=th.long)
            result = self.restore(env, positions, internal=internal, indices=indices)
            th.testing.assert_close(result[:2], before[:2], rtol=0, atol=0)
            self.assertTrue(result[2, :, 25].bool().all())
            self.assertTrue((result[2, :, -3] == 1).all())  # previous jump input
            action = th.zeros((env.n_envs, 7), device="cuda", dtype=th.int32)
            action[2 * env.n_cars:, 6] = 1
            next_obs = env.step(action)[0].as_subclass(th.Tensor).reshape_as(result)
            # Holding the restored jump latch must not produce a fresh jump.
            self.assertFalse(next_obs[2, :, -13].bool().any())  # hasJumped
            self.assertFalse(next_obs[2, :, -12].bool().any())  # isJumping

    def test_contacts_with_other_restored_car_are_order_independent(self):
        env = self.make_env(simulations=2)
        positions = th.tensor([[[0., 0., 17.], [0., 0., 65.]],
                               [[0., 0., 65.], [0., 0., 17.]]], device="cuda")
        result = self.restore(env, positions)
        self.assertTrue(result[0, 1, 25].bool())
        th.testing.assert_close(result[0, 1, 25], result[1, 0, 25])

    def test_ball_contacts_do_not_advance_jump_or_flip_state(self):
        env = self.make_env(simulations=2)
        ball = th.tensor([[0., 0., 500.], [1000., 0., 500.]], device="cuda")
        env.set_ball(ball, th.zeros_like(ball), th.zeros_like(ball))
        positions = th.tensor([[[0., 0., 590.], [2000., 0., 500.]]] * 2, device="cuda")
        result = self.restore(env, positions)
        self.assertTrue(result[0, 0, 25].bool())
        self.assertFalse(result[1, 0, 25].bool())
        th.testing.assert_close(result[..., -5:-1], th.zeros((2, 2, 4), device="cuda"))

    def test_optional_controls_preserve_default_prefix_and_are_car_local(self):
        for team in (1, 2, 3):
            for normalize in (False, True):
                env = self.make_env(simulations=1, team=team, normalize=normalize)
                extended = self.make_env(simulations=1, team=team, normalize=normalize,
                                         previous_controls=True)
                actions = th.tensor([[.3, -.4, .5, 1., 1., -.6, 0.]], device="cuda")
                actions = actions.expand(team * 2, -1).contiguous()
                # Exercise the native continuous path to verify the actual
                # throttle/steer/yaw/pitch/roll/jump/boost/slide ordering.
                env._env.step_continuous(actions.reshape(1, team * 2, 7))
                extended._env.step_continuous(actions.reshape(1, team * 2, 7))
                default = env._observe()
                observation = extended._observe()
                th.testing.assert_close(observation.as_subclass(th.Tensor)[..., :-8],
                                        default.as_subclass(th.Tensor), rtol=0, atol=0)
                th.testing.assert_close(env.action_codec.mask(default),
                                        extended.action_codec.mask(observation))
                expected = th.tensor([[.5, .3, .3, -.4, -.6, 0., 1., 1.]], device="cuda")
                th.testing.assert_close(observation.ego_previous_controls.as_subclass(th.Tensor),
                                        expected.expand(team * 2, -1))
                self.assertEqual(observation.shape[-1], 98 + 27 * team * 2)
                self.assertEqual(default.shape[-1], 90 + 27 * team * 2)
                self.assertEqual(observation.ego_controller.shape[-1], 5)
                self.assertEqual(default.ego_controller.shape[-1], 5)

    def test_timeout_final_observation_keeps_controller_before_autoreset(self):
        for previous_controls in (False, True):
            env = self.make_env(simulations=1, no_touch_timeout_ticks=1,
                                previous_controls=previous_controls)
            action = th.ones((2, 7), device="cuda", dtype=th.int32)
            observation, _, terminated, truncated, info = env.step(action)
            self.assertTrue(truncated.all())
            self.assertFalse(terminated.any())
            self.assertEqual(info["final_obs"].shape, observation.shape)
            self.assertTrue(info["final_obs"].ego_has_jumped.all())
            self.assertFalse(observation.ego_has_jumped.any())
            if previous_controls:
                self.assertTrue(info["final_obs"].ego_previous_controls[..., 5].bool().all())
                self.assertFalse(observation.ego_previous_controls.bool().any())


if __name__ == "__main__":
    th.set_num_threads(1)
    unittest.main()
