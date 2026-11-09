import unittest

import torch
from carl.gymnasium.torch import CARLTorchVectorEnv
from test_gpu import check_tensor


class GpuBackendTests(unittest.TestCase):

    def make_env(self, n_sim=4, **kwargs):
        env = CARLTorchVectorEnv(n_sim=n_sim, n_blue=1, n_orange=1, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_outputs(self):
        for n_sim, discrete in ((17, False), (17, True), (8192, False)):
            with self.subTest(n_sim=n_sim, discrete=discrete):
                env = self.make_env(
                    n_sim,
                    discrete_actions=discrete,
                    max_ticks=16,
                    overtime_timeout_ticks=8,
                )
                dtype = torch.int64 if discrete else torch.float32
                actions = torch.zeros(env.action_space.shape, dtype=dtype, device=env.device)
                initial = env.reset()
                snapshot = initial.clone()
                check_tensor(initial, env.observation_space.shape, torch.float32)
                self.assertEqual(
                    torch.from_dlpack(env._env.get_obs()).__dlpack_device__(),
                    (10 if torch.version.hip else 2, 0),
                )
                saw_truncation = False

                for _ in range(8):
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    for value, shape, dtype in (
                        (obs, env.observation_space.shape, torch.float32),
                        (reward, (env.n_envs,), torch.float32),
                        (terminated, (env.n_envs,), torch.bool),
                        (truncated, (env.n_envs,), torch.bool),
                    ):
                        check_tensor(value, shape, dtype)
                    saw_truncation |= truncated.any().item()

                self.assertTrue(saw_truncation)
                env.close()
                torch.testing.assert_close(initial, snapshot, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
