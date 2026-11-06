import unittest

import torch
from carl.gymnasium.torch import CARLTorchVectorEnv


class StreamHandoffTests(unittest.TestCase):

    def test_stream_handoff(self):
        def run(synchronize):
            env = CARLTorchVectorEnv(
                n_sim=4, n_blue=1, n_orange=1, seed=7, synchronize=synchronize,
            )
            self.addCleanup(env.close)
            actions = torch.zeros(env.action_space.shape, device=env.device)
            frames = [env.reset().clone()]
            for step in range(8):
                if step == 4:
                    zeros = torch.zeros((env.n_sim, 3), device=env.device)
                    position = zeros.clone()
                    position[:, 2] = 800.0
                    env.set_ball(position, zeros, zeros)
                    torch.cuda._sleep(50_000_000)
                    frames.append(torch.from_dlpack(env._env.get_obs()).clone())
                    obs = env.reset()
                else:
                    obs = env.step(actions)[0]
                frames.append(obs.clone())
            env.close()
            return frames

        expected = run(synchronize=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = run(synchronize=False)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
