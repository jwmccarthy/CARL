import os
from pathlib import Path

import carl
import torch
from carl.gymnasium.state import CARLObservation
from carl.gymnasium.torch import CARLTorchVectorEnv


def check_tensor(value, shape, dtype):
    assert isinstance(value, torch.Tensor)
    assert value.is_cuda, "Expected a real CUDA tensor"
    assert tuple(value.shape) == tuple(shape), (value.shape, shape)
    assert value.dtype == dtype, value.dtype
    assert torch.isfinite(value).all().item(), "Non-finite tensor values"


def main():
    arena = Path(os.environ["CARL_ARENA_OBJ"])
    assert arena.is_file(), f"Arena asset missing: {arena}"

    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a working CUDA GPU/backend")

    env = CARLTorchVectorEnv(
        n_sim=1,
        n_blue=1,
        n_orange=1,
        frameskip=1,
        discrete_actions=True,
        synchronize=True,
    )

    try:
        observation = env.reset()

        assert isinstance(observation, CARLObservation)
        check_tensor(observation, env.observation_space.shape, torch.float32)
        check_tensor(observation.ball.position, (env.n_envs, 3), torch.float32)
        check_tensor(
            observation.cars.position, (env.n_envs, env.n_cars, 3), torch.float32
        )

        actions = torch.zeros(
            (env.n_envs, len(carl.ACTION_NVECS)),
            dtype=torch.long,
            device=env.device,
        )
        observation, reward, terminated, truncated, info = env.step(actions)

        assert isinstance(observation, CARLObservation)

        check_tensor(observation, env.observation_space.shape, torch.float32)
        check_tensor(reward, (env.n_envs,), torch.float32)
        check_tensor(terminated, (env.n_envs,), torch.bool)
        check_tensor(truncated, (env.n_envs,), torch.bool)
        
        assert isinstance(info, dict)

        torch.cuda.synchronize(env.device)
    finally:
        env.close()

    print("CARL GPU smoke passed (one 1v1 simulation, one step)")


if __name__ == "__main__":
    main()
