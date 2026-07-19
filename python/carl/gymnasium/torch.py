from typing import Any

import numpy as np
import torch as th
import gymnasium as gym

from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

import carl

class CARLTorchVectorEnv(VectorEnv):

    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.SAME_STEP}

    render_mode = spec = None

    def __init__(
        self,
        n_sim:         int,
        n_blue:        int,
        n_orange:      int,
        seed:          int = 0,
        frameskip:     int = 8,
        *,
        invert_orange: bool = True,
        copy_outputs:  bool = True,
        synchronize:   bool = True,
    ) -> None:
        super().__init__()

        self.num_envs = n_sim
        self.device = th.device("cuda:0")
        self.closed = False

        self._seed = seed
        self._copy_outputs = copy_outputs
        self._synchronize = synchronize

        self._env = carl.Env(
            n_sim=n_sim,
            n_blue=n_blue,
            n_orange=n_orange,
            seed=seed,
            frameskip=frameskip,
            invert_orange=invert_orange,
        )

        self.n_cars = self._env.n_cars
        self._action_shape = (n_sim, self.n_cars, 7)

        # Space specs for single env
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.obs_dim,), dtype=np.float32
        )

        self.single_action_space = gym.spaces.MultiDiscrete(
            np.asarray(self._env.action_nvec, dtype=np.int64)
        )

        # Full batch obs/act spaces
        self.observation_space = batch_space(self.single_observation_space, n_sim)

        self.action_space = batch_space(self.single_action_space, n_sim)

    def _sync(self) -> None:
        if self._synchronize:
            th.cuda.synchronize(self.device)

    def _from_carl(self, capsule: object) -> th.Tensor:
        tensor = th.from_dlpack(capsule)
        return tensor.clone() if self._copy_outputs else tensor

    def reset(
        self,
        *,
        seed:    int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[th.Tensor, dict[str, Any]]:
        if self.closed:
            raise RuntimeError("Environment is closed")

        if seed is not None and seed != self._seed:
            raise ValueError("CARL's seed is constructor-only")

        if options:
            raise NotImplementedError("CARL reset options are not supported")

        self._env.reset()
        self._sync()

        return self._from_carl(self._env.get_obs()), {}

    def step(
        self, actions: th.Tensor | np.ndarray
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, dict[str, Any]]:
        if self.closed:
            raise RuntimeError("Environment is closed")

        act = th.as_tensor(actions, dtype=th.int32, device=self.device).contiguous()

        if tuple(act.shape) != self._action_shape:
            raise ValueError(
                f"Expected actions shaped {self._action_shape}, "
                f"got {tuple(act.shape)}"
            )

        self._sync()

        obs_capsule = self._env.step(act)

        self._sync()

        obs = self._from_carl(obs_capsule)
        rew = self._from_carl(self._env.get_rewards())
        don = self._from_carl(self._env.get_dones())

        terms = don & rew.ne(0)
        trunc = don & ~terms

        info = {
            "ball_touches": self._from_carl(self._env.get_ball_touches())
        }

        return obs, rew, terms, trunc, info

    def render(self) -> None:
        return None

    def close(self, **kwargs: Any) -> None:
        if self.closed:
            return
        self._sync()
        del self._env
        self.closed = True
