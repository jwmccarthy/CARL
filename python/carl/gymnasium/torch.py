from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch as th
import gymnasium as gym

from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

import carl

from .action import CARLActionCodec
from .state import BOOST_PAD_POSITIONS, CarlEvents, CarlState, RewardContext


RewardFunction = Callable[[RewardContext], th.Tensor]


class CARLTorchVectorEnv(VectorEnv):
    metadata = {
        "render_modes":    [],
        "autoreset_mode": AutoresetMode.SAME_STEP,
    }

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
        synchronize:   bool = False,
        reward_funcs:  Iterable[RewardFunction] | None = None,
        reward_scale:  float = 1.0,
    ) -> None:
        super().__init__()

        self.n_sim = n_sim
        self.device = th.device("cuda:0")
        self.closed = False

        self._seed = seed
        self._copy_outputs = copy_outputs
        self._synchronize = synchronize
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        self.reward_funcs = list(reward_funcs or ())
        self.reward_scale = reward_scale
        self._state: CarlState | None = None

        self._env = carl.Env(
            n_sim=n_sim,
            n_blue=n_blue,
            n_orange=n_orange,
            seed=seed,
            frameskip=frameskip,
            invert_orange=invert_orange,
        )

        self.n_cars = self._env.n_cars
        self.n_envs = n_sim * self.n_cars
        self._action_shape = (self.n_envs, 7)
        self._team_sign = th.cat(
            (
                th.ones(n_blue, device=self.device),
                -th.ones(n_orange, device=self.device),
            )
        )
        self._boost_pad_positions = th.tensor(
            BOOST_PAD_POSITIONS, dtype=th.float32, device=self.device
        )
        self._episode_return = th.zeros(
            self.n_envs, dtype=th.float32, device=self.device
        )
        self._episode_length = th.zeros(self.n_envs, dtype=th.int64, device=self.device)
        self.action_codec = CARLActionCodec().to(self.device)
        self._set_spaces()

    def _set_spaces(self) -> None:
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._env.obs_dim,),
            dtype=np.float32,
        )

        self.single_action_space = gym.spaces.MultiDiscrete(
            np.asarray(self._env.action_nvec[0], dtype=np.int64)
        )
        self.observation_space = batch_space(self.single_observation_space, self.n_envs)
        self.action_space = batch_space(self.single_action_space, self.n_envs)

    def _sync(self) -> None:
        if self._synchronize:
            th.cuda.synchronize(self.device)

    def _from_carl(self, capsule: object) -> th.Tensor:
        tensor = th.from_dlpack(capsule)
        return tensor.clone() if self._copy_outputs else tensor

    def _state_from_carl(self, capsule: object) -> CarlState:
        # State snapshots must survive reuse of the native I/O buffers.
        raw = th.from_dlpack(capsule).clone()
        return CarlState.from_raw(
            raw, self.n_cars, self._boost_pad_positions, self._team_sign
        )

    def register_reward(self, reward_function: RewardFunction) -> RewardFunction:
        if not callable(reward_function):
            raise TypeError("reward_function must be callable")
        self.reward_funcs.append(reward_function)
        return reward_function

    def action_mask(self, observation: th.Tensor) -> th.Tensor:
        return self.action_codec.mask(observation)

    def _prepare_actions(self, actions: th.Tensor | np.ndarray) -> th.Tensor:
        act = th.as_tensor(actions, dtype=th.int32, device=self.device).contiguous()
        if tuple(act.shape) != self._action_shape:
            raise ValueError(
                f"Expected actions shaped {self._action_shape}, got {tuple(act.shape)}"
            )
        return act.view(self.n_sim, self.n_cars, 7)

    def _custom_reward(
        self,
        actions: th.Tensor,
        score_delta: th.Tensor,
        done: th.Tensor,
        terminated: th.Tensor,
        truncated: th.Tensor,
    ) -> th.Tensor:
        if self._state is None:
            raise RuntimeError("reset must be called before using custom rewards")

        current = self._state_from_carl(self._env.get_transition_state())

        context = RewardContext(
            current=current,
            previous=self._state,
            events=CarlEvents(
                score_delta=score_delta,
                done=done,
                terminated=terminated,
                truncated=truncated,
            ),
            actions=actions,
        )

        reward = th.zeros(
            (self.n_sim, self.n_cars),
            dtype=th.float32,
            device=self.device,
        )

        for reward_func in self.reward_funcs:
            component = reward_func(context)
            if component.shape != reward.shape:
                raise ValueError(
                    f"Reward returned {tuple(component.shape)}; "
                    f"expected {tuple(reward.shape)}"
                )
            reward += component

        return reward * self.reward_scale

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> th.Tensor:
        if self.closed:
            raise RuntimeError("Environment is closed")

        if seed is not None and seed != self._seed:
            raise ValueError("CARL's seed is constructor-only")

        if options:
            raise NotImplementedError("CARL reset options are not supported")

        self._env.reset()
        self._sync()

        observation = self._from_carl(self._env.get_obs())
        self._state = (
            self._state_from_carl(self._env.get_state())
            if self.reward_funcs
            else None
        )
        self._episode_return.zero_()
        self._episode_length.zero_()
        return observation.view(self.n_envs, self._env.obs_dim)

    def step(
        self, actions: th.Tensor | np.ndarray
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, dict[str, Any]]:
        if self.closed:
            raise RuntimeError("Environment is closed")

        act = self._prepare_actions(actions)

        self._sync()

        obs_capsule = self._env.step(act)

        self._sync()

        obs = self._from_carl(obs_capsule).view(self.n_envs, self._env.obs_dim)
        score_delta = self._from_carl(self._env.get_rewards())
        don = self._from_carl(self._env.get_dones())

        terms = don & score_delta.ne(0)
        trunc = don & ~terms

        if self.reward_funcs:
            rew = self._custom_reward(act, score_delta, don, terms, trunc).reshape(
                self.n_envs
            )
            self._state = self._state_from_carl(self._env.get_state())
        else:
            rew = (score_delta[:, None] * self._team_sign).reshape(self.n_envs)

        terms = terms[:, None].expand(-1, self.n_cars).reshape(self.n_envs)
        trunc = trunc[:, None].expand(-1, self.n_cars).reshape(self.n_envs)
        done = terms | trunc

        self._episode_return += rew
        self._episode_length += 1
        finished = done.nonzero(as_tuple=True)[0]
        info = {"reward": [], "length": []}
        if finished.numel():
            info = {
                "reward": self._episode_return[finished].cpu().tolist(),
                "length": self._episode_length[finished].cpu().tolist(),
            }
            self._episode_return[finished] = 0
            self._episode_length[finished] = 0

        return obs, rew, terms, trunc, info

    def render(self) -> None:
        return None

    def close(self, **kwargs: Any) -> None:
        if self.closed:
            return
        self._sync()
        del self._env
        self.closed = True
