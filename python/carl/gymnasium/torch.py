import math
import numpy as np
import torch as th

from typing import Any
from collections.abc import Callable, Iterable, Mapping

import gymnasium as gym
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

import carl
from .action import CARLActionCodec
from .state import (
    BOOST_PAD_POSITIONS,
    CARLObservation,
    REGULATION_TICKS,
    CarlEvents,
    CarlState,
    RewardContext,
    RewardResult,
)

RewardFunction = Callable[[RewardContext], th.Tensor | RewardResult]
ResetState = Mapping[str, th.Tensor]
ResetStateProvider = Callable[[th.Tensor], ResetState | None]


def _ticks(ticks: int | None, seconds: float | None, default: int | None) -> int | None:
    """Resolve a tick/seconds timeout pair."""

    if ticks is not None and seconds is not None:
        raise ValueError("Specify ticks or seconds, not both")
    if seconds is not None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Timeout seconds must be positive and finite")
        ticks = math.ceil(seconds * 120)
    if ticks is not None and ticks < 1:
        raise ValueError("Timeout ticks must be positive")

    return default if ticks is None else ticks


class CARLTorchVectorEnv(VectorEnv):
    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.SAME_STEP}
    render_mode = spec = None

    def __init__(
        self,
        n_sim:                    int,
        n_blue:                   int,
        n_orange:                 int,
        seed:                     int = 0,
        frameskip:                int = 8,
        max_ticks:                int = REGULATION_TICKS,
        *,
        overtime_timeout_ticks:   int | None = None,
        overtime_timeout_seconds: float | None = None,
        no_touch_timeout_ticks:   int | None = None,
        no_touch_timeout_seconds: float | None = None,
        invert_orange:            bool = True,
        normalize:                bool = False,
        copy_outputs:             bool = True,
        synchronize:              bool = False,
        reward_funcs:             Iterable[RewardFunction] | None = None,
        reward_scale:             float = 1.0,
        reset_state_provider:     ResetStateProvider | None = None,
    ) -> None:
        super().__init__()
        
        if max_ticks < 1 or reward_scale <= 0:
            raise ValueError("max_ticks and reward_scale must be positive")

        self.n_sim = n_sim
        self.device = th.device("cuda:0")
        self.closed = False
        self._seed = seed
        self._copy_outputs = copy_outputs
        self._synchronize = synchronize
        self.reward_funcs = list(reward_funcs or ())
        self.reward_scale = reward_scale
        self.reset_state_provider = reset_state_provider
        self._state: CarlState | None = None
        self._observation: CARLObservation | None = None

        self._env = carl.Env(
            n_sim=n_sim,
            n_blue=n_blue,
            n_orange=n_orange,
            seed=seed,
            frameskip=frameskip,
            invert_orange=invert_orange,
            normalize=normalize,
        )
        self._env.max_ticks = max_ticks
        self._env.overtime_timeout_ticks = _ticks(
            overtime_timeout_ticks, overtime_timeout_seconds, REGULATION_TICKS
        )
        self._env.no_touch_timeout_ticks = _ticks(
            no_touch_timeout_ticks, no_touch_timeout_seconds, 0
        )

        self.n_cars = self._env.n_cars
        self.n_envs = n_sim * self.n_cars
        self._team_sign = th.tensor(
            [1] * n_blue + [-1] * n_orange,
            dtype=th.float32,
            device=self.device,
        )
        self._boost_pad_positions = th.tensor(
            BOOST_PAD_POSITIONS, dtype=th.float32, device=self.device
        )
        self.action_codec = CARLActionCodec().to(self.device)

        self._episode_return = th.zeros(self.n_envs, device=self.device)
        self._episode_length = th.zeros(self.n_envs, dtype=th.int64, device=self.device)
        self._score_difference = th.zeros(n_sim, dtype=th.int32, device=self.device)
        self._episode_ticks = th.zeros(n_sim, dtype=th.int32, device=self.device)
        self._overtime = th.zeros(n_sim, dtype=th.bool, device=self.device)

        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self._env.obs_dim,), np.float32
        )
        self.single_action_space = gym.spaces.MultiDiscrete(
            np.asarray(self._env.action_nvec[0], dtype=np.int64)
        )
        self.observation_space = batch_space(self.single_observation_space, self.n_envs)
        self.action_space = batch_space(self.single_action_space, self.n_envs)

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("Environment is closed")

    def _sync(self) -> None:
        if self._synchronize:
            th.cuda.synchronize(self.device)

    def _tensor(self, capsule: object, *, copy: bool | None = None) -> th.Tensor:
        x = th.from_dlpack(capsule)
        should_copy = self._copy_outputs if copy is None else copy
        return x.clone() if should_copy else x

    def _carl_state(self, capsule: object) -> CarlState:
        return CarlState.from_raw(
            self._tensor(capsule, copy=True),
            self.n_cars,
            self._boost_pad_positions,
            self._team_sign,
        )

    def _observe(self) -> CARLObservation:
        self._sync()
        obs = self._tensor(self._env.get_obs()).view(self.n_envs, self._env.obs_dim)
        observation = CARLObservation.from_tensor(obs, self.n_cars)
        self._state = self._carl_state(self._env.get_state()) if self.reward_funcs else None
        self._observation = observation if self.reward_funcs else None
        return observation

    def _apply_reset_state(self, reset_mask: th.Tensor) -> None:
        if self.reset_state_provider is None or not reset_mask.any():
            return

        state = self.reset_state_provider(reset_mask)
        if state is None:
            return

        indices = state.get("simulation_indices", reset_mask.nonzero(as_tuple=True)[0])
        if not indices.numel():
            return

        self._env.set_ball(
            state["ball_position"].contiguous(),
            state["ball_velocity"].contiguous(),
            state["ball_angular_velocity"].contiguous(),
            simulation_indices=indices.contiguous(),
        )
        self._env.set_car(
            state["car_position"].contiguous(),
            state["car_rotation"].contiguous(),
            state["car_velocity"].contiguous(),
            state["car_angular_velocity"].contiguous(),
            state["car_demoed"].contiguous(),
            boost=(
                state["car_boost"].contiguous()
                if "car_boost" in state else None
            ),
            simulation_indices=indices.contiguous(),
        )

        if "blue_score" in state:
            self._env.set_match_state(
                state["blue_score"].contiguous(),
                state["orange_score"].contiguous(),
                state["episode_ticks"].contiguous(),
                simulation_indices=indices.contiguous(),
            )

    def _clear_sim_stats(self, mask: th.Tensor | None = None) -> None:
        if mask is None:
            self._episode_return.zero_()
            self._episode_length.zero_()
            self._score_difference.zero_()
            self._episode_ticks.zero_()
            self._overtime.zero_()
        else:
            self._score_difference[mask] = 0
            self._episode_ticks[mask] = 0
            self._overtime[mask] = False

    def _per_car(self, x: th.Tensor) -> th.Tensor:
        return x[:, None].expand(-1, self.n_cars).reshape(self.n_envs)

    def register_reward(self, reward_function: RewardFunction) -> RewardFunction:
        self.reward_funcs.append(reward_function)
        return reward_function

    def action_mask(self, observation: th.Tensor) -> th.Tensor:
        return self.action_codec.mask(observation)

    def _custom_reward(
        self,
        actions:     th.Tensor,
        score_delta: th.Tensor,
        done:        th.Tensor,
        terminated:  th.Tensor,
        truncated:   th.Tensor,
    ) -> tuple[th.Tensor, dict[str, list[Any]]]:
        if self._state is None or self._observation is None:
            raise RuntimeError("reset must be called before using custom rewards")

        current_observation = CARLObservation.from_tensor(
            self._tensor(self._env.get_obs()).view(
                self.n_envs,
                self._env.obs_dim,
            ),
            self.n_cars,
        )
        context = RewardContext(
            current=self._carl_state(self._env.get_transition_state()),
            previous=self._state,
            current_observation=current_observation,
            previous_observation=self._observation,
            events=CarlEvents(
                score_delta=score_delta,
                done=done,
                terminated=terminated,
                truncated=truncated,
            ),
            actions=actions,
            score_difference=self._score_difference,
            episode_ticks=self._episode_ticks,
            overtime=self._overtime,
        )

        reward = th.zeros((self.n_sim, self.n_cars), device=self.device)
        info: dict[str, list[Any]] = {}

        for func in self.reward_funcs:
            result = func(context)
            if isinstance(result, RewardResult):
                info.update(result.info)
                result = result.reward
            if result.shape != reward.shape:
                raise ValueError(f"Reward must have shape {tuple(reward.shape)}")
            reward += result

        return (reward * self.reward_scale).reshape(self.n_envs), info

    def reset(
        self,
        *,
        seed:    int | None = None,
        options: dict[str, Any] | None = None,
    ) -> th.Tensor:
        self._check_open()
        if seed is not None and seed != self._seed:
            raise ValueError("CARL's seed is constructor-only")
        if options:
            raise NotImplementedError("CARL reset options are not supported")

        self._env.reset()
        self._apply_reset_state(th.ones(self.n_sim, dtype=th.bool, device=self.device))
        self._clear_sim_stats()
        return self._observe()

    def set_ball(
        self,
        position:           th.Tensor,
        velocity:           th.Tensor,
        angular_velocity:   th.Tensor,
        *,
        simulation_indices: th.Tensor | None = None,
    ) -> th.Tensor:
        self._check_open()
        self._sync()
        self._env.set_ball(
            position.contiguous(),
            velocity.contiguous(),
            angular_velocity.contiguous(),
            simulation_indices=(
                simulation_indices.contiguous()
                if simulation_indices is not None else None
            ),
        )
        return self._observe()

    def set_car(
        self,
        position:           th.Tensor,
        rotation:           th.Tensor,
        velocity:           th.Tensor,
        angular_velocity:   th.Tensor,
        demoed:             th.Tensor,
        *,
        boost:              th.Tensor | None = None,
        simulation_indices: th.Tensor | None = None,
    ) -> th.Tensor:
        self._check_open()
        self._sync()
        self._env.set_car(
            position.contiguous(),
            rotation.contiguous(),
            velocity.contiguous(),
            angular_velocity.contiguous(),
            demoed.contiguous(),
            boost=boost.contiguous() if boost is not None else None,
            simulation_indices=(
                simulation_indices.contiguous()
                if simulation_indices is not None else None
            ),
        )
        return self._observe()

    def step(
        self, actions: th.Tensor | np.ndarray
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, dict[str, Any]]:
        self._check_open()

        actions = th.as_tensor(actions, dtype=th.int32, device=self.device).contiguous()
        if actions.shape != (self.n_envs, 7):
            raise ValueError(f"Expected actions shaped {(self.n_envs, 7)}")
        actions = actions.view(self.n_sim, self.n_cars, 7)

        self._sync()
        self._env.step(actions)
        self._sync()

        score_delta = self._tensor(self._env.get_rewards())
        sim_done = self._tensor(self._env.get_dones())
        self._score_difference = self._tensor(self._env.get_transition_score_difference())
        self._episode_ticks = self._tensor(self._env.get_transition_episode_ticks())
        self._overtime = self._tensor(self._env.get_transition_overtime())

        sim_terminated = sim_done & score_delta.ne(0)
        sim_truncated = sim_done & ~sim_terminated

        if self.reward_funcs:
            reward, reward_info = self._custom_reward(
                actions, score_delta, sim_done, sim_terminated, sim_truncated
            )
        else:
            reward = (score_delta[:, None] * self._team_sign).reshape(self.n_envs)
            reward_info = {}

        self._apply_reset_state(sim_done)
        self._clear_sim_stats(sim_done)
        obs = self._observe()

        terminated = self._per_car(sim_terminated)
        truncated = self._per_car(sim_truncated)
        done = terminated | truncated

        self._episode_return += reward
        self._episode_length += 1
        finished = done.nonzero(as_tuple=True)[0]
        info: dict[str, Any] = {"reward": [], "length": []}

        if finished.numel():
            info = {
                "reward":     self._episode_return[finished].cpu().tolist(),
                "length":     self._episode_length[finished].cpu().tolist(),
                "final_obs":  CARLObservation.from_tensor(
                    self._tensor(self._env.get_transition_obs()).view(
                        self.n_envs, self._env.obs_dim
                    ),
                    self.n_cars,
                ),
                "_final_obs": done,
                **reward_info,
            }
            self._episode_return[finished] = 0
            self._episode_length[finished] = 0

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self, **_: Any) -> None:
        if not self.closed:
            self._sync()
            del self._env
            self.closed = True
