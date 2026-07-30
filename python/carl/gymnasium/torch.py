from collections.abc import Callable, Iterable, Mapping
import math
from typing import Any

import numpy as np
import torch as th
import gymnasium as gym

from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

import carl

from .action import CARLActionCodec
from .state import (
    BOOST_PAD_POSITIONS,
    REGULATION_TICKS,
    CarlEvents,
    CarlState,
    RewardContext,
    RewardResult,
)


RewardFunction = Callable[[RewardContext], th.Tensor | RewardResult]
ResetState = Mapping[str, th.Tensor]
ResetStateProvider = Callable[[th.Tensor], ResetState | None]

_RESET_STATE_FIELDS = (
    "ball_position",
    "ball_velocity",
    "ball_angular_velocity",
    "car_position",
    "car_rotation",
    "car_velocity",
    "car_angular_velocity",
    "car_demoed",
)


class CARLTorchVectorEnv(VectorEnv):
    metadata = {
        "render_modes":    [],
        "autoreset_mode": AutoresetMode.SAME_STEP,
    }

    render_mode = spec = None

    def __init__(
        self,
        n_sim:                int,
        n_blue:               int,
        n_orange:             int,
        seed:                 int = 0,
        frameskip:            int = 8,
        max_ticks:            int = REGULATION_TICKS,
        *,
        no_touch_timeout_ticks:   int | None = None,
        no_touch_timeout_seconds: float | None = None,
        invert_orange:        bool = True,
        normalize:            bool = False,
        copy_outputs:         bool = True,
        synchronize:          bool = False,
        reward_funcs:         Iterable[RewardFunction] | None = None,
        reward_scale:         float = 1.0,
        reset_state_provider: ResetStateProvider | None = None,
    ) -> None:
        super().__init__()

        self.n_sim = n_sim
        self.device = th.device("cuda:0")
        self.closed = False

        self._seed = seed
        self._copy_outputs = copy_outputs
        self._synchronize = synchronize
        if max_ticks < 1:
            raise ValueError("max_ticks must be positive")
        if (
            no_touch_timeout_ticks is not None
            and no_touch_timeout_seconds is not None
        ):
            raise ValueError(
                "Specify no_touch_timeout_ticks or no_touch_timeout_seconds, not both"
            )
        if no_touch_timeout_ticks is not None and no_touch_timeout_ticks < 1:
            raise ValueError("no_touch_timeout_ticks must be positive")
        if no_touch_timeout_seconds is not None:
            if (
                not math.isfinite(no_touch_timeout_seconds)
                or no_touch_timeout_seconds <= 0
            ):
                raise ValueError("no_touch_timeout_seconds must be positive and finite")
            no_touch_timeout_ticks = math.ceil(no_touch_timeout_seconds * 120)
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if reset_state_provider is not None and not callable(reset_state_provider):
            raise TypeError("reset_state_provider must be callable")
        self.reward_funcs = list(reward_funcs or ())
        self.reward_scale = reward_scale
        self.reset_state_provider = reset_state_provider
        self._state: CarlState | None = None

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
        self._env.no_touch_timeout_ticks = no_touch_timeout_ticks or 0

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
        self._score_difference = th.zeros(self.n_sim, dtype=th.int32, device=self.device)
        self._episode_ticks = th.zeros(self.n_sim, dtype=th.int32, device=self.device)
        self._overtime = th.zeros(self.n_sim, dtype=th.bool, device=self.device)
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

    def _refresh_state(self) -> th.Tensor:
        self._sync()
        observation = self._from_carl(self._env.get_obs())
        self._state = (
            self._state_from_carl(self._env.get_state())
            if self.reward_funcs
            else None
        )
        return observation.view(self.n_envs, self._env.obs_dim)

    def _apply_reset_states(self, reset_mask: th.Tensor) -> None:
        if self.reset_state_provider is None:
            return

        simulation_indices = reset_mask.nonzero(as_tuple=True)[0]
        if not simulation_indices.numel():
            return

        state = self.reset_state_provider(reset_mask)
        if state is None:
            return
        if not isinstance(state, Mapping):
            raise TypeError("reset_state_provider must return a mapping or None")

        missing = [field for field in _RESET_STATE_FIELDS if field not in state]
        if missing:
            raise KeyError(f"reset state is missing fields: {', '.join(missing)}")

        selected_indices = state.get("simulation_indices")
        if selected_indices is not None:
            if not isinstance(selected_indices, th.Tensor):
                raise TypeError("simulation_indices must be a tensor")
            if selected_indices.dtype != th.int64 or selected_indices.ndim != 1:
                raise TypeError("simulation_indices must be one-dimensional int64")
            if selected_indices.device != reset_mask.device:
                raise ValueError("simulation_indices must be on the reset mask device")
            if not selected_indices.numel():
                return
            if (
                selected_indices.lt(0).any()
                or selected_indices.ge(self.n_sim).any()
                or not reset_mask[selected_indices].all()
            ):
                raise ValueError("simulation_indices must select reset simulations")
            if selected_indices.unique().numel() != selected_indices.numel():
                raise ValueError("simulation_indices must be unique")
            simulation_indices = selected_indices.contiguous()

        self._env.set_ball(
            state["ball_position"],
            state["ball_velocity"],
            state["ball_angular_velocity"],
            simulation_indices=simulation_indices,
        )
        self._env.set_car(
            state["car_position"],
            state["car_rotation"],
            state["car_velocity"],
            state["car_angular_velocity"],
            state["car_demoed"],
            boost=state.get("car_boost"),
            simulation_indices=simulation_indices,
        )
        match_fields = ("blue_score", "orange_score", "episode_ticks")
        if any(field in state for field in match_fields):
            missing_match = [field for field in match_fields if field not in state]
            if missing_match:
                raise KeyError(
                    f"reset state is missing fields: {', '.join(missing_match)}"
                )
            self._env.set_match_state(
                state["blue_score"].contiguous(),
                state["orange_score"].contiguous(),
                state["episode_ticks"].contiguous(),
                simulation_indices=simulation_indices,
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
        actions:     th.Tensor,
        score_delta: th.Tensor,
        done:        th.Tensor,
        terminated:  th.Tensor,
        truncated:   th.Tensor,
    ) -> tuple[th.Tensor, dict[str, list[Any]]]:
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
            score_difference=self._score_difference,
            episode_ticks=self._episode_ticks,
            overtime=self._overtime,
        )

        reward = th.zeros(
            (self.n_sim, self.n_cars),
            dtype=th.float32,
            device=self.device,
        )
        info = {}

        for reward_func in self.reward_funcs:
            component = reward_func(context)
            if isinstance(component, RewardResult):
                expected = int(done.sum().item()) * self.n_cars
                for key, values in component.info.items():
                    if key in ("reward", "length"):
                        raise ValueError(f"reward diagnostic uses reserved key: {key}")
                    if key in info:
                        raise ValueError(f"duplicate reward diagnostic key: {key}")
                    if not isinstance(values, list):
                        raise TypeError(f"reward diagnostic {key} must be a list")
                    if len(values) != expected:
                        raise ValueError(
                            f"reward diagnostic {key} returned {len(values)} values; "
                            f"expected {expected}"
                        )
                    info[key] = values
                component = component.reward
            if component.shape != reward.shape:
                raise ValueError(
                    f"Reward returned {tuple(component.shape)}; "
                    f"expected {tuple(reward.shape)}"
                )
            reward += component

        return reward * self.reward_scale, info

    def reset(
        self,
        *,
        seed:    int | None = None,
        options: dict[str, Any] | None = None,
    ) -> th.Tensor:
        if self.closed:
            raise RuntimeError("Environment is closed")

        if seed is not None and seed != self._seed:
            raise ValueError("CARL's seed is constructor-only")

        if options:
            raise NotImplementedError("CARL reset options are not supported")

        self._env.reset()
        reset_mask = th.ones(self.n_sim, dtype=th.bool, device=self.device)
        self._apply_reset_states(reset_mask)
        self._episode_return.zero_()
        self._episode_length.zero_()
        self._score_difference.zero_()
        self._episode_ticks.zero_()
        self._overtime.zero_()
        return self._refresh_state()

    def set_ball(
        self,
        position:           th.Tensor,
        velocity:           th.Tensor,
        angular_velocity:   th.Tensor,
        *,
        simulation_indices: th.Tensor | None = None,
    ) -> th.Tensor:
        if self.closed:
            raise RuntimeError("Environment is closed")

        self._sync()
        self._env.set_ball(
            position,
            velocity,
            angular_velocity,
            simulation_indices=simulation_indices,
        )
        return self._refresh_state()

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
        if self.closed:
            raise RuntimeError("Environment is closed")

        self._sync()
        self._env.set_car(
            position,
            rotation,
            velocity,
            angular_velocity,
            demoed,
            boost=boost,
            simulation_indices=simulation_indices,
        )
        return self._refresh_state()

    def step(
        self, actions: th.Tensor | np.ndarray
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, dict[str, Any]]:
        if self.closed:
            raise RuntimeError("Environment is closed")

        act = self._prepare_actions(actions)

        self._sync()

        self._env.step(act)

        self._sync()

        score_delta = self._from_carl(self._env.get_rewards())
        don = self._from_carl(self._env.get_dones())
        self._score_difference = self._from_carl(
            self._env.get_transition_score_difference()
        )
        self._episode_ticks = self._from_carl(
            self._env.get_transition_episode_ticks()
        )
        self._overtime = self._from_carl(self._env.get_transition_overtime())

        terms = don & score_delta.ne(0)
        trunc = don & ~terms

        if self.reward_funcs:
            rew, reward_info = self._custom_reward(
                act, score_delta, don, terms, trunc
            )
            rew = rew.reshape(self.n_envs)
        else:
            rew = (score_delta[:, None] * self._team_sign).reshape(self.n_envs)
            reward_info = {}

        self._apply_reset_states(don)
        self._score_difference[don] = 0
        self._episode_ticks[don] = 0
        self._overtime[don] = False
        obs = self._refresh_state()

        terms = terms[:, None].expand(-1, self.n_cars).reshape(self.n_envs)
        trunc = trunc[:, None].expand(-1, self.n_cars).reshape(self.n_envs)
        done = terms | trunc

        self._episode_return += rew
        self._episode_length += self._env.frameskip
        finished = done.nonzero(as_tuple=True)[0]
        info = {"reward": [], "length": []}
        if finished.numel():
            final_obs = self._from_carl(self._env.get_transition_obs()).view(
                self.n_envs, self._env.obs_dim
            )
            info = {
                "reward": self._episode_return[finished].cpu().tolist(),
                "length": self._episode_length[finished].cpu().tolist(),
                "final_obs": final_obs,
                "_final_obs": done,
                **reward_info,
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
