from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import torch as th

Reward = Callable[["RewardContext"], th.Tensor]


@dataclass(frozen=True)
class RewardContext:
    ball_position:      th.Tensor
    ball_velocity:      th.Tensor
    car_position:       th.Tensor
    car_velocity:       th.Tensor
    car_forward:        th.Tensor
    boost:              th.Tensor
    demoed:             th.Tensor
    previous_boost:     th.Tensor
    previous_demoed:    th.Tensor
    prev_ball_distance: th.Tensor
    touch:              th.Tensor
    touch_height:       th.Tensor
    last_toucher:       th.Tensor
    score_delta:        th.Tensor
    done:               th.Tensor
    team_sign:          th.Tensor
    n_blue:             int

    @property
    def n_cars(self) -> int:
        return self.car_position.shape[1]

    @property
    def n_orange(self) -> int:
        return self.n_cars - self.n_blue

    @property
    def car_to_ball(self) -> th.Tensor:
        return self.ball_position - self.car_position


class RewardRegistry(Mapping[str, Reward]):
    """Registry of reward terms computed from a shared CARL step context."""

    def __init__(self) -> None:
        self._rewards: dict[str, Reward] = {}

    def __getitem__(self, name: str) -> Reward:
        return self._rewards[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rewards)

    def __len__(self) -> int:
        return len(self._rewards)

    def register(self, name: str) -> Callable[[Reward], Reward]:
        def decorator(reward: Reward) -> Reward:
            if name in self._rewards:
                raise ValueError(f"Reward {name!r} is already registered")
            self._rewards[name] = reward
            return reward

        return decorator


class TorchReward:
    """Compose registered per-car reward terms while retaining step state."""

    def __init__(
        self,
        n_blue:        int,
        n_orange:      int,
        weights:       Mapping[str, float],
        *,
        scale:         float = 1.0,
        invert_orange: bool = True,
        registry:      Mapping[str, Reward] | None = None,
    ) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")

        registry = REWARDS if registry is None else registry

        unknown = set(weights) - set(registry)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise KeyError(f"Unknown reward terms: {names}")

        self.n_blue = n_blue
        self.n_orange = n_orange
        self.n_cars = n_blue + n_orange
        self.weights = dict(weights)
        self.scale = scale
        self.invert_orange = invert_orange
        self.registry = registry
        self._initialized = False

    def _state(self, observation: th.Tensor) -> tuple[th.Tensor, ...]:
        if observation.ndim != 3 or observation.shape[1] != self.n_cars:
            raise ValueError(
                "Expected observations shaped "
                f"[n_sim, {self.n_cars}, obs_dim], got {tuple(observation.shape)}"
            )
        if observation.shape[2] != 9 + 21 * self.n_cars:
            raise ValueError("Observation has an unexpected feature dimension")

        view = observation

        ball_position = view[..., :3].clone()
        ball_velocity = view[..., 3:6].clone()

        self_car = view[..., 9:30]
        car_position = self_car[..., :3].clone()
        car_velocity = self_car[..., 3:6].clone()
        car_forward = self_car[..., 9:12].clone()

        boost = self_car[..., 15].clamp(0.0, 100.0)
        demoed = self_car[..., 17].bool()

        if not self.invert_orange:
            orange = slice(self.n_blue, None)

            ball_position[:, orange, :2].neg_()
            ball_velocity[:, orange, :2].neg_()

            car_position[:, orange, :2].neg_()
            car_velocity[:, orange, :2].neg_()
            car_forward[:, orange, :2].neg_()

        return (
            ball_position,
            ball_velocity,
            car_position,
            car_velocity,
            car_forward,
            boost,
            demoed,
        )

    def reset(self, observation: th.Tensor) -> None:
        state = self._state(observation)
        ball_position, _, car_position, _, _, boost, demoed = state
        shape = boost.shape

        self.previous_boost = boost.clone()
        self.previous_demoed = demoed.clone()
        self.prev_ball_distance = (ball_position - car_position).norm(dim=-1)

        self.last_toucher = th.full(
            (shape[0],), -1, dtype=th.long, device=observation.device
        )

        self._initialized = True

    def step(
        self,
        observation: th.Tensor,
        score_delta: th.Tensor,
        touch:       th.Tensor,
        done:        th.Tensor,
    ) -> tuple[th.Tensor, dict[str, th.Tensor]]:
        if not self._initialized:
            raise RuntimeError("TorchReward.reset must be called before step")

        state = self._state(observation)
        ball_position, ball_velocity, car_position, car_velocity = state[:4]
        car_forward, boost, demoed = state[4:]

        touch = touch.bool()
        done = done.bool()

        toucher = th.where(
            touch,
            th.arange(self.n_cars, device=touch.device)[None, :],
            -1,
        ).amax(dim=1)
        self.last_toucher.copy_(th.where(toucher >= 0, toucher, self.last_toucher))

        team_sign = th.cat(
            (
                th.ones(self.n_blue, device=observation.device),
                -th.ones(self.n_orange, device=observation.device),
            )
        )
        context = RewardContext(
            ball_position=ball_position,
            ball_velocity=ball_velocity,
            car_position=car_position,
            car_velocity=car_velocity,
            car_forward=car_forward,
            boost=boost,
            demoed=demoed,
            previous_boost=self.previous_boost,
            previous_demoed=self.previous_demoed,
            prev_ball_distance=self.prev_ball_distance,
            touch=touch,
            touch_height=ball_position[..., 2],
            last_toucher=self.last_toucher,
            score_delta=score_delta,
            done=done,
            team_sign=team_sign,
            n_blue=self.n_blue,
        )

        components = {name: self.registry[name](context) for name in self.weights}
        individual = th.zeros_like(boost)

        for name, weight in self.weights.items():
            individual += weight * components[name]

        reward = individual * self.scale

        self.previous_boost.copy_(boost)
        self.previous_demoed.copy_(demoed)
        self.prev_ball_distance.copy_((ball_position - car_position).norm(dim=-1))

        self.last_toucher[done] = -1

        return reward, components


REWARDS = RewardRegistry()

__all__ = [
    "REWARDS",
    "Reward",
    "RewardContext",
    "RewardRegistry",
    "TorchReward",
]
