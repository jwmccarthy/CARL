import torch as th

from typing import Any
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardResult:
    reward: th.Tensor
    info:   Mapping[str, list[Any]] = field(default_factory=dict)


BOOST_PAD_POSITIONS = (
    (-3584.0,     0.0, 73.0),
    ( 3584.0,     0.0, 73.0),
    (-3072.0,  4096.0, 73.0),
    ( 3072.0,  4096.0, 73.0),
    (-3072.0, -4096.0, 73.0),
    ( 3072.0, -4096.0, 73.0),
    (    0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    ( 1792.0, -4184.0, 70.0),
    ( -940.0, -3308.0, 70.0),
    (  940.0, -3308.0, 70.0),
    (    0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    ( 3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    ( 1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (    0.0, -1024.0, 70.0),
    ( 2048.0, -1036.0, 70.0),
    (-1024.0,     0.0, 70.0),
    ( 1024.0,     0.0, 70.0),
    (-2048.0,  1036.0, 70.0),
    (    0.0,  1024.0, 70.0),
    ( 2048.0,  1036.0, 70.0),
    (-1788.0,  2300.0, 70.0),
    ( 1788.0,  2300.0, 70.0),
    (-3584.0,  2484.0, 70.0),
    ( 3584.0,  2484.0, 70.0),
    (    0.0,  2816.0, 70.0),
    ( -940.0,  3308.0, 70.0),
    (  940.0,  3308.0, 70.0),
    (-1792.0,  4184.0, 70.0),
    ( 1792.0,  4184.0, 70.0),
    (    0.0,  4240.0, 70.0),
)
REGULATION_TICKS = 5 * 60 * 120


@dataclass(frozen=True)
class CarlState:
    raw:                 th.Tensor
    n_cars:              int
    boost_pad_positions: th.Tensor
    team_sign:           th.Tensor

    @classmethod
    def from_raw(
        cls,
        raw:                 th.Tensor,
        n_cars:              int,
        boost_pad_positions: th.Tensor,
        team_sign:           th.Tensor,
    ) -> "CarlState":
        car_end = 9 + 22 * n_cars
        expected = car_end + len(BOOST_PAD_POSITIONS)

        if raw.ndim != 2 or raw.shape[1] != expected:
            raise ValueError(
                f"Expected raw state shaped [n_sim, {expected}], "
                f"got {tuple(raw.shape)}"
            )

        return cls(raw, n_cars, boost_pad_positions, team_sign)

    @property
    def ball_values(self) -> th.Tensor:
        return self.raw[:, :9]

    @property
    def ball_position(self) -> th.Tensor:
        return self.ball_values[..., 0:3]

    @property
    def ball_velocity(self) -> th.Tensor:
        return self.ball_values[..., 3:6]

    @property
    def ball_angular_velocity(self) -> th.Tensor:
        return self.ball_values[..., 6:9]

    @property
    def car_values(self) -> th.Tensor:
        car_end = 9 + 22 * self.n_cars
        return self.raw[:, 9:car_end].view(self.raw.shape[0], self.n_cars, 22)

    @property
    def car_position(self) -> th.Tensor:
        return self.car_values[..., 0:3]

    @property
    def car_velocity(self) -> th.Tensor:
        return self.car_values[..., 3:6]

    @property
    def car_angular_velocity(self) -> th.Tensor:
        return self.car_values[..., 6:9]

    @property
    def car_forward(self) -> th.Tensor:
        return self.car_values[..., 9:12]

    @property
    def car_up(self) -> th.Tensor:
        return self.car_values[..., 12:15]

    @property
    def car_boost(self) -> th.Tensor:
        return self.car_values[..., 15]

    @property
    def car_on_ground(self) -> th.Tensor:
        return self.car_values[..., 16].bool()

    @property
    def car_demoed(self) -> th.Tensor:
        return self.car_values[..., 17].bool()

    @property
    def car_has_flipped(self) -> th.Tensor:
        return self.car_values[..., 18].bool()

    @property
    def car_has_double_jumped(self) -> th.Tensor:
        return self.car_values[..., 19].bool()

    @property
    def car_is_boosting(self) -> th.Tensor:
        return self.car_values[..., 20].bool()

    @property
    def car_ball_touches(self) -> th.Tensor:
        return self.car_values[..., 21].bool()

    @property
    def boost_pad_values(self) -> th.Tensor:
        return self.raw[:, 9 + 22 * self.n_cars:]

    @property
    def boost_pad_active(self) -> th.Tensor:
        return self.boost_pad_values.bool()


@dataclass(frozen=True)
class CarlEvents:
    score_delta: th.Tensor  # [n_sim]
    done:        th.Tensor  # [n_sim]
    terminated:  th.Tensor  # [n_sim]
    truncated:   th.Tensor  # [n_sim]


@dataclass(frozen=True)
class RewardContext:
    current:          CarlState
    previous:         CarlState
    events:           CarlEvents
    actions:          th.Tensor
    score_difference: th.Tensor
    episode_ticks:    th.Tensor
    overtime:         th.Tensor


__all__ = [
    "BOOST_PAD_POSITIONS",
    "REGULATION_TICKS",
    "CarlEvents",
    "CarlState",
    "RewardContext",
]
