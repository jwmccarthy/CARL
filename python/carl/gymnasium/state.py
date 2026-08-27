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


class _CARLTensor(th.Tensor):

    @classmethod
    def from_tensor(cls, tensor: th.Tensor):
        return tensor.as_subclass(cls)

    @classmethod
    def __torch_function__(cls, function, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        with th._C.DisableTorchFunctionSubclass():
            return function(*args, **kwargs)


class CARLBall(_CARLTensor):

    @property
    def position(self) -> th.Tensor:
        return self[..., :3]

    @property
    def velocity(self) -> th.Tensor:
        return self[..., 3:6]

    @property
    def angular_velocity(self) -> th.Tensor:
        return self[..., 6:9]


class CARLCar(_CARLTensor):

    @property
    def position(self) -> th.Tensor:
        return self[..., :3]

    @property
    def velocity(self) -> th.Tensor:
        return self[..., 3:6]

    @property
    def angular_velocity(self) -> th.Tensor:
        return self[..., 6:9]

    @property
    def forward(self) -> th.Tensor:
        return self[..., 9:12]

    @property
    def up(self) -> th.Tensor:
        return self[..., 12:15]

    @property
    def boost(self) -> th.Tensor:
        return self[..., 15]

    @property
    def on_ground(self) -> th.Tensor:
        return self[..., 16].bool()

    @property
    def demoed(self) -> th.Tensor:
        return self[..., 17].bool()

    @property
    def has_flipped(self) -> th.Tensor:
        return self[..., 18].bool()

    @property
    def has_double_jumped(self) -> th.Tensor:
        return self[..., 19].bool()

    @property
    def is_boosting(self) -> th.Tensor:
        return self[..., 20].bool()


class CARLCars(_CARLTensor):

    @staticmethod
    def from_tensor(tensor: th.Tensor, n_cars: int) -> "CARLCars":
        cars = tensor.as_subclass(CARLCars)
        cars._n_cars = n_cars
        return cars

    @property
    def n_cars(self) -> int:
        return self._n_cars

    @property
    def ego(self) -> CARLCar:
        return CARLCar.from_tensor(self[..., 0, :])

    @property
    def position(self) -> th.Tensor:
        return self[..., :3]

    @property
    def velocity(self) -> th.Tensor:
        return self[..., 3:6]

    @property
    def angular_velocity(self) -> th.Tensor:
        return self[..., 6:9]

    @property
    def forward(self) -> th.Tensor:
        return self[..., 9:12]

    @property
    def up(self) -> th.Tensor:
        return self[..., 12:15]

    @property
    def boost(self) -> th.Tensor:
        return self[..., 15]

    @property
    def on_ground(self) -> th.Tensor:
        return self[..., 16].bool()

    @property
    def demoed(self) -> th.Tensor:
        return self[..., 17].bool()

    @property
    def has_flipped(self) -> th.Tensor:
        return self[..., 18].bool()

    @property
    def has_double_jumped(self) -> th.Tensor:
        return self[..., 19].bool()

    @property
    def is_boosting(self) -> th.Tensor:
        return self[..., 20].bool()

    @property
    def ego_position(self) -> th.Tensor:
        return self.ego.position

    @property
    def ego_velocity(self) -> th.Tensor:
        return self.ego.velocity

    @property
    def ego_angular_velocity(self) -> th.Tensor:
        return self.ego.angular_velocity

    @property
    def ego_forward(self) -> th.Tensor:
        return self.ego.forward

    @property
    def ego_up(self) -> th.Tensor:
        return self.ego.up

    @property
    def ego_boost(self) -> th.Tensor:
        return self.ego.boost


class CARLObservation(_CARLTensor):
    """A tensor observation with named views into CARL's packed layout."""

    _n_cars: int

    @staticmethod
    def from_tensor(tensor: th.Tensor, n_cars: int) -> "CARLObservation":
        observation = tensor.as_subclass(CARLObservation)
        observation._n_cars = n_cars
        return observation

    @property
    def ball(self) -> CARLBall:
        return CARLBall.from_tensor(self[..., :9])

    @property
    def n_cars(self) -> int:
        return self._n_cars

    @property
    def ball_values(self) -> CARLBall:
        return self.ball

    @property
    def ball_position(self) -> th.Tensor:
        return self.ball.position

    @property
    def ball_velocity(self) -> th.Tensor:
        return self.ball.velocity

    @property
    def ball_angular_velocity(self) -> th.Tensor:
        return self.ball.angular_velocity

    @property
    def cars(self) -> CARLCars:
        values = self[..., 9:self.car_end].view(
            *self.shape[:-1], self._n_cars, 21
        )
        return CARLCars.from_tensor(values, self._n_cars)

    @property
    def car_values(self) -> CARLCars:
        return self.cars

    @property
    def ego_values(self) -> th.Tensor:
        return self.cars.ego

    @property
    def car_position(self) -> th.Tensor:
        return self.cars.position

    @property
    def car_velocity(self) -> th.Tensor:
        return self.cars.velocity

    @property
    def car_angular_velocity(self) -> th.Tensor:
        return self.cars.angular_velocity

    @property
    def car_forward(self) -> th.Tensor:
        return self.cars.forward

    @property
    def car_up(self) -> th.Tensor:
        return self.cars.up

    @property
    def car_boost(self) -> th.Tensor:
        return self.cars.boost

    @property
    def car_on_ground(self) -> th.Tensor:
        return self.cars.on_ground

    @property
    def car_demoed(self) -> th.Tensor:
        return self.cars.demoed

    @property
    def car_has_flipped(self) -> th.Tensor:
        return self.cars.has_flipped

    @property
    def car_has_double_jumped(self) -> th.Tensor:
        return self.cars.has_double_jumped

    @property
    def car_is_boosting(self) -> th.Tensor:
        return self.cars.is_boosting

    @property
    def boost_pad_active(self) -> th.Tensor:
        end = self.car_end + len(BOOST_PAD_POSITIONS)
        return self[..., self.car_end:end].bool()

    @property
    def boost_pad_distance(self) -> th.Tensor:
        start = self.car_end + len(BOOST_PAD_POSITIONS)
        return self[..., start:start + len(BOOST_PAD_POSITIONS)]

    @property
    def ego_ball_relative(self) -> th.Tensor:
        start = self.car_end + 2 * len(BOOST_PAD_POSITIONS)
        return self[..., start:start + 6]

    @property
    def ego_other_relative(self) -> th.Tensor:
        start = self.car_end + 2 * len(BOOST_PAD_POSITIONS) + 6
        end = start + 6 * (self._n_cars - 1)
        return self[..., start:end].view(
            *self.shape[:-1],
            self._n_cars - 1,
            6
        )

    @property
    def own_goal_relative(self) -> th.Tensor:
        return self[..., -6:-3]

    @property
    def opponent_goal_relative(self) -> th.Tensor:
        return self[..., -3:]

    @property
    def car_end(self) -> int:
        return 9 + 21 * self._n_cars


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
    current:              CarlState
    previous:             CarlState
    current_observation:  CARLObservation
    previous_observation: CARLObservation
    events:               CarlEvents
    actions:              th.Tensor
    score_difference:     th.Tensor
    episode_ticks:        th.Tensor
    overtime:             th.Tensor


__all__ = [
    "BOOST_PAD_POSITIONS",
    "CARLBall",
    "CARLCar",
    "CARLCars",
    "CARLObservation",
    "REGULATION_TICKS",
    "CarlEvents",
    "CarlState",
    "RewardContext",
]
