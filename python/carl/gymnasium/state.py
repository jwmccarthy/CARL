import torch as th
import carl

from typing import Any
from collections.abc import Mapping
from dataclasses import dataclass, field
from carl import BOOST_PAD_POSITIONS, REGULATION_TICKS


@dataclass(frozen=True)
class RewardResult:
    reward: th.Tensor
    info:   Mapping[str, list[Any]] = field(default_factory=dict)


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
        return self[..., carl.OBS_POS:carl.OBS_VEL]

    @property
    def velocity(self) -> th.Tensor:
        return self[..., carl.OBS_VEL:carl.OBS_ANG]

    @property
    def angular_velocity(self) -> th.Tensor:
        return self[..., carl.OBS_ANG:carl.OBS_BALL]


class CARLCar(_CARLTensor):

    @property
    def position(self) -> th.Tensor:
        return self[..., carl.OBS_POS:carl.OBS_VEL]

    @property
    def velocity(self) -> th.Tensor:
        return self[..., carl.OBS_VEL:carl.OBS_ANG]

    @property
    def angular_velocity(self) -> th.Tensor:
        return self[..., carl.OBS_ANG:carl.OBS_FORWARD]

    @property
    def forward(self) -> th.Tensor:
        return self[..., carl.OBS_FORWARD:carl.OBS_UP]

    @property
    def up(self) -> th.Tensor:
        return self[..., carl.OBS_UP:carl.OBS_BOOST]

    @property
    def boost(self) -> th.Tensor:
        return self[..., carl.OBS_BOOST]

    @property
    def on_ground(self) -> th.Tensor:
        return self[..., carl.OBS_ON_GROUND].bool()

    @property
    def demoed(self) -> th.Tensor:
        return self[..., carl.OBS_DEMOED].bool()

    @property
    def has_flipped(self) -> th.Tensor:
        return self[..., carl.OBS_HAS_FLIPPED].bool()

    @property
    def has_double_jumped(self) -> th.Tensor:
        return self[..., carl.OBS_HAS_DOUBLE_JUMPED].bool()

    @property
    def is_boosting(self) -> th.Tensor:
        return self[..., carl.OBS_IS_BOOSTING].bool()


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
        return self[..., carl.OBS_POS:carl.OBS_VEL]

    @property
    def velocity(self) -> th.Tensor:
        return self[..., carl.OBS_VEL:carl.OBS_ANG]

    @property
    def angular_velocity(self) -> th.Tensor:
        return self[..., carl.OBS_ANG:carl.OBS_FORWARD]

    @property
    def forward(self) -> th.Tensor:
        return self[..., carl.OBS_FORWARD:carl.OBS_UP]

    @property
    def up(self) -> th.Tensor:
        return self[..., carl.OBS_UP:carl.OBS_BOOST]

    @property
    def boost(self) -> th.Tensor:
        return self[..., carl.OBS_BOOST]

    @property
    def on_ground(self) -> th.Tensor:
        return self[..., carl.OBS_ON_GROUND].bool()

    @property
    def demoed(self) -> th.Tensor:
        return self[..., carl.OBS_DEMOED].bool()

    @property
    def has_flipped(self) -> th.Tensor:
        return self[..., carl.OBS_HAS_FLIPPED].bool()

    @property
    def has_double_jumped(self) -> th.Tensor:
        return self[..., carl.OBS_HAS_DOUBLE_JUMPED].bool()

    @property
    def is_boosting(self) -> th.Tensor:
        return self[..., carl.OBS_IS_BOOSTING].bool()

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
        return CARLBall.from_tensor(self[..., :carl.OBS_BALL])

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
        values = self[..., carl.OBS_BALL:self.car_end].view(
            *self.shape[:-1], self._n_cars, carl.OBS_PER_CAR
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
        end = self.car_end + carl.NUM_BOOST_PADS
        return self[..., self.car_end:end].bool()

    @property
    def boost_pad_distance(self) -> th.Tensor:
        start = self.car_end + carl.NUM_BOOST_PADS
        return self[..., start:start + carl.NUM_BOOST_PADS]

    @property
    def ego_ball_relative(self) -> th.Tensor:
        start = self.car_end + carl.OBS_BOOST_PADS
        return self[..., start:start + carl.OBS_RELATIVE_EGO_BALL]

    @property
    def ego_other_relative(self) -> th.Tensor:
        start = self.car_end + carl.OBS_BOOST_PADS + carl.OBS_RELATIVE_EGO_BALL
        end = start + carl.OBS_RELATIVE_PER_OTHER_CAR * (self._n_cars - 1)
        return self[..., start:end].view(
            *self.shape[:-1],
            self._n_cars - 1,
            carl.OBS_RELATIVE_PER_OTHER_CAR
        )

    @property
    def _goal_start(self) -> int:
        return (self.car_end + carl.OBS_BOOST_PADS + carl.OBS_RELATIVE_EGO_BALL
                + carl.OBS_RELATIVE_PER_OTHER_CAR * (self._n_cars - 1))

    @property
    def own_goal_relative(self) -> th.Tensor:
        start = self._goal_start
        return self[..., start:start + carl.OBS_VECTOR_SIZE]

    @property
    def opponent_goal_relative(self) -> th.Tensor:
        start = self._goal_start + carl.OBS_VECTOR_SIZE
        return self[..., start:start + carl.OBS_VECTOR_SIZE]

    @property
    def ego_has_flip_or_jump(self) -> th.Tensor:
        return self[..., carl.OBS_EGO_FLIP_INDEX].bool()

    @property
    def ego_flip_window_remaining(self) -> th.Tensor:
        return self[..., carl.OBS_EGO_DODGE_TIME_INDEX]

    @property
    def car_end(self) -> int:
        return carl.OBS_BALL + carl.OBS_PER_CAR * self._n_cars


@dataclass(frozen=True)
class CARLMatchReset:
    """Match state for the simulations selected by a reset provider."""

    blue_score:    th.Tensor
    orange_score:  th.Tensor
    episode_ticks: th.Tensor


@dataclass(frozen=True)
class CARLResetState:
    """Ball and cars use physics units unless ``normalized`` is true."""

    simulation_indices: th.Tensor
    ball:               CARLBall
    cars:               CARLCars
    car_internal_state: th.Tensor | None = None
    match:              CARLMatchReset | None = None
    normalized:         bool = False

    def validate(self, reset_mask: th.Tensor, n_cars: int) -> None:
        indices = self.simulation_indices
        if (indices.ndim != 1 or indices.dtype != th.int64
                or indices.device != reset_mask.device):
            raise ValueError("simulation_indices must be int64 on the environment device")
        if ((indices < 0) | (indices >= len(reset_mask))).any():
            raise ValueError("simulation_indices are out of range")
        if not reset_mask[indices].all() or th.unique(indices).numel() != len(indices):
            raise ValueError("simulation_indices must be unique and selected for reset")
        if (self.ball.shape != (len(indices), carl.OBS_BALL)
                or self.cars.shape != (len(indices), n_cars, carl.OBS_PER_CAR)):
            raise ValueError("ball or cars have the wrong shape")

    def physical(self) -> tuple[CARLBall, CARLCars]:
        if not self.normalized:
            return self.ball, self.cars

        ball = CARLBall.from_tensor(self.ball.clone())
        cars = CARLCars.from_tensor(self.cars.clone(), self.cars.n_cars)
        
        position_scale = ball.position.new_tensor(carl.OBS_POSITION_SCALE)

        ball.position.mul_(position_scale)
        ball.velocity.mul_(carl.BALL_MAX_SPEED)
        ball.angular_velocity.mul_(carl.BALL_MAX_ANG_SPEED)

        cars.position.mul_(position_scale)
        cars.velocity.mul_(carl.CAR_MAX_SPEED)
        cars.angular_velocity.mul_(carl.CAR_MAX_ANG_SPEED)
        cars.boost.mul_(carl.BOOST_MAX)

        return ball, cars


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
        car_end = carl.OBS_BALL + carl.STATE_PER_CAR * n_cars
        expected = car_end + carl.NUM_BOOST_PADS

        if raw.ndim != 2 or raw.shape[1] != expected:
            raise ValueError(
                f"Expected raw state shaped [n_sim, {expected}], "
                f"got {tuple(raw.shape)}"
            )

        return cls(raw, n_cars, boost_pad_positions, team_sign)

    @property
    def ball_values(self) -> th.Tensor:
        return self.raw[:, :carl.OBS_BALL]

    @property
    def ball_position(self) -> th.Tensor:
        return self.ball_values[..., carl.OBS_POS:carl.OBS_VEL]

    @property
    def ball_velocity(self) -> th.Tensor:
        return self.ball_values[..., carl.OBS_VEL:carl.OBS_ANG]

    @property
    def ball_angular_velocity(self) -> th.Tensor:
        return self.ball_values[..., carl.OBS_ANG:carl.OBS_BALL]

    @property
    def car_values(self) -> th.Tensor:
        car_end = carl.OBS_BALL + carl.STATE_PER_CAR * self.n_cars
        return self.raw[:, carl.OBS_BALL:car_end].view(
            self.raw.shape[0], self.n_cars, carl.STATE_PER_CAR
        )

    @property
    def car_position(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_POS:carl.OBS_VEL]

    @property
    def car_velocity(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_VEL:carl.OBS_ANG]

    @property
    def car_angular_velocity(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_ANG:carl.OBS_FORWARD]

    @property
    def car_forward(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_FORWARD:carl.OBS_UP]

    @property
    def car_up(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_UP:carl.OBS_BOOST]

    @property
    def car_boost(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_BOOST]

    @property
    def car_on_ground(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_ON_GROUND].bool()

    @property
    def car_demoed(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_DEMOED].bool()

    @property
    def car_has_flipped(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_HAS_FLIPPED].bool()

    @property
    def car_has_double_jumped(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_HAS_DOUBLE_JUMPED].bool()

    @property
    def car_is_boosting(self) -> th.Tensor:
        return self.car_values[..., carl.OBS_IS_BOOSTING].bool()

    @property
    def car_ball_touches(self) -> th.Tensor:
        return self.car_values[..., carl.STATE_BALL_TOUCH].bool()

    @property
    def boost_pad_values(self) -> th.Tensor:
        return self.raw[:, carl.OBS_BALL + carl.STATE_PER_CAR * self.n_cars:]

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
    "CARLMatchReset",
    "CARLObservation",
    "CARLResetState",
    "REGULATION_TICKS",
    "CarlEvents",
    "CarlState",
    "RewardContext",
]
