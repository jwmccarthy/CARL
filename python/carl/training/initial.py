from dataclasses import dataclass, field
from typing import Union

from ..events import ArenaSurface


@dataclass
class Uniform:
    low:  float
    high: float

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError("low must not exceed high")


Scalar = Union[float, Uniform]


@dataclass
class Box2D:
    low:  tuple[float, float]
    high: tuple[float, float]


@dataclass
class Box3D:
    low:  tuple[float, float, float]
    high: tuple[float, float, float]


Vector2 = Union[tuple[Scalar, Scalar], Box2D]
Vector3 = Union[tuple[Scalar, Scalar, Scalar], Box3D]


@dataclass
class Euler:
    roll:  Scalar = 0.0
    pitch: Scalar = 0.0
    yaw:   Scalar = 0.0


@dataclass
class OnSurface:
    surface:  ArenaSurface = ArenaSurface.FLOOR
    position: Vector2      = (0.0, 0.0)
    heading:  Scalar       = 0.0
    side:     int          = 1

    def __post_init__(self):
        if self.side not in (-1, 1):
            raise ValueError("side must be -1 or 1")


@dataclass
class InAir:
    position:    Vector3
    orientation: Euler = field(default_factory=Euler)


@dataclass
class CarState:
    placement:        Union[OnSurface, InAir]
    velocity:         Vector3 = (0.0, 0.0, 0.0)
    angular_velocity: Vector3 = (0.0, 0.0, 0.0)
    boost:            Scalar  = 100.0


@dataclass
class BallState:
    position:         Vector3
    velocity:         Vector3 = (0.0, 0.0, 0.0)
    angular_velocity: Vector3 = (0.0, 0.0, 0.0)


@dataclass
class InitialState:
    ego:  CarState
    ball: BallState
