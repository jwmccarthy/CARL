from dataclasses import dataclass

from .enums import ArenaSurface, ContactSurface


class Condition:

    pass


@dataclass
class StateCondition(Condition):
    metric:  str
    minimum: float | None = None
    maximum: float | None = None


@dataclass
class BallTouch(Condition):
    surface:        ContactSurface | None = None
    min_height:     float          | None = None
    min_speed_gain: float          | None = None
    max_speed_gain: float          | None = None


@dataclass
class Jump(Condition):
    pass


@dataclass
class DoubleJump(Condition):
    pass


@dataclass
class Dodge(Condition):
    pass


@dataclass
class FlipReset(Condition):
    pass


@dataclass
class LeaveSurface(Condition):
    surface: ArenaSurface | None = None
    entity:  str = "car"


@dataclass
class SurfaceContact(Condition):
    surface: ArenaSurface | None = None
    entity:  str = "car"


@dataclass
class Demoed(Condition):
    pass


@dataclass
class BallAccelerates(Condition):
    by: float


@dataclass
class Goal(Condition):
    pass
