"""Left/right reflection of training definitions, not event matching behavior."""

from copy import deepcopy
from dataclasses import replace
import math

from ..events import (
    ArenaSurface, BallAccelerates, BallTouch, Condition, ContactSurface, Demoed,
    Dodge, DoubleJump, FlipReset, Goal, Jump, LeaveSurface, StateCondition,
    SurfaceContact,
)
from .definitions import Exercise, Stage
from .initial import Box2D, Box3D, Euler, InAir, OnSurface, Scalar, Uniform, Vector2, Vector3


def _negative(value: Scalar) -> Scalar:
    if isinstance(value, Uniform):
        return Uniform(-value.high, -value.low)
    return -value


def _reflected_heading(value: Scalar) -> Scalar:
    if isinstance(value, Uniform):
        return Uniform(math.pi - value.high, math.pi - value.low)
    return math.pi - value


def _vector(
    value: Vector2 | Vector3,
    signs: tuple[int, ...],
) -> Vector2 | Vector3:
    if isinstance(value, (Box2D, Box3D)):
        low = tuple(low if sign == 1 else -high for low, high, sign in zip(value.low, value.high, signs))
        high = tuple(high if sign == 1 else -low for low, high, sign in zip(value.low, value.high, signs))
        return type(value)(low, high)

    return tuple(deepcopy(component) if sign == 1 else _negative(component)
                 for component, sign in zip(value, signs))


def _placement(placement: OnSurface | InAir) -> OnSurface | InAir:
    if isinstance(placement, InAir):
        orientation = placement.orientation

        return InAir(
            position=_vector(placement.position, (-1, 1, 1)),
            orientation=Euler(
                roll=_negative(orientation.roll),
                pitch=deepcopy(orientation.pitch),
                yaw=_reflected_heading(orientation.yaw),
            ),
        )

    if not isinstance(placement, OnSurface):
        raise ValueError(f"unsupported mirrored placement: {type(placement).__name__}")
    if placement.surface == ArenaSurface.SIDE_WALL:
        # Side-wall coordinates are (y, z); switching walls reverses the heading basis.
        return replace(placement, position=deepcopy(placement.position),
                       heading=_negative(placement.heading), side=-placement.side)
    if placement.surface not in (ArenaSurface.FLOOR, ArenaSurface.CEILING, ArenaSurface.BACK_WALL):
        raise ValueError("unsupported mirrored arena surface")

    # Floor/ceiling coordinates are (x, y); back-wall coordinates are (x, z).
    return replace(placement, position=_vector(placement.position, (-1, 1)),
                   heading=_reflected_heading(placement.heading))


def _condition(condition: Condition) -> Condition:
    if isinstance(condition, StateCondition):
        if condition.metric in ("car_x", "ball_x"):
            return replace(condition,
                           minimum=-condition.maximum if condition.maximum is not None else None,
                           maximum=-condition.minimum if condition.minimum is not None else None)
        return deepcopy(condition)
    if isinstance(condition, BallTouch):
        parts = {ContactSurface.LEFT: ContactSurface.RIGHT, ContactSurface.RIGHT: ContactSurface.LEFT}
        if isinstance(condition.surface, tuple):
            return replace(condition, surface=tuple(parts.get(part, part) for part in condition.surface))
        return replace(condition, surface=parts.get(condition.surface, condition.surface))
    if isinstance(condition, (Jump, DoubleJump, Dodge, FlipReset, Demoed, Goal,
                              SurfaceContact, LeaveSurface, BallAccelerates)):
        return deepcopy(condition)

    raise ValueError(f"unsupported mirrored condition: {type(condition).__name__}")


def _stage(stage: Stage) -> Stage:
    return replace(
        stage,
        event=_condition(stage.event),
        fail_on=[_condition(condition) for condition in stage.fail_on],
        repeat_on=_condition(stage.repeat_on) if stage.repeat_on is not None else None,
    )


def mirror_exercise(
    exercise: Exercise,
    *,
    name:     str | None = None,
) -> Exercise:
    """Reflect polar vectors with (-1,1,1), axial spin with (1,-1,-1).

    Y and the scoring team do not change. State-condition X intervals reverse
    their bounds, LEFT/RIGHT car parts swap, and local surface headings follow
    the reset provider's basis. The source and its mutable distributions are
    never modified or shared with the returned definition.
    """
    initial = deepcopy(exercise.initial)
    initial.ego.placement = _placement(initial.ego.placement)
    initial.ego.velocity = _vector(initial.ego.velocity, (-1, 1, 1))
    initial.ego.angular_velocity = _vector(initial.ego.angular_velocity, (1, -1, -1))
    initial.ball.position = _vector(initial.ball.position, (-1, 1, 1))
    initial.ball.velocity = _vector(initial.ball.velocity, (-1, 1, 1))
    initial.ball.angular_velocity = _vector(initial.ball.angular_velocity, (1, -1, -1))

    return replace(
        exercise,
        name=f"{exercise.name} (mirrored)" if name is None else name,
        initial=initial,
        stages=[_stage(stage) for stage in exercise.stages],
        fail_on=[_condition(condition) for condition in exercise.fail_on],
    )
