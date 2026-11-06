from dataclasses import dataclass
from enum import IntEnum

import torch as th

from .runtime import (
    BALL_CONTACT, BALL_GOAL, BALL_LEAVE, CAR_CONTACT, CAR_EVENT, CAR_LEAVE,
    STATE_METRICS, STATE_VALUE, TOUCH_GAIN, TOUCH_SURFACE,
)


class TargetKind(IntEnum):

    PADDING       = 0
    CAR_EVENT     = CAR_EVENT
    BALL_GOAL     = BALL_GOAL
    CAR_CONTACT   = CAR_CONTACT
    CAR_LEAVE     = CAR_LEAVE
    BALL_CONTACT  = BALL_CONTACT
    BALL_LEAVE    = BALL_LEAVE
    TOUCH_SURFACE = TOUCH_SURFACE
    STATE_VALUE   = STATE_VALUE
    TOUCH_GAIN    = TOUCH_GAIN


class TargetEntity(IntEnum):

    NONE = 0
    CAR  = 1
    BALL = 2


class StateMetric(IntEnum):

    BALL_X      = STATE_METRICS["ball_x"]
    BALL_Y      = STATE_METRICS["ball_y"]
    BALL_HEIGHT = STATE_METRICS["ball_height"]
    BALL_SPEED  = STATE_METRICS["ball_speed"]
    CAR_X       = STATE_METRICS["car_x"]
    CAR_Y       = STATE_METRICS["car_y"]
    CAR_HEIGHT  = STATE_METRICS["car_height"]
    CAR_SPEED   = STATE_METRICS["car_speed"]
    CAR_BOOST   = STATE_METRICS["car_boost"]


@dataclass(frozen=True)
class TargetConditions:

    valid:              th.Tensor
    kind:               th.Tensor
    entity:             th.Tensor
    event_mask:         th.Tensor
    arena_surface_mask: th.Tensor
    touch_surface_mask: th.Tensor
    state_metric:       th.Tensor
    has_state_metric:   th.Tensor
    minimum:            th.Tensor
    has_minimum:        th.Tensor
    maximum:            th.Tensor
    has_maximum:        th.Tensor
    goal_side:          th.Tensor
    min_touch_speed_gain:     th.Tensor
    has_min_touch_speed_gain: th.Tensor
    max_touch_speed_gain:     th.Tensor
    has_max_touch_speed_gain: th.Tensor


@dataclass(frozen=True)
class TrainingTarget:

    stages:                TargetConditions
    failures:              TargetConditions
    repeats:               TargetConditions
    max_repeats:           th.Tensor
    has_repeat_limit:      th.Tensor
    repeat_count:          th.Tensor
    max_total_matches:     th.Tensor
    has_total_match_limit: th.Tensor
    total_matches:         th.Tensor
    failure_exercise_wide: th.Tensor
    deadline_ticks:        th.Tensor
    min_ticks:             th.Tensor
    has_deadline:          th.Tensor
    deadline_from_start:   th.Tensor
    exercise_index:        th.Tensor
    stage_index:           th.Tensor
    ticks_remaining:       th.Tensor
    has_remaining:         th.Tensor
    active:                th.Tensor


def _conditions(kind, mask, parameter, minimum, maximum, valid):
    is_state = (kind == STATE_VALUE) & valid
    arena = ((kind == CAR_CONTACT) | (kind == CAR_LEAVE) |
             (kind == BALL_CONTACT) | (kind == BALL_LEAVE)) & valid
    touch = ((kind == TOUCH_SURFACE) | (kind == TOUCH_GAIN)) & valid
    touch_speed = (kind == TOUCH_GAIN) & valid
    event = ((kind == CAR_EVENT) | (kind == BALL_GOAL)) & valid

    car = ((kind == CAR_EVENT) | (kind == CAR_CONTACT) | (kind == CAR_LEAVE) |
           (kind == TOUCH_SURFACE) | (kind == TOUCH_GAIN) | (is_state & (mask >= StateMetric.CAR_X))) & valid
    ball = ((kind == BALL_GOAL) | (kind == BALL_CONTACT) | (kind == BALL_LEAVE) |
            (is_state & (mask < StateMetric.CAR_X))) & valid
    entity = th.where(car, int(TargetEntity.CAR), th.where(ball, int(TargetEntity.BALL), 0))

    has_minimum = is_state & th.isfinite(minimum)
    has_maximum = is_state & th.isfinite(maximum)

    return TargetConditions(
        valid              = valid,
        kind               = th.where(valid, kind, 0),
        entity             = entity,
        event_mask         = th.where(event, mask, 0),
        arena_surface_mask = th.where(arena, mask, 0),
        touch_surface_mask = th.where(touch, mask, 0),
        state_metric       = th.where(is_state, mask, 0),
        has_state_metric   = is_state,
        minimum            = th.where(has_minimum, minimum, 0),
        has_minimum        = has_minimum,
        maximum            = th.where(has_maximum, maximum, 0),
        has_maximum        = has_maximum,
        goal_side          = th.where((kind == BALL_GOAL) & valid, parameter, 0),
        min_touch_speed_gain     = th.where(touch_speed & th.isfinite(minimum), minimum, 0),
        has_min_touch_speed_gain = touch_speed & th.isfinite(minimum),
        max_touch_speed_gain     = th.where(touch_speed & th.isfinite(maximum), maximum, 0),
        has_max_touch_speed_gain = touch_speed & th.isfinite(maximum),
    )


def training_target(runtime):
    pack = runtime.pack
    index = runtime.exercise_index
    valid = pack.valid[index]
    failure_kind = pack.failure_kind[index]
    failure_valid = (failure_kind != 0) & valid.unsqueeze(-1)
    deadlines = pack.deadline[index]
    has_deadline = (deadlines >= 0) & valid
    has_remaining = (pack.deadline[index, runtime.stage_index] >= 0) & runtime.active

    stages = _conditions(
        pack.kind[index], pack.mask[index], pack.parameter[index],
        pack.minimum[index], pack.maximum[index], valid,
    )
    failures = _conditions(
        failure_kind, pack.failure_mask[index], pack.failure_parameter[index],
        pack.failure_minimum[index], pack.failure_maximum[index], failure_valid,
    )
    repeats = _conditions(
        pack.repeat_kind[index], pack.repeat_mask[index], pack.repeat_parameter[index],
        pack.repeat_minimum[index], pack.repeat_maximum[index], (pack.repeat_kind[index] != 0) & valid,
    )

    return TrainingTarget(
        stages                = stages,
        failures              = failures,
        repeats               = repeats,
        max_repeats           = pack.max_repeats[index].clamp_min(0),
        has_repeat_limit      = (pack.max_repeats[index] >= 0) & valid,
        repeat_count          = runtime.repeat_count.clone(),
        max_total_matches     = pack.max_total_matches[index].clamp_min(0),
        has_total_match_limit = (pack.max_total_matches[index] >= 0) & valid,
        total_matches         = runtime.total_matches.clone(),
        failure_exercise_wide = pack.failure_exercise_wide[index] & failure_valid,
        deadline_ticks        = th.where(has_deadline, deadlines, 0),
        min_ticks             = th.where(valid, pack.min_ticks[index], 0),
        has_deadline          = has_deadline,
        deadline_from_start   = pack.deadline_from_start[index] & has_deadline,
        exercise_index        = runtime.exercise_index.clone(),
        stage_index           = runtime.stage_index.clone(),
        ticks_remaining       = th.where(has_remaining, runtime.ticks_remaining, 0),
        has_remaining         = has_remaining,
        active                = runtime.active.clone(),
    )
