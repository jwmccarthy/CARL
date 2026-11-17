from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch as th

import carl

from ..events import ArenaSurface, ContactSurface
from .runtime import (
    BALL, CAR, CONTACT, EVENT, EVENT_FLAGS, GOAL, LEAVE, STATE, STATE_METRICS,
    TOUCH, CompiledTrainingPack, ConditionColumns, TrainingPackRuntime,
)


class TargetKind(IntEnum):

    PADDING = 0
    EVENT   = EVENT
    GOAL    = GOAL
    CONTACT = CONTACT
    LEAVE   = LEAVE
    TOUCH   = TOUCH
    STATE   = STATE


class TargetEntity(IntEnum):

    NONE = 0
    CAR  = CAR
    BALL = BALL


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
class TargetConditions(ConditionColumns):
    valid: th.Tensor

    @property
    def has_state_metric(self) -> th.Tensor:
        return (self.kind == STATE) & self.valid


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


def _conditions(
    pack:   CompiledTrainingPack,
    index:  th.Tensor,
    prefix: str,
    valid:  th.Tensor,
) -> TargetConditions:
    return TargetConditions(**vars(pack.conditions(index, prefix)), valid=valid)


def training_target(runtime: TrainingPackRuntime) -> TrainingTarget:
    pack = runtime.pack
    index = runtime.exercise_index
    valid = pack.valid[index]
    failure_valid = (pack.failure_kind[index] != 0) & valid.unsqueeze(-1)
    has_deadline = pack.has_deadline[index] & valid
    has_remaining = pack.has_deadline[index, runtime.stage_index] & runtime.active

    return TrainingTarget(
        stages                = _conditions(pack, index, "", valid),
        failures              = _conditions(pack, index, "failure_", failure_valid),
        repeats               = _conditions(pack, index, "repeat_", (pack.repeat_kind[index] != 0) & valid),
        max_repeats           = pack.max_repeats[index],
        has_repeat_limit      = pack.has_max_repeats[index] & valid,
        repeat_count          = runtime.repeat_count.clone(),
        max_total_matches     = pack.max_total_matches[index],
        has_total_match_limit = pack.has_max_total_matches[index] & valid,
        total_matches         = runtime.total_matches.clone(),
        failure_exercise_wide = pack.failure_exercise_wide[index] & failure_valid,
        deadline_ticks        = pack.deadline[index],
        min_ticks             = pack.min_ticks[index],
        has_deadline          = has_deadline,
        deadline_from_start   = pack.deadline_from_start[index] & has_deadline,
        exercise_index        = runtime.exercise_index.clone(),
        stage_index           = runtime.stage_index.clone(),
        ticks_remaining       = th.where(has_remaining, runtime.ticks_remaining, 0),
        has_remaining         = has_remaining,
        active                = runtime.active.clone(),
    )


# Use the same physical-unit scales as gymnasium/state.py and RLConstants.cuh.
# Signed positions/gains retain their sign; no learned or pack-specific scales.
FEATURE_SCALES = {
    "ball_x": carl.OBS_POSITION_SCALE[0],
    "ball_y": carl.OBS_POSITION_SCALE[1],
    "ball_height": carl.OBS_POSITION_SCALE[2],
    "ball_speed": carl.BALL_MAX_SPEED,
    "car_x": carl.OBS_POSITION_SCALE[0],
    "car_y": carl.OBS_POSITION_SCALE[1],
    "car_height": carl.OBS_POSITION_SCALE[2],
    "car_speed": carl.CAR_MAX_SPEED,
    "car_boost": carl.BOOST_MAX,
    "gain": carl.BALL_MAX_SPEED,
    "ticks": carl.REGULATION_TICKS,
}

_FEATURE_WIDTHS = {
    "kind": len(TargetKind) - 1,
    "entity": len(TargetEntity) - 1,
    "arena_surfaces": len(ArenaSurface),
    "car_parts": len(ContactSurface),
    "state_metric": len(StateMetric),
    "state_min": 2,
    "state_max": 2,
    "gain_min": 2,
    "gain_max": 2,
    "goal_side": 1,
    "deadline": 2,
    "min_ticks": 2,
    "progress": 3,
    "event_flags": len(EVENT_FLAGS),
    "valid": 1,
}
POLICY_FEATURE_SLICES = {}
POLICY_FEATURE_DIM = sum(_FEATURE_WIDTHS.values())
_offset = 0
for _name, _width in _FEATURE_WIDTHS.items():
    POLICY_FEATURE_SLICES[_name] = slice(_offset, _offset + _width)
    _offset += _width


@dataclass(frozen=True)
class PolicyFeatures:
    window:   th.Tensor
    failures: th.Tensor

    @property
    def window_valid(self) -> th.Tensor:
        return self.window[..., POLICY_FEATURE_SLICES["valid"]].squeeze(-1)

    @property
    def failure_valid(self) -> th.Tensor:
        return self.failures[..., POLICY_FEATURE_SLICES["valid"]].squeeze(-1)


def _feature_constants(device: th.device) -> dict[str, th.Tensor]:
    """Allocate once at runtime construction, never in the policy hot path."""
    values = {
        "kind": list(range(EVENT, STATE + 1)),
        "entity": [CAR, BALL],
        "arena_surfaces": [int(surface) for surface in ArenaSurface],
        "car_parts": [1 << int(part) for part in ContactSurface],
        "state_metric": list(StateMetric),
        "event_flags": [int(flag) for flag in EVENT_FLAGS.values()],
    }
    constants = {name: th.tensor(value, device=device) for name, value in values.items()}
    constants["state_scales"] = th.tensor(
        [FEATURE_SCALES[name] for name in STATE_METRICS], dtype=th.float32, device=device,
    )

    return constants


def _multi_hot(
    mask: th.Tensor,
    bits: th.Tensor,
) -> th.Tensor:
    return (mask.unsqueeze(-1) & bits) != 0


def _condition_features(
    condition:    ConditionColumns,
    valid:        th.Tensor,
    team_sign:    th.Tensor,
    tick_cap:     float,
    constants:    dict[str, th.Tensor],
    *,
    deadline:     th.Tensor | None = None,
    has_deadline: th.Tensor | None = None,
    min_ticks:    th.Tensor | None = None,
    progress:     th.Tensor | None = None,
) -> th.Tensor:
    zeros = th.zeros_like(condition.kind, dtype=th.float32)
    scale = constants["state_scales"][condition.state_metric]

    def bound(
        value: th.Tensor,
        has:   th.Tensor,
        scale: th.Tensor | float,
    ) -> th.Tensor:
        return th.stack((value / scale, has), dim=-1)

    if deadline is None:
        deadline = has_deadline = min_ticks = zeros
    if progress is None:
        progress = zeros.unsqueeze(-1).expand(*zeros.shape, 3)
    time_scale = math.log1p(tick_cap)
    features = th.cat((
        condition.kind.unsqueeze(-1) == constants["kind"],
        condition.entity.unsqueeze(-1) == constants["entity"],
        _multi_hot(condition.arena_surfaces, constants["arena_surfaces"]),
        _multi_hot(condition.car_parts, constants["car_parts"]),
        ((condition.state_metric.unsqueeze(-1) == constants["state_metric"]) &
         (condition.kind == STATE).unsqueeze(-1)),
        bound(condition.state_min, condition.has_state_min, scale),
        bound(condition.state_max, condition.has_state_max, scale),
        bound(condition.gain_min, condition.has_gain_min, FEATURE_SCALES["gain"]),
        bound(condition.gain_max, condition.has_gain_max, FEATURE_SCALES["gain"]),
        (condition.goal_side * team_sign.unsqueeze(-1)).unsqueeze(-1),
        th.stack((deadline.float().log1p() / time_scale, has_deadline), dim=-1),
        th.stack((min_ticks.float().log1p() / time_scale, min_ticks > 0), dim=-1),
        progress,
        (_multi_hot(condition.event_flags, constants["event_flags"]) &
         (condition.kind == EVENT).unsqueeze(-1)),
        valid.unsqueeze(-1),
    ), dim=-1).to(th.float32)

    return th.where(valid.unsqueeze(-1), features, 0)


def policy_features(
    runtime:  TrainingPackRuntime,
    K:        int = 3,
    *,
    tick_cap: float = FEATURE_SCALES["ticks"],
) -> PolicyFeatures:
    """Fixed float32 features on the runtime device, with no exercise identity.

    ``window`` is [n_sim, K, POLICY_FEATURE_DIM], starting at the current stage;
    ``failures`` is [n_sim, F, POLICY_FEATURE_DIM] for that stage only. Each
    exercise-wide failure appears once. The last component is a valid bit;
    padded slots are entirely zero. See POLICY_FEATURE_SLICES for the layout.

    Bounds are divided by FEATURE_SCALES. Deadline and minimum ticks are
    (log1p(ticks) / log1p(tick_cap), has), with has=min_ticks>0 for minimum
    ticks. Only slot 0 has progress ratios (remaining time, repeats, total
    matches); unlimited quantities are zero. Goal side is +1 for the opponent's
    goal using runtime.team_sign (blue=+1, orange=-1).
    """
    if not isinstance(K, int) or isinstance(K, bool) or K < 1:
        raise ValueError("K must be a positive integer")
    if not math.isfinite(tick_cap) or tick_cap <= 0:
        raise ValueError("tick_cap must be positive and finite")

    pack = runtime.pack
    exercise = runtime.exercise_index.unsqueeze(-1)
    stages = runtime.stage_index.unsqueeze(-1) + th.arange(K, device=pack.kind.device)
    valid = stages < pack.lengths[runtime.exercise_index].unsqueeze(-1)
    index = exercise, stages.clamp_max(pack.kind.shape[1] - 1)
    valid = valid & pack.valid[index]
    progress = th.zeros((runtime.n_sim, K, 3), dtype=th.float32, device=pack.kind.device)
    current = runtime.exercise_index, runtime.stage_index
    for column, (value, limit, has_limit) in enumerate((
        (runtime.ticks_remaining, pack.deadline[current], pack.has_deadline[current]),
        (runtime.repeat_count, pack.max_repeats[current], pack.has_max_repeats[current]),
        (runtime.total_matches, pack.max_total_matches[current], pack.has_max_total_matches[current]),
    )):
        progress[:, 0, column] = th.where(has_limit, value.float() / limit.clamp_min(1), 0)

    window = _condition_features(
        pack.conditions(index), valid, runtime.team_sign, tick_cap, runtime._feature_constants,
        deadline=pack.deadline[index], has_deadline=pack.has_deadline[index],
        min_ticks=pack.min_ticks[index], progress=progress,
    )
    failure_conditions = pack.conditions(current, "failure_")
    failures = _condition_features(
        failure_conditions, failure_conditions.kind != 0, runtime.team_sign, tick_cap, runtime._feature_constants,
    )

    return PolicyFeatures(window=window, failures=failures)
