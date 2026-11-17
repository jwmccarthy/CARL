from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch as th

from ..events import (
    ArenaSurface, BallTouch, BallTouchBatch, CarlEvents, Condition, ContactSurface,
    Demoed, Dodge, DoubleJump, Event, FlipReset, Goal, Jump, LeaveSurface,
    StateCondition, SurfaceContact,
)
from .definitions import TrainingPack

if TYPE_CHECKING:
    from ..gymnasium.state import CarlState
    from .reset import TrainingPackResetProvider
    from .targets import TrainingTarget


EVENT = 1
GOAL = 2
CONTACT = 3
LEAVE = 4
TOUCH = 5
STATE = 6

CAR = 1
BALL = 2

ALL_ARENA_SURFACES = sum(int(surface) for surface in ArenaSurface)
ALL_CAR_PARTS = sum(1 << int(part) for part in ContactSurface)

EVENT_FLAGS = {
    Jump: Event.JUMP,
    DoubleJump: Event.DOUBLE_JUMP,
    Dodge: Event.DODGE,
    FlipReset: Event.FLIP_RESET,
    Demoed: Event.DEMOED,
}

STATE_METRICS = {
    "ball_x":      0,
    "ball_y":      1,
    "ball_height": 2,
    "ball_speed":  3,
    "car_x":       4,
    "car_y":       5,
    "car_height":  6,
    "car_speed":   7,
    "car_boost":   8,
}

STATE_ENTITIES = {
    "ball_x": BALL,
    "ball_y": BALL,
    "ball_height": BALL,
    "ball_speed": BALL,
    "car_x": CAR,
    "car_y": CAR,
    "car_height": CAR,
    "car_speed": CAR,
    "car_boost": CAR,
}

CONDITION_DTYPES = {
    "kind": th.int64,
    "entity": th.int64,
    "event_flags": th.int64,
    "arena_surfaces": th.int64,
    "car_parts": th.int64,
    "state_metric": th.int64,
    "goal_side": th.int64,
    "state_min": th.float32,
    "state_max": th.float32,
    "gain_min": th.float32,
    "gain_max": th.float32,
    "has_state_min": th.bool,
    "has_state_max": th.bool,
    "has_gain_min": th.bool,
    "has_gain_max": th.bool,
}


def _bounds(
    prefix:  str,
    minimum: float | None,
    maximum: float | None,
    name:    str,
) -> dict[str, float | bool]:
    for bound in (minimum, maximum):
        if bound is not None and not math.isfinite(bound):
            raise ValueError(f"{name} bounds must be finite")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{name} minimum must not exceed maximum")

    return {
        f"{prefix}_min": minimum if minimum is not None else 0,
        f"{prefix}_max": maximum if maximum is not None else 0,
        f"has_{prefix}_min": minimum is not None,
        f"has_{prefix}_max": maximum is not None,
    }


def _condition(
    condition:   Condition,
    target_goal: str,
) -> dict[str, int | float | bool]:
    if type(condition) in EVENT_FLAGS:
        return dict(kind=EVENT, entity=CAR, event_flags=int(EVENT_FLAGS[type(condition)]))

    if isinstance(condition, StateCondition):
        if condition.metric not in STATE_METRICS:
            raise ValueError(f"unsupported state metric: {condition.metric}")
        if condition.minimum is None and condition.maximum is None:
            raise ValueError("state conditions require a minimum or maximum")

        return dict(
            kind=STATE, entity=STATE_ENTITIES[condition.metric],
            state_metric=STATE_METRICS[condition.metric],
            **_bounds("state", condition.minimum, condition.maximum, "state condition"),
        )

    if isinstance(condition, BallTouch):
        has_gain = condition.min_speed_gain is not None or condition.max_speed_gain is not None
        car_parts = ALL_CAR_PARTS
        if condition.surface is not None:
            parts = condition.surface if isinstance(condition.surface, tuple) else (condition.surface,)
            if not parts:
                raise ValueError("ball-touch car parts must not be empty")
            car_parts = 0
            for part in parts:
                car_parts |= 1 << int(ContactSurface(part))

        return dict(
            kind=TOUCH, entity=CAR,
            car_parts=car_parts,
            # The unqualified condition uses the car flag, not touch capture.
            event_flags=int(Event.BALL_TOUCH) if car_parts == ALL_CAR_PARTS and not has_gain else 0,
            **_bounds("gain", condition.min_speed_gain, condition.max_speed_gain, "ball-touch speed change"),
        )

    if isinstance(condition, (SurfaceContact, LeaveSurface)):
        if condition.entity not in ("car", "ball"):
            raise ValueError("surface condition entity must be 'car' or 'ball'")
        surfaces = int(condition.surface) if condition.surface is not None else ALL_ARENA_SURFACES
        if surfaces <= 0 or surfaces & ~ALL_ARENA_SURFACES:
            raise ValueError("invalid arena surface mask")
        wildcard_leave = (isinstance(condition, LeaveSurface) and condition.entity == "car"
                          and condition.surface is None)

        return dict(
            kind=CONTACT if isinstance(condition, SurfaceContact) else LEAVE,
            entity=CAR if condition.entity == "car" else BALL,
            arena_surfaces=surfaces,
            event_flags=int(Event.LEAVE_SURFACE) if wildcard_leave else 0,
        )

    if isinstance(condition, Goal):
        sides = {"positive_y": 1, "negative_y": -1}
        if target_goal not in sides:
            raise ValueError("target_goal must be 'positive_y' or 'negative_y'")

        return dict(kind=GOAL, entity=BALL, event_flags=int(Event.GOAL), goal_side=sides[target_goal])

    raise ValueError(f"unsupported training condition: {type(condition).__name__}")


@dataclass(frozen=True)
class ConditionColumns:
    kind:           th.Tensor
    entity:         th.Tensor
    event_flags:    th.Tensor
    arena_surfaces: th.Tensor
    car_parts:      th.Tensor
    state_metric:   th.Tensor
    goal_side:      th.Tensor
    state_min:      th.Tensor
    state_max:      th.Tensor
    gain_min:       th.Tensor
    gain_max:       th.Tensor
    has_state_min:  th.Tensor
    has_state_max:  th.Tensor
    has_gain_min:   th.Tensor
    has_gain_max:   th.Tensor


@dataclass(frozen=True)
class CompiledTrainingPack(ConditionColumns):
    source:                 TrainingPack
    deadline:               th.Tensor
    has_deadline:           th.Tensor
    min_ticks:              th.Tensor
    deadline_from_start:    th.Tensor
    repeat_kind:            th.Tensor
    repeat_entity:          th.Tensor
    repeat_event_flags:     th.Tensor
    repeat_arena_surfaces:  th.Tensor
    repeat_car_parts:       th.Tensor
    repeat_state_metric:    th.Tensor
    repeat_goal_side:       th.Tensor
    repeat_state_min:       th.Tensor
    repeat_state_max:       th.Tensor
    repeat_gain_min:        th.Tensor
    repeat_gain_max:        th.Tensor
    repeat_has_state_min:   th.Tensor
    repeat_has_state_max:   th.Tensor
    repeat_has_gain_min:    th.Tensor
    repeat_has_gain_max:    th.Tensor
    max_repeats:            th.Tensor
    has_max_repeats:        th.Tensor
    max_total_matches:      th.Tensor
    has_max_total_matches:  th.Tensor
    valid:                  th.Tensor
    failure_kind:           th.Tensor
    failure_entity:         th.Tensor
    failure_event_flags:    th.Tensor
    failure_arena_surfaces: th.Tensor
    failure_car_parts:      th.Tensor
    failure_state_metric:   th.Tensor
    failure_goal_side:      th.Tensor
    failure_state_min:      th.Tensor
    failure_state_max:      th.Tensor
    failure_gain_min:       th.Tensor
    failure_gain_max:       th.Tensor
    failure_has_state_min:  th.Tensor
    failure_has_state_max:  th.Tensor
    failure_has_gain_min:   th.Tensor
    failure_has_gain_max:   th.Tensor
    failure_exercise_wide:  th.Tensor
    lengths:                th.Tensor
    needs_touch:            bool
    needs_touch_speed:      bool
    needs_state:            bool

    def conditions(
        self,
        index:  th.Tensor | tuple[th.Tensor, ...],
        prefix: str = "",
    ) -> ConditionColumns:
        return ConditionColumns(**{
            name: getattr(self, prefix + name)[index] for name in CONDITION_DTYPES
        })

    @classmethod
    def compile(
        cls,
        pack:   TrainingPack,
        device: str | th.device = "cuda:0",
    ) -> CompiledTrainingPack:
        exercises = pack.exercises
        width = max(len(exercise.stages) for exercise in exercises)
        failures = max(1, max(len(exercise.fail_on) + len(stage.fail_on)
                              for exercise in exercises for stage in exercise.stages))
        shape = (len(exercises), width)
        columns = {
            prefix + name: th.zeros((*shape, failures) if prefix == "failure_" else shape, dtype=dtype)
            for prefix in ("", "repeat_", "failure_")
            for name, dtype in CONDITION_DTYPES.items()
        }
        for name, dtype in {
            "deadline": th.int64, "has_deadline": th.bool, "min_ticks": th.int64,
            "deadline_from_start": th.bool, "max_repeats": th.int64,
            "has_max_repeats": th.bool, "max_total_matches": th.int64,
            "has_max_total_matches": th.bool, "valid": th.bool,
        }.items():
            columns[name] = th.zeros(shape, dtype=dtype)
        columns["failure_exercise_wide"] = th.zeros((*shape, failures), dtype=th.bool)

        needs_touch = needs_touch_speed = needs_state = False

        def store(
            condition:   Condition,
            target_goal: str,
            index:       tuple[int, ...],
            prefix:      str = "",
        ) -> dict[str, int | float | bool]:
            nonlocal needs_touch, needs_touch_speed, needs_state
            values = _condition(condition, target_goal)
            for name, value in values.items():
                columns[prefix + name][index] = value
            gain = values.get("has_gain_min", False) or values.get("has_gain_max", False)
            needs_touch |= values["kind"] == TOUCH and (values["car_parts"] != ALL_CAR_PARTS or gain)
            needs_touch_speed |= values["kind"] == TOUCH and gain
            needs_state |= values["kind"] == STATE

            return values

        for index, exercise in enumerate(exercises):
            for stage_index, stage in enumerate(exercise.stages):
                # Definitions are mutable; revalidate with the same single source of rules.
                stage.__post_init__()
                slot = index, stage_index
                event = store(stage.event, exercise.target_goal, slot)
                columns["valid"][slot] = True
                columns["min_ticks"][slot] = stage.min_ticks
                columns["deadline_from_start"][slot] = stage.from_exercise_start
                for name, value in (("deadline", stage.within_ticks), ("max_repeats", stage.max_repeats),
                                    ("max_total_matches", stage.max_total_matches)):
                    if value is not None:
                        columns[name][slot] = value
                        columns["has_" + name][slot] = True

                repeat = (store(stage.repeat_on, exercise.target_goal, slot, "repeat_")
                          if stage.repeat_on is not None else {})
                if stage.max_total_matches is not None and not any(
                    condition.get("kind") == TOUCH and (
                        condition.get("has_gain_min", False) or condition.get("has_gain_max", False)
                    ) for condition in (event, repeat)
                ):
                    raise ValueError("max_total_matches requires a speed-qualified touch or repeat")

                for failure_index, condition in enumerate(exercise.fail_on + stage.fail_on):
                    failure_slot = *slot, failure_index
                    store(condition, exercise.target_goal, failure_slot, "failure_")
                    columns["failure_exercise_wide"][failure_slot] = failure_index < len(exercise.fail_on)

        return cls(
            source=pack,
            **{name: value.to(device) for name, value in columns.items()},
            lengths=th.tensor([len(e.stages) for e in exercises], device=device),
            needs_touch=needs_touch, needs_touch_speed=needs_touch_speed, needs_state=needs_state,
        )


@dataclass
class TrainingStep:
    succeeded: th.Tensor
    failed:    th.Tensor
    timed_out: th.Tensor
    advanced:  th.Tensor
    repeated:  th.Tensor | None = None


def _has_gain(condition: ConditionColumns) -> th.Tensor:
    return (condition.kind == TOUCH) & (condition.has_gain_min | condition.has_gain_max)


def _gain_matches(
    condition: ConditionColumns,
    touch:     BallTouchBatch,
) -> th.Tensor:
    surfaces = touch.tick_surfaces[:, :, 0].unsqueeze(-1)
    gain = (touch.tick_speed[:, :, 0] - touch.tick_speed_before[:, :, 0]).unsqueeze(-1)
    return (((surfaces & condition.car_parts.unsqueeze(0)) != 0) &
            (~condition.has_gain_min.unsqueeze(0) | (gain >= condition.gain_min.unsqueeze(0))) &
            (~condition.has_gain_max.unsqueeze(0) | (gain <= condition.gain_max.unsqueeze(0))))


class TrainingPackRuntime:

    def __init__(
        self,
        pack:      CompiledTrainingPack,
        n_sim:     int,
        *,
        team_sign: int | th.Tensor = 1,
    ) -> None:
        if n_sim < 1:
            raise ValueError("n_sim must be positive")

        self.pack = pack
        self.n_sim = n_sim
        device = pack.kind.device
        from .targets import _feature_constants

        self._feature_constants = _feature_constants(device)
        # As in CarlState: blue attacks positive Y (+1), orange negative Y (-1).
        self.team_sign = th.broadcast_to(th.as_tensor(team_sign, dtype=th.float32, device=device), (n_sim,)).clone()
        th._assert_async(((self.team_sign == 1) | (self.team_sign == -1)).all(), "team_sign must be +1 or -1")

        self.exercise_index = th.zeros(n_sim, dtype=th.int64, device=device)
        self.stage_index = th.zeros_like(self.exercise_index)
        self.ticks_remaining = th.zeros_like(self.exercise_index)
        self.elapsed_ticks = th.zeros_like(self.exercise_index)
        self.stage_elapsed_ticks = th.zeros_like(self.exercise_index)
        self.repeat_count = th.zeros_like(self.exercise_index)
        self.total_matches = th.zeros_like(self.exercise_index)
        self.active = th.zeros(n_sim, dtype=th.bool, device=device)

    def reset(
        self,
        selected:       th.Tensor,
        exercise_index: th.Tensor,
    ) -> None:
        if selected.dtype != th.bool or selected.shape != (self.n_sim,):
            raise ValueError("selected must be a per-simulation boolean mask")
        if exercise_index.dtype != th.int64 or exercise_index.shape != (self.n_sim,):
            raise ValueError("exercise_index must be a per-simulation int64 tensor")
        if selected.device != self.active.device or exercise_index.device != self.active.device:
            raise ValueError("reset tensors must be on the runtime device")

        indices = th.where(selected, exercise_index, self.exercise_index)
        th._assert_async(((indices >= 0) & (indices < len(self.pack.lengths))).all(),
                         "exercise index out of range")
        deadlines = self.pack.deadline[indices, 0]

        self.exercise_index = indices
        self.stage_index = th.where(selected, 0, self.stage_index)
        self.ticks_remaining = th.where(selected, deadlines, self.ticks_remaining)
        self.elapsed_ticks = th.where(selected, 0, self.elapsed_ticks)
        self.stage_elapsed_ticks = th.where(selected, 0, self.stage_elapsed_ticks)
        self.repeat_count = th.where(selected, 0, self.repeat_count)
        self.total_matches = th.where(selected, 0, self.total_matches)
        self.active = self.active | selected

    def target_sequence(self) -> TrainingTarget:
        from .targets import training_target
        return training_target(self)

    def _matches(
        self,
        condition:    ConditionColumns,
        events:       CarlEvents,
        state_values: th.Tensor | None,
    ) -> th.Tensor:
        car = events.cars.flags[:, 0].unsqueeze(-1)
        ball = events.ball.flags.unsqueeze(-1)
        goal_side = events.ball.goal_side.unsqueeze(-1)
        car_contact = events.cars.surface_contacted[:, 0].unsqueeze(-1)
        car_left = events.cars.surface_left[:, 0].unsqueeze(-1)
        ball_contact = events.ball.surface_entered.unsqueeze(-1)
        ball_left = events.ball.surface_left.unsqueeze(-1)
        touch = (events.ball_touches.surfaces[:, 0].unsqueeze(-1)
                 if events.ball_touches is not None else th.zeros_like(car))

        state_match = th.zeros_like(condition.kind, dtype=th.bool)
        speed_match = th.zeros_like(state_match)
        has_gain = _has_gain(condition)
        if (events.ball_touches is not None and events.ball_touches.tick_speed is not None
                and events.ball_touches.tick_speed_before is not None):
            speed_match = _gain_matches(condition, events.ball_touches).any(dim=0)

        if state_values is not None:
            values = state_values.gather(1, condition.state_metric)
            state_match = ((condition.kind == STATE) &
                           (~condition.has_state_min | (values >= condition.state_min)) &
                           (~condition.has_state_max | (values <= condition.state_max)))
            # Goal detection resets the ball before the transition state is captured.
            goal = (ball & int(Event.GOAL)) != 0
            state_match = state_match & ~(goal & (condition.entity == BALL))

        is_car = condition.entity == CAR
        flag_leave = (condition.event_flags & int(Event.LEAVE_SURFACE)) != 0
        flag_touch = (condition.event_flags & int(Event.BALL_TOUCH)) != 0
        contact = th.where(is_car, car_contact, ball_contact) & condition.arena_surfaces
        ceiling = (is_car & ((condition.arena_surfaces & int(ArenaSurface.CEILING)) != 0) &
                   ((car & int(Event.CEILING_BODY_CONTACT)) != 0))
        left = th.where(is_car, car_left, ball_left) & condition.arena_surfaces

        return (
            ((condition.kind == EVENT) & ((car & condition.event_flags) != 0)) |
            ((condition.kind == GOAL) & ((ball & int(Event.GOAL)) != 0) & (goal_side == condition.goal_side)) |
            ((condition.kind == CONTACT) & ((contact != 0) | ceiling)) |
            ((condition.kind == LEAVE) & th.where(flag_leave, (car & int(Event.LEAVE_SURFACE)) != 0, left != 0)) |
            ((condition.kind == TOUCH) & th.where(
                has_gain, speed_match,
                th.where(flag_touch, (car & int(Event.BALL_TOUCH)) != 0, (touch & condition.car_parts) != 0),
            )) |
            state_match
        )

    def step(
        self,
        events:    CarlEvents,
        frameskip: int,
        state:     CarlState | None = None,
    ) -> TrainingStep:
        if frameskip < 1:
            raise ValueError("frameskip must be positive")
        if events.cars is None or events.ball is None:
            raise ValueError("training requires event capture")
        if events.cars.flags.shape != (self.n_sim, 1):
            raise ValueError("training requires one car per simulation")
        if self.pack.needs_touch and events.ball_touches is None:
            raise ValueError("surface-qualified ball touches require touch capture")
        if self.pack.needs_touch_speed and (events.ball_touches.tick_speed is None
                                            or events.ball_touches.tick_surfaces is None
                                            or events.ball_touches.tick_speed_before is None):
            raise ValueError("ball-touch speed conditions require per-tick touch capture")
        if self.pack.needs_state and state is None:
            raise ValueError("state conditions require the transition state")

        events.wait()
        state_values = None
        if self.pack.needs_state:
            if state.ball_position.shape != (self.n_sim, 3) or state.car_position.shape != (self.n_sim, 1, 3):
                raise ValueError("state conditions require one car per simulation")
            state_values = th.stack((
                state.ball_position[:, 0],
                state.ball_position[:, 1],
                state.ball_position[:, 2],
                state.ball_velocity.norm(dim=-1),
                state.car_position[:, 0, 0],
                state.car_position[:, 0, 1],
                state.car_position[:, 0, 2],
                state.car_velocity[:, 0].norm(dim=-1),
                state.car_boost[:, 0],
            ), dim=-1)

        index = self.exercise_index, self.stage_index
        expanded_index = self.exercise_index.unsqueeze(-1), self.stage_index.unsqueeze(-1)
        condition = self.pack.conditions(expanded_index)
        repeat = self.pack.conditions(expanded_index, "repeat_")
        matched = self._matches(condition, events, state_values).squeeze(-1)
        elapsed_for_minimum = th.where(self.pack.deadline_from_start[index], self.elapsed_ticks,
                                       self.stage_elapsed_ticks)
        matched = matched & (elapsed_for_minimum >= self.pack.min_ticks[index])
        repeat_match = self._matches(repeat, events, state_values).squeeze(-1)
        repeat_events = repeat_match.long()
        total_events = th.zeros_like(repeat_events)
        event_has_gain = _has_gain(condition).squeeze(-1)
        repeat_has_gain = _has_gain(repeat).squeeze(-1)

        if events.ball_touches is not None and events.ball_touches.tick_speed_before is not None:
            gain_count = _gain_matches(repeat, events.ball_touches).sum(dim=0).squeeze(-1)
            repeat_events = th.where(repeat_has_gain, gain_count, repeat_events)
            event_count = _gain_matches(condition, events.ball_touches).sum(dim=0).squeeze(-1)
            total_events = th.where(event_has_gain, event_count, repeat_events)
        failed_event = self._matches(self.pack.conditions(index, "failure_"), events, state_values).any(dim=-1)

        remaining = self.ticks_remaining
        has_deadline = self.pack.has_deadline[index]
        in_time = ~has_deadline | (remaining >= frameskip)
        timed_out = has_deadline & (remaining <= frameskip) & ~((matched | repeat_match) & in_time)
        limit = self.pack.max_repeats[index]
        overflow = self.pack.has_max_repeats[index] & (self.repeat_count + repeat_events > limit) & ~matched & in_time
        total_limit = self.pack.max_total_matches[index]
        total_overflow = (self.pack.has_max_total_matches[index] & (self.total_matches + total_events > total_limit) &
                          ~(~event_has_gain & matched) & in_time)
        failed = self.active & (failed_event | overflow | total_overflow)
        timed_out = self.active & timed_out & ~failed
        advanced = self.active & matched & in_time & ~failed & ~timed_out
        repeated = self.active & repeat_match & in_time & ~failed & ~timed_out & ~advanced
        succeeded = advanced & (self.stage_index + 1 == self.pack.lengths[self.exercise_index])

        next_stage = th.where(advanced & ~succeeded, self.stage_index + 1, self.stage_index)
        next_deadline = self.pack.deadline[self.exercise_index, next_stage]
        exercise_elapsed = self.elapsed_ticks + frameskip
        next_deadline = th.where(
            self.pack.deadline_from_start[self.exercise_index, next_stage],
            next_deadline - exercise_elapsed, next_deadline,
        )
        elapsed = th.where(has_deadline, remaining - frameskip, remaining)
        self.ticks_remaining = th.where(
            self.active, th.where(advanced & ~succeeded, next_deadline,
                                  th.where(repeated, self.pack.deadline[index], elapsed)), remaining,
        )
        self.stage_index = next_stage
        self.repeat_count = th.where(advanced, 0, self.repeat_count + repeated.long() * repeat_events)
        counted = self.active & ~failed & ~timed_out & in_time & self.pack.has_max_total_matches[index]
        self.total_matches = self.total_matches + counted.long() * total_events
        self.elapsed_ticks = th.where(self.active, exercise_elapsed, self.elapsed_ticks)
        self.stage_elapsed_ticks = th.where(self.active, th.where(advanced, 0, self.stage_elapsed_ticks + frameskip),
                                            self.stage_elapsed_ticks)
        self.active = self.active & ~failed & ~timed_out & ~succeeded

        return TrainingStep(succeeded=succeeded, failed=failed, timed_out=timed_out, advanced=advanced,
                            repeated=repeated)


class TrainingPackTerminationManager:

    def __init__(
        self,
        runtime:  TrainingPackRuntime,
        provider: TrainingPackResetProvider,
    ) -> None:
        if runtime.n_sim != len(provider.exercise_index):
            raise ValueError("runtime and reset provider must have the same simulation count")
        if runtime.pack.kind.shape[0] != len(provider.pack.exercises):
            raise ValueError("runtime and reset provider must have the same exercise count")
        if runtime.pack.source is not provider.pack:
            raise ValueError("runtime and reset provider must use the same training pack")

        self.runtime = runtime
        self.provider = provider

    def reset(self, reset_mask: th.Tensor) -> None:
        self.runtime.reset(reset_mask, self.provider.exercise_index)

    def step(
        self,
        events:    CarlEvents,
        frameskip: int,
        state:     CarlState | None = None,
    ) -> TrainingStep:
        return self.runtime.step(events, frameskip, state=state)
