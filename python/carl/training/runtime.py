from dataclasses import dataclass
import math

import torch as th

from ..events import (
    ArenaSurface, BallTouch, Demoed, Dodge, DoubleJump, Event, FlipReset, Goal, Jump,
    LeaveSurface, StateCondition, SurfaceContact,
)


CAR_EVENT = 1
BALL_GOAL = 2
CAR_CONTACT = 3
CAR_LEAVE = 4
BALL_CONTACT = 5
BALL_LEAVE = 6
TOUCH_SURFACE = 7
STATE_VALUE = 8
TOUCH_GAIN = 9

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


def _condition(condition, target_goal):
    flags = {
        Jump: Event.JUMP,
        DoubleJump: Event.DOUBLE_JUMP,
        Dodge: Event.DODGE,
        FlipReset: Event.FLIP_RESET,
        Demoed: Event.DEMOED,
    }

    if type(condition) in flags:
        return CAR_EVENT, int(flags[type(condition)]), 0

    if isinstance(condition, StateCondition):
        if condition.metric not in STATE_METRICS:
            raise ValueError(f"unsupported state metric: {condition.metric}")

        if condition.minimum is None and condition.maximum is None:
            raise ValueError("state conditions require a minimum or maximum")

        for bound in (condition.minimum, condition.maximum):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("state condition bounds must be finite")

        if (condition.minimum is not None and condition.maximum is not None
                and condition.minimum > condition.maximum):
            raise ValueError("state condition minimum must not exceed maximum")

        return STATE_VALUE, STATE_METRICS[condition.metric], 0

    if isinstance(condition, BallTouch):
        if condition.min_height is not None:
            raise ValueError("ball-touch height filters are not supported")

        if condition.min_speed_gain is not None or condition.max_speed_gain is not None:
            for bound in (condition.min_speed_gain, condition.max_speed_gain):
                if bound is not None and not math.isfinite(bound):
                    raise ValueError("ball-touch speed changes must be finite")
            if (condition.min_speed_gain is not None and condition.max_speed_gain is not None
                    and condition.min_speed_gain > condition.max_speed_gain):
                raise ValueError("minimum ball-touch speed change must not exceed maximum")
            return TOUCH_GAIN, (1 << int(condition.surface)) if condition.surface is not None else 63, 0

        if condition.surface is None:
            return CAR_EVENT, int(Event.BALL_TOUCH), 0

        return TOUCH_SURFACE, 1 << int(condition.surface), 0

    if isinstance(condition, (SurfaceContact, LeaveSurface)):
        if condition.entity not in ("car", "ball"):
            raise ValueError("surface condition entity must be 'car' or 'ball'")

        if isinstance(condition, LeaveSurface) and condition.entity == "car" and condition.surface is None:
            return CAR_EVENT, int(Event.LEAVE_SURFACE), 0

        mask = int(condition.surface) if condition.surface is not None else 31

        if mask <= 0 or mask & ~31:
            raise ValueError("invalid arena surface mask")

        if condition.entity == "car":
            kind = CAR_CONTACT if isinstance(condition, SurfaceContact) else CAR_LEAVE
        else:
            kind = BALL_CONTACT if isinstance(condition, SurfaceContact) else BALL_LEAVE

        return kind, mask, 0

    if isinstance(condition, Goal):
        sides = {"positive_y": 1, "negative_y": -1}

        if target_goal not in sides:
            raise ValueError("target_goal must be 'positive_y' or 'negative_y'")

        return BALL_GOAL, int(Event.GOAL), sides[target_goal]

    raise ValueError(f"unsupported training condition: {type(condition).__name__}")


@dataclass
class CompiledTrainingPack:

    source:                object
    kind:                  th.Tensor
    mask:                  th.Tensor
    parameter:             th.Tensor
    minimum:               th.Tensor
    maximum:               th.Tensor
    deadline:              th.Tensor
    deadline_from_start:   th.Tensor
    repeat_kind:           th.Tensor
    repeat_mask:           th.Tensor
    repeat_parameter:      th.Tensor
    repeat_minimum:        th.Tensor
    repeat_maximum:        th.Tensor
    max_repeats:           th.Tensor
    max_total_matches:      th.Tensor
    valid:                 th.Tensor
    failure_kind:          th.Tensor
    failure_mask:          th.Tensor
    failure_parameter:     th.Tensor
    failure_minimum:       th.Tensor
    failure_maximum:       th.Tensor
    failure_exercise_wide: th.Tensor
    lengths:               th.Tensor
    needs_touch:           bool
    needs_touch_speed:     bool
    needs_state:           bool

    @classmethod
    def compile(cls, pack, device="cuda:0"):
        exercises = pack.exercises
        width = max(len(exercise.stages) for exercise in exercises)
        failures = max(1, max(len(exercise.fail_on) + len(stage.fail_on)
                              for exercise in exercises for stage in exercise.stages))
        shape = (len(exercises), width)

        kinds = th.zeros(shape, dtype=th.int64)
        masks = th.zeros_like(kinds)
        parameters = th.zeros_like(kinds)
        minimum = th.full(shape, -math.inf, dtype=th.float32)
        maximum = th.full(shape, math.inf, dtype=th.float32)
        deadlines = th.full(shape, -1, dtype=th.int64)
        deadline_from_start = th.zeros(shape, dtype=th.bool)
        repeat_kind = th.zeros(shape, dtype=th.int64)
        repeat_mask = th.zeros_like(repeat_kind)
        repeat_parameter = th.zeros_like(repeat_kind)
        repeat_minimum = th.full(shape, -math.inf, dtype=th.float32)
        repeat_maximum = th.full(shape, math.inf, dtype=th.float32)
        max_repeats = th.full(shape, -1, dtype=th.int64)
        max_total_matches = th.full(shape, -1, dtype=th.int64)
        valid = th.zeros(shape, dtype=th.bool)
        failure_kind = th.zeros((*shape, failures), dtype=th.int64)
        failure_mask = th.zeros_like(failure_kind)
        failure_parameter = th.zeros_like(failure_kind)
        failure_minimum = th.full((*shape, failures), -math.inf, dtype=th.float32)
        failure_maximum = th.full((*shape, failures), math.inf, dtype=th.float32)
        failure_exercise_wide = th.zeros((*shape, failures), dtype=th.bool)

        needs_touch = False
        needs_touch_speed = False
        needs_state = False

        for index, exercise in enumerate(exercises):
            for stage_index, stage in enumerate(exercise.stages):
                kind, mask, parameter = _condition(stage.event, exercise.target_goal)
                kinds[index, stage_index] = kind
                masks[index, stage_index] = mask
                parameters[index, stage_index] = parameter

                if stage.within_ticks is not None and stage.within_ticks <= 0:
                    raise ValueError("within_ticks must be positive")

                deadlines[index, stage_index] = stage.within_ticks if stage.within_ticks is not None else -1
                deadline_from_start[index, stage_index] = stage.from_exercise_start
                valid[index, stage_index] = True
                max_total_matches[index, stage_index] = (stage.max_total_matches
                                                        if stage.max_total_matches is not None else -1)
                needs_touch |= kind in (TOUCH_SURFACE, TOUCH_GAIN)
                needs_touch_speed |= kind == TOUCH_GAIN
                needs_state |= kind == STATE_VALUE

                if kind == STATE_VALUE:
                    if stage.event.minimum is not None:
                        minimum[index, stage_index] = stage.event.minimum
                    if stage.event.maximum is not None:
                        maximum[index, stage_index] = stage.event.maximum

                if kind == TOUCH_GAIN:
                    if stage.event.min_speed_gain is not None:
                        minimum[index, stage_index] = stage.event.min_speed_gain
                    if stage.event.max_speed_gain is not None:
                        maximum[index, stage_index] = stage.event.max_speed_gain

                if stage.repeat_on is not None:
                    max_repeats[index, stage_index] = stage.max_repeats if stage.max_repeats is not None else -1
                    repeat = stage.repeat_on
                    kind, mask, parameter = _condition(repeat, exercise.target_goal)
                    repeat_kind[index, stage_index] = kind
                    repeat_mask[index, stage_index] = mask
                    repeat_parameter[index, stage_index] = parameter
                    needs_touch |= kind in (TOUCH_SURFACE, TOUCH_GAIN)
                    needs_touch_speed |= kind == TOUCH_GAIN
                    needs_state |= kind == STATE_VALUE

                    if kind in (STATE_VALUE, TOUCH_GAIN):
                        lower = repeat.minimum if kind == STATE_VALUE else repeat.min_speed_gain
                        upper = repeat.maximum if kind == STATE_VALUE else repeat.max_speed_gain
                        if lower is not None:
                            repeat_minimum[index, stage_index] = lower
                        if upper is not None:
                            repeat_maximum[index, stage_index] = upper

                if stage.max_total_matches is not None:
                    event_kind = kinds[index, stage_index].item()
                    repeated_kind = repeat_kind[index, stage_index].item()
                    if event_kind != TOUCH_GAIN and repeated_kind != TOUCH_GAIN:
                        raise ValueError("max_total_matches requires a speed-qualified touch or repeat")

                for failure_index, condition in enumerate(exercise.fail_on + stage.fail_on):
                    kind, mask, parameter = _condition(condition, exercise.target_goal)
                    failure_kind[index, stage_index, failure_index] = kind
                    failure_mask[index, stage_index, failure_index] = mask
                    failure_parameter[index, stage_index, failure_index] = parameter
                    failure_exercise_wide[index, stage_index, failure_index] = failure_index < len(exercise.fail_on)
                    needs_touch |= kind in (TOUCH_SURFACE, TOUCH_GAIN)
                    needs_touch_speed |= kind == TOUCH_GAIN
                    needs_state |= kind == STATE_VALUE

                    if kind == STATE_VALUE:
                        if condition.minimum is not None:
                            failure_minimum[index, stage_index, failure_index] = condition.minimum
                        if condition.maximum is not None:
                            failure_maximum[index, stage_index, failure_index] = condition.maximum

                    if kind == TOUCH_GAIN:
                        if condition.min_speed_gain is not None:
                            failure_minimum[index, stage_index, failure_index] = condition.min_speed_gain
                        if condition.max_speed_gain is not None:
                            failure_maximum[index, stage_index, failure_index] = condition.max_speed_gain

        return cls(
            source                = pack,
            kind                  = kinds.to(device),
            mask                  = masks.to(device),
            parameter             = parameters.to(device),
            minimum               = minimum.to(device),
            maximum               = maximum.to(device),
            deadline              = deadlines.to(device),
            deadline_from_start   = deadline_from_start.to(device),
            repeat_kind           = repeat_kind.to(device),
            repeat_mask           = repeat_mask.to(device),
            repeat_parameter      = repeat_parameter.to(device),
            repeat_minimum        = repeat_minimum.to(device),
            repeat_maximum        = repeat_maximum.to(device),
            max_repeats           = max_repeats.to(device),
            max_total_matches     = max_total_matches.to(device),
            valid                 = valid.to(device),
            failure_kind          = failure_kind.to(device),
            failure_mask          = failure_mask.to(device),
            failure_parameter     = failure_parameter.to(device),
            failure_minimum       = failure_minimum.to(device),
            failure_maximum       = failure_maximum.to(device),
            failure_exercise_wide = failure_exercise_wide.to(device),
            lengths               = th.tensor([len(e.stages) for e in exercises], device=device),
            needs_touch           = needs_touch,
            needs_touch_speed     = needs_touch_speed,
            needs_state           = needs_state,
        )


@dataclass
class TrainingStep:

    succeeded: th.Tensor
    failed:    th.Tensor
    timed_out: th.Tensor
    advanced:  th.Tensor
    repeated:  th.Tensor | None = None


class TrainingPackRuntime:

    def __init__(self, pack: CompiledTrainingPack, n_sim: int):
        if n_sim < 1:
            raise ValueError("n_sim must be positive")

        self.pack = pack
        self.n_sim = n_sim
        device = pack.kind.device

        self.exercise_index = th.zeros(n_sim, dtype=th.int64, device=device)
        self.stage_index = th.zeros_like(self.exercise_index)
        self.ticks_remaining = th.full_like(self.exercise_index, -1)
        self.elapsed_ticks = th.zeros_like(self.exercise_index)
        self.repeat_count = th.zeros_like(self.exercise_index)
        self.total_matches = th.zeros_like(self.exercise_index)
        self.active = th.zeros(n_sim, dtype=th.bool, device=device)

    def reset(self, selected: th.Tensor, exercise_index: th.Tensor):
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
        self.repeat_count = th.where(selected, 0, self.repeat_count)
        self.total_matches = th.where(selected, 0, self.total_matches)
        self.active = self.active | selected

    def target_sequence(self):
        from .targets import training_target

        return training_target(self)

    def _matches(self, kind, mask, parameter, minimum, maximum, events, state_values):
        car = events.cars.flags[:, 0].unsqueeze(-1)
        ball = events.ball.flags.unsqueeze(-1)
        goal_side = events.ball.goal_side.unsqueeze(-1)

        car_contact = events.cars.surface_contacted[:, 0].unsqueeze(-1)
        car_left = events.cars.surface_left[:, 0].unsqueeze(-1)
        ball_contact = events.ball.surface_entered.unsqueeze(-1)
        ball_left = events.ball.surface_left.unsqueeze(-1)

        touch = (events.ball_touches.surfaces[:, 0].unsqueeze(-1)
                 if events.ball_touches is not None else th.zeros_like(car))

        state_match = th.zeros_like(kind, dtype=th.bool)
        speed_match = th.zeros_like(kind, dtype=th.bool)

        if (events.ball_touches is not None and events.ball_touches.tick_speed is not None
                and events.ball_touches.tick_speed_before is not None):
            surfaces = events.ball_touches.tick_surfaces[:, :, 0].unsqueeze(-1)
            speed = (events.ball_touches.tick_speed[:, :, 0] -
                     events.ball_touches.tick_speed_before[:, :, 0]).unsqueeze(-1)
            speed_match = (((surfaces & mask.unsqueeze(0)) != 0) &
                           (speed >= minimum.unsqueeze(0)) & (speed <= maximum.unsqueeze(0))).any(dim=0)

        if state_values is not None:
            values = state_values.gather(1, mask.clamp(0, len(STATE_METRICS) - 1))
            state_match = (kind == STATE_VALUE) & (values >= minimum) & (values <= maximum)

        return (
            ((kind == CAR_EVENT) & ((car & mask) != 0)) |
            ((kind == BALL_GOAL) & ((ball & mask) != 0) & (goal_side == parameter)) |
            ((kind == CAR_CONTACT) & (((car_contact & mask) != 0) |
             (((mask & int(ArenaSurface.CEILING)) != 0) & ((car & int(Event.CEILING_BODY_CONTACT)) != 0)))) |
            ((kind == CAR_LEAVE) & ((car_left & mask) != 0)) |
            ((kind == BALL_CONTACT) & ((ball_contact & mask) != 0)) |
            ((kind == BALL_LEAVE) & ((ball_left & mask) != 0)) |
            ((kind == TOUCH_SURFACE) & ((touch & mask) != 0)) |
            ((kind == TOUCH_GAIN) & speed_match) |
            state_match
        )

    def step(self, events, frameskip: int, state=None) -> TrainingStep:
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
        kind = self.pack.kind[index].unsqueeze(-1)
        mask = self.pack.mask[index].unsqueeze(-1)
        parameter = self.pack.parameter[index].unsqueeze(-1)

        matched = self._matches(
            kind, mask, parameter, self.pack.minimum[index].unsqueeze(-1),
            self.pack.maximum[index].unsqueeze(-1), events, state_values,
        ).squeeze(-1)
        repeat_match = self._matches(
            self.pack.repeat_kind[index].unsqueeze(-1), self.pack.repeat_mask[index].unsqueeze(-1),
            self.pack.repeat_parameter[index].unsqueeze(-1), self.pack.repeat_minimum[index].unsqueeze(-1),
            self.pack.repeat_maximum[index].unsqueeze(-1), events, state_values,
        ).squeeze(-1)
        repeat_events = repeat_match.long()
        total_events = th.zeros_like(repeat_events)

        if events.ball_touches is not None and events.ball_touches.tick_speed_before is not None:
            touch = events.ball_touches
            surfaces = touch.tick_surfaces[:, :, 0]
            gain = touch.tick_speed[:, :, 0] - touch.tick_speed_before[:, :, 0]
            gain_count = (((surfaces & self.pack.repeat_mask[index].unsqueeze(0)) != 0) &
                          (gain >= self.pack.repeat_minimum[index].unsqueeze(0)) &
                          (gain <= self.pack.repeat_maximum[index].unsqueeze(0))).sum(dim=0)
            repeat_events = th.where(self.pack.repeat_kind[index] == TOUCH_GAIN, gain_count, repeat_events)
            event_count = (((surfaces & self.pack.mask[index].unsqueeze(0)) != 0) &
                           (gain >= self.pack.minimum[index].unsqueeze(0)) &
                           (gain <= self.pack.maximum[index].unsqueeze(0))).sum(dim=0)
            total_events = th.where(self.pack.kind[index] == TOUCH_GAIN, event_count, repeat_events)
        failed_event = self._matches(
            self.pack.failure_kind[index], self.pack.failure_mask[index],
            self.pack.failure_parameter[index], self.pack.failure_minimum[index],
            self.pack.failure_maximum[index], events, state_values,
        ).any(dim=-1)

        remaining = self.ticks_remaining
        has_deadline = self.pack.deadline[index] >= 0
        in_time = ~has_deadline | (remaining >= frameskip)
        timed_out = has_deadline & (remaining <= frameskip) & ~((matched | repeat_match) & in_time)

        limit = self.pack.max_repeats[index]
        overflow = (limit >= 0) & (self.repeat_count + repeat_events > limit) & ~matched & in_time
        total_limit = self.pack.max_total_matches[index]
        total_overflow = ((total_limit >= 0) & (self.total_matches + total_events > total_limit) &
                          ~((self.pack.kind[index] != TOUCH_GAIN) & matched) & in_time)
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
        counted = self.active & ~failed & ~timed_out & in_time & (total_limit >= 0)
        self.total_matches = self.total_matches + counted.long() * total_events
        self.elapsed_ticks = th.where(self.active, exercise_elapsed, self.elapsed_ticks)
        self.active = self.active & ~failed & ~timed_out & ~succeeded

        return TrainingStep(succeeded=succeeded, failed=failed, timed_out=timed_out, advanced=advanced,
                            repeated=repeated)


class TrainingPackTerminationManager:

    def __init__(self, runtime: TrainingPackRuntime, provider):
        if runtime.n_sim != len(provider.exercise_index):
            raise ValueError("runtime and reset provider must have the same simulation count")

        if runtime.pack.kind.shape[0] != len(provider.pack.exercises):
            raise ValueError("runtime and reset provider must have the same exercise count")

        if runtime.pack.source is not provider.pack:
            raise ValueError("runtime and reset provider must use the same training pack")

        self.runtime = runtime
        self.provider = provider

    def reset(self, reset_mask: th.Tensor):
        self.runtime.reset(reset_mask, self.provider.exercise_index)

    def step(self, events, frameskip: int, state=None) -> TrainingStep:
        return self.runtime.step(events, frameskip, state=state)
