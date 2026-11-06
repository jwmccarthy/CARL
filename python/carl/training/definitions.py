from dataclasses import dataclass, field

from .initial import InitialState
from ..events import Condition


@dataclass
class Stage:

    event:        Condition
    within_ticks: int | None       = None
    fail_on:      list[Condition] = field(default_factory=list)
    from_exercise_start: bool = False
    repeat_on: Condition | None = None
    max_repeats: int | None = None
    max_total_matches: int | None = None

    def __post_init__(self):
        if self.within_ticks is not None and self.within_ticks <= 0:
            raise ValueError("within_ticks must be positive")

        if self.from_exercise_start and self.within_ticks is None:
            raise ValueError("exercise-start deadline requires within_ticks")

        if self.repeat_on is not None and self.from_exercise_start:
            raise ValueError("repeat_on cannot renew an exercise-start deadline")

        if self.max_repeats is not None and (self.repeat_on is None or self.max_repeats < 0):
            raise ValueError("max_repeats requires a repeat condition and must be nonnegative")

        if self.max_total_matches is not None and self.max_total_matches < 1:
            raise ValueError("max_total_matches must be positive")


@dataclass
class Exercise:

    name:        str
    initial:     InitialState
    stages:      list[Stage]
    fail_on:     list[Condition] = field(default_factory=list)
    target_goal: str             = "positive_y"

    def __post_init__(self):
        if not self.stages:
            raise ValueError("an exercise needs at least one stage")


@dataclass
class TrainingPack:

    name:      str
    exercises: list[Exercise]

    def __post_init__(self):
        if not self.exercises:
            raise ValueError("a training pack needs at least one exercise")
