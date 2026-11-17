"""Specific aerial event sequences, paired with their left/right reflections.

One stage advances per decision window. Persistent state checkpoints can follow
an event, but Jump -> LeaveSurface is deliberately not used: both usually occur
in the same window. Failure guards impose contact heights and clean flight;
exercise-start deadlines prevent a slow setup from qualifying as a fast aerial.
"""

import math

import carl
from carl.events import (
    ArenaSurface, BallTouch, ContactSurface, Demoed, Dodge, DoubleJump, Goal, Jump,
    LeaveSurface, StateCondition, SurfaceContact,
)
from carl.training import (
    BallState, CarState, Euler, Exercise, InitialState, InAir, OnSurface, Stage,
    TrainingPack, Uniform,
)


SETUP_CONTACTS = (ContactSurface.FRONT, ContactSurface.ROOF)
BALL_HEIGHT_INCREASE = 250


def _ticks(seconds: float) -> int:
    return round(seconds * carl.PHYS_TICKS_PER_SECOND)


def _ball_flight_failures(height: float = carl.BALL_REST_Z + 2) -> list[SurfaceContact | StateCondition]:
    return [
        SurfaceContact(ArenaSurface.FLOOR, entity="ball"),
        StateCondition("ball_height", maximum=height),
    ]


def _airborne_failures(
    car_height:  float = carl.CAR_REST_Z + 2,
    ball_height: float = carl.BALL_REST_Z + 2,
) -> list[SurfaceContact | StateCondition]:
    return [
        SurfaceContact(entity="car"),
        StateCondition("car_height", maximum=car_height),
        *_ball_flight_failures(ball_height),
    ]


def _touch_failures(
    car_height:  float,
    ball_height: float,
) -> list[SurfaceContact | StateCondition | BallTouch]:
    # Nose/roof edge overlap is legitimate. Rear, side and underbody contacts
    # are not the intended catch or push and should not be ignored until a
    # later lucky touch. One mask represents all four disallowed faces.
    return [
        *_airborne_failures(car_height, ball_height),
        BallTouch((ContactSurface.REAR, ContactSurface.LEFT, ContactSurface.RIGHT, ContactSurface.UNDERSIDE)),
    ]


def _ground_start(ball_height: Uniform) -> InitialState:
    return InitialState(
        ego=CarState(
            placement=OnSurface(
                position=(Uniform(-1050, -850), Uniform(-2700, -2500)),
                heading=Uniform(math.pi / 2 - 0.08, math.pi / 2 + 0.08),
            ),
            boost=carl.BOOST_MAX,
        ),
        ball=BallState(
            position=(Uniform(-1000, -800), Uniform(-1800, -1600), ball_height),
            velocity=(0, Uniform(150, 250), Uniform(250, 350)),
        ),
    )


def _air_start(
    car_height:  float,
    ball_height: float,
    *,
    shot:        bool = False,
) -> InitialState:
    return InitialState(
        ego=CarState(
            placement=InAir(
                position=(Uniform(-1050, -950), Uniform(-900, -800), car_height),
                orientation=Euler(pitch=-0.15 if shot else -0.25, yaw=math.pi / 2),
            ),
            velocity=(0, 800 if shot else 500, 100),
            boost=carl.BOOST_MAX,
        ),
        ball=BallState(
            position=(Uniform(-1050, -950), Uniform(-350, -250), ball_height),
            velocity=(0, 250, 200),
        ),
    )


def _dribble_start() -> InitialState:
    return InitialState(
        ego=CarState(
            placement=InAir(
                position=(Uniform(-1050, -950), Uniform(-900, -800), 650),
                orientation=Euler(pitch=-0.25, yaw=math.pi / 2),
            ),
            velocity=(0, 400, 150), boost=carl.BOOST_MAX,
        ),
        ball=BallState(
            position=(Uniform(-1050, -950), Uniform(-700, -600), 850),
            velocity=(0, 300, 200),
        ),
    )


def _v2_pack() -> TrainingPack:
    """Specific event sequences and original v2 starts, retained for checkpoints."""
    exercises = [
        Exercise(
            name="Aerial touch",
            initial=_ground_start(Uniform(750, 950)),
            stages=[
                # Leave the floor, establish height, accelerate toward the ball,
                # then make a useful nose/roof contact while both bodies are high.
                Stage(LeaveSurface(ArenaSurface.FLOOR), within_ticks=_ticks(0.9)),
                Stage(StateCondition("car_height", minimum=350), within_ticks=_ticks(1.2),
                      from_exercise_start=True, fail_on=_airborne_failures()),
                Stage(StateCondition("car_speed", minimum=800), within_ticks=_ticks(1.6),
                      from_exercise_start=True, fail_on=_airborne_failures(300)),
                Stage(BallTouch(SETUP_CONTACTS, min_speed_gain=60), within_ticks=_ticks(2.4),
                      from_exercise_start=True, fail_on=_touch_failures(350, 600)),
            ],
            fail_on=[Demoed()],
        ),
        Exercise(
            name="Fast aerial",
            initial=_ground_start(Uniform(1100, 1300)),
            stages=[
                Stage(Jump(), within_ticks=_ticks(0.65)),
                # A released/repressed neutral jump, not a directional flip.
                Stage(DoubleJump(), min_ticks=_ticks(0.08), within_ticks=_ticks(0.7),
                      fail_on=_airborne_failures()),
                Stage(StateCondition("car_height", minimum=950), within_ticks=_ticks(1.6),
                      from_exercise_start=True, fail_on=_airborne_failures()),
                Stage(BallTouch(SETUP_CONTACTS, min_speed_gain=100), within_ticks=_ticks(2.6),
                      from_exercise_start=True, fail_on=_touch_failures(750, 900)),
            ],
            fail_on=[Demoed(), Dodge()],
        ),
        Exercise(
            name="Aerial shot",
            initial=_air_start(650, 950, shot=True),
            stages=[
                Stage(BallTouch(SETUP_CONTACTS, min_speed_gain=250), within_ticks=_ticks(1.8),
                      fail_on=_touch_failures(450, 650)),
                # Persistent checkpoints validate power and forward progress
                # without demanding a second touch from an already-good shot.
                Stage(StateCondition("ball_speed", minimum=1400), within_ticks=_ticks(0.25),
                      fail_on=_ball_flight_failures()),
                Stage(StateCondition("ball_y", minimum=2000), within_ticks=_ticks(3.5),
                      from_exercise_start=True, fail_on=_ball_flight_failures()),
                Stage(Goal(), within_ticks=_ticks(5.5), from_exercise_start=True,
                      fail_on=_ball_flight_failures()),
            ],
            fail_on=[Demoed()],
        ),
        Exercise(
            name="Wall aerial",
            initial=InitialState(
                ego=CarState(
                    placement=OnSurface(
                        surface=ArenaSurface.SIDE_WALL, side=-1,
                        position=(Uniform(-1900, -1700), Uniform(450, 550)),
                        heading=Uniform(0.55, 0.75),
                    ),
                    velocity=(0, 500, 350), boost=carl.BOOST_MAX,
                ),
                ball=BallState(
                    position=(Uniform(-3300, -3100), Uniform(-1200, -1000), Uniform(1000, 1150)),
                    velocity=(100, 200, 300),
                ),
            ),
            stages=[
                Stage(LeaveSurface(ArenaSurface.SIDE_WALL), within_ticks=_ticks(0.9)),
                # Clear the wall inward, rather than falling down its face.
                # Mirroring turns this X lower bound into the opposite upper bound.
                Stage(StateCondition("car_x", minimum=-3500), within_ticks=_ticks(0.85),
                      fail_on=_airborne_failures(250)),
                Stage(StateCondition("car_height", minimum=700), within_ticks=_ticks(1.5),
                      from_exercise_start=True, fail_on=_airborne_failures(300)),
                Stage(BallTouch(SETUP_CONTACTS, min_speed_gain=150), within_ticks=_ticks(2.8),
                      from_exercise_start=True, fail_on=_touch_failures(500, 650)),
                Stage(Goal(), within_ticks=_ticks(6.5), from_exercise_start=True,
                      fail_on=_ball_flight_failures()),
            ],
            fail_on=[Demoed()],
        ),
        Exercise(
            name="Ceiling aerial",
            initial=InitialState(
                ego=CarState(
                    placement=OnSurface(
                        surface=ArenaSurface.CEILING,
                        position=(Uniform(-1000, -800), Uniform(-1700, -1500)),
                        heading=-math.pi / 2,
                    ),
                    velocity=(0, 350, 0), boost=carl.BOOST_MAX,
                ),
                ball=BallState(
                    position=(Uniform(-1000, -800), Uniform(-800, -600), Uniform(1150, 1350)),
                    velocity=(0, 200, 100),
                ),
            ),
            stages=[
                # Drop without jumping, preserving the ceiling departure's flip.
                Stage(LeaveSurface(ArenaSurface.CEILING), within_ticks=_ticks(0.8),
                      fail_on=[Jump(), DoubleJump(), Dodge()]),
                Stage(StateCondition("car_height", maximum=1600), within_ticks=_ticks(1.3),
                      from_exercise_start=True, fail_on=[*_airborne_failures(), Jump(), DoubleJump(), Dodge()]),
                Stage(BallTouch(SETUP_CONTACTS, min_speed_gain=100), within_ticks=_ticks(2.8),
                      from_exercise_start=True, fail_on=[*_touch_failures(500, 650), Jump(), DoubleJump(), Dodge()]),
                Stage(Goal(), within_ticks=_ticks(6.5), from_exercise_start=True,
                      fail_on=_ball_flight_failures()),
            ],
            fail_on=[Demoed()],
        ),
        Exercise(
            name="Air dribble",
            initial=_dribble_start(),
            stages=[
                # Soft nose/roof catch, pitch the nose under the ball, then
                # sustain two spaced nose pushes. Negative catch gain permits
                # cushioning a falling ball instead of forcing an acceleration.
                Stage(BallTouch(SETUP_CONTACTS, min_speed_gain=-300, max_speed_gain=200),
                      within_ticks=_ticks(1.8), fail_on=_touch_failures(350, 550)),
                Stage(BallTouch(ContactSurface.FRONT, min_speed_gain=0, max_speed_gain=250),
                      min_ticks=_ticks(0.2), within_ticks=_ticks(1.2), fail_on=_touch_failures(300, 500)),
                Stage(BallTouch(ContactSurface.FRONT, min_speed_gain=-50, max_speed_gain=250),
                      min_ticks=_ticks(0.2), within_ticks=_ticks(1.2), fail_on=_touch_failures(300, 500)),
                Stage(StateCondition("ball_y", minimum=1500), within_ticks=_ticks(4.5),
                      from_exercise_start=True, fail_on=_airborne_failures(200, 300)),
                Stage(Goal(), within_ticks=_ticks(7), from_exercise_start=True,
                      fail_on=_airborne_failures(100)),
            ],
            fail_on=[Demoed(), Dodge()],
        ),
    ]

    return TrainingPack(name="Mirrored aerial maneuvers v2", exercises=exercises).with_mirrors()


def aerial_pack(version: int = 3) -> TrainingPack:
    """Six mirrored skills; v3 raises every v2 ball start by another 250 units.

    Car states, launch velocities, and stage requirements are unchanged. Earlier
    versions remain available explicitly for checkpoint playback and resume.
    """
    if version == 1:
        if __package__:
            from .legacy_pack import aerial_pack as legacy_pack
        else:
            from legacy_pack import aerial_pack as legacy_pack
        return legacy_pack()
    if version not in (2, 3):
        raise ValueError("aerial pack version must be 1, 2, or 3")
    pack = _v2_pack()
    if version == 3:
        pack.name = "Mirrored aerial maneuvers v3"
        for exercise in pack.exercises:
            x, y, height = exercise.initial.ball.position
            raised = (Uniform(height.low + BALL_HEIGHT_INCREASE, height.high + BALL_HEIGHT_INCREASE)
                      if isinstance(height, Uniform) else height + BALL_HEIGHT_INCREASE)
            exercise.initial.ball.position = (x, y, raised)

    return pack


PACK = aerial_pack()
