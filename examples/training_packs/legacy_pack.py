"""Original aerial curriculum, retained for v1 checkpoint playback/resume."""

import math

import carl
from carl.events import (
    ArenaSurface, BallTouch, ContactSurface, Demoed, DoubleJump, Goal, Jump,
    LeaveSurface, StateCondition, SurfaceContact,
)
from carl.training import (
    BallState, CarState, Euler, Exercise, InitialState, InAir, OnSurface, Stage,
    TrainingPack, Uniform,
)


def _ticks(seconds: float) -> int:
    return round(seconds * carl.PHYS_TICKS_PER_SECOND)


def _airborne_failures() -> list[SurfaceContact | StateCondition]:
    return [
        SurfaceContact(entity="car"),
        StateCondition("car_height", maximum=carl.CAR_REST_Z + 2),
        StateCondition("ball_height", maximum=carl.BALL_REST_Z + 2),
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
            velocity=(0, Uniform(150, 250), Uniform(100, 200)),
        ),
    )


def _air_start(
    car_height:  float,
    ball_height: float,
) -> InitialState:
    return InitialState(
        ego=CarState(
            placement=InAir(
                position=(Uniform(-1050, -950), Uniform(-900, -800), car_height),
                orientation=Euler(pitch=-0.15, yaw=math.pi / 2),
            ),
            velocity=(0, 500, 100),
            boost=carl.BOOST_MAX,
        ),
        ball=BallState(
            position=(Uniform(-1050, -950), Uniform(-350, -250), ball_height),
            velocity=(0, 250, 50),
        ),
    )


def aerial_pack() -> TrainingPack:
    """Six skills, twelve independently sampled original/mirrored exercises.

    Starts progress from ground takeoffs to airborne control, wall launches,
    ceiling drops and repeated aerial touches. Only stage completions are
    rewarded by the PPO example; contacts on the same physics window cannot
    collect multiple stages. Floor/arena failures apply after takeoff, never
    to a ground start. Goal detection suppresses reset-ball height failures.
    """
    touch = BallTouch(ContactSurface.FRONT, min_speed_gain=100)
    exercises = [
        Exercise(
            name="Aerial touch",
            initial=_ground_start(Uniform(600, 800)),
            stages=[
                Stage(LeaveSurface(ArenaSurface.FLOOR), within_ticks=_ticks(2)),
                Stage(StateCondition("car_height", minimum=400), within_ticks=_ticks(3), fail_on=_airborne_failures()),
                Stage(BallTouch(), within_ticks=_ticks(3), fail_on=_airborne_failures()),
            ],
            fail_on=[Demoed()],
        ),
        Exercise(
            name="Fast aerial",
            initial=_ground_start(Uniform(1000, 1200)),
            stages=[
                Stage(Jump(), within_ticks=_ticks(2)),
                Stage(DoubleJump(), min_ticks=_ticks(0.1), within_ticks=_ticks(1), fail_on=_airborne_failures()),
                Stage(StateCondition("car_height", minimum=850), within_ticks=_ticks(3), fail_on=_airborne_failures()),
                Stage(touch, within_ticks=_ticks(3), fail_on=_airborne_failures()),
            ],
            fail_on=[Demoed()],
        ),
        Exercise(
            name="Aerial shot",
            initial=_air_start(650, 850),
            stages=[
                Stage(touch, within_ticks=_ticks(3), fail_on=_airborne_failures()),
                Stage(Goal(), within_ticks=_ticks(6), fail_on=_airborne_failures()),
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
                    position=(Uniform(-3300, -3100), Uniform(-1200, -1000), Uniform(850, 1000)),
                    velocity=(100, 200, 150),
                ),
            ),
            stages=[
                Stage(LeaveSurface(ArenaSurface.SIDE_WALL), within_ticks=_ticks(2)),
                Stage(touch, within_ticks=_ticks(3), fail_on=_airborne_failures()),
                Stage(Goal(), within_ticks=_ticks(6), fail_on=_airborne_failures()),
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
                    position=(Uniform(-1000, -800), Uniform(-800, -600), Uniform(1000, 1200)),
                    velocity=(0, 200, -100),
                ),
            ),
            stages=[
                Stage(LeaveSurface(ArenaSurface.CEILING), within_ticks=_ticks(2)),
                Stage(BallTouch(min_speed_gain=100), within_ticks=_ticks(3), fail_on=_airborne_failures()),
                Stage(Goal(), within_ticks=_ticks(6), fail_on=_airborne_failures()),
            ],
            fail_on=[Demoed()],
        ),
        Exercise(
            name="Air dribble",
            initial=_air_start(550, 750),
            stages=[
                Stage(BallTouch(min_speed_gain=25), within_ticks=_ticks(3), fail_on=_airborne_failures()),
                Stage(BallTouch(min_speed_gain=25), min_ticks=_ticks(0.2),
                      within_ticks=_ticks(2), fail_on=_airborne_failures()),
                Stage(BallTouch(min_speed_gain=25), min_ticks=_ticks(0.2),
                      within_ticks=_ticks(2), fail_on=_airborne_failures()),
                Stage(Goal(), within_ticks=_ticks(6), fail_on=_airborne_failures()),
            ],
            fail_on=[Demoed()],
        ),
    ]

    return TrainingPack(name="Mirrored aerial maneuvers", exercises=exercises).with_mirrors()
