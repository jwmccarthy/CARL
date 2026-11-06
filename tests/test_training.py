import unittest

from carl.events import ContactSurface
from carl.training import (
    BallState, BallTouch, Box2D, Box3D, CarState, Euler, Exercise, FlipReset,
    SurfaceContact, InitialState, InAir, LeaveSurface, OnSurface, Stage, TrainingPack,
    Uniform,
)


class TrainingDefinitionTests(unittest.TestCase):

    def setUp(self):
        self.initial = InitialState(
            ego = CarState(
                placement = OnSurface(position=Box2D((-500, -1000), (500, 0))),
                boost     = Uniform(60, 100),
            ),
            ball = BallState(
                position = Box3D((-300, 500, 600), (300, 900, 800)),
                velocity = (0, Uniform(200, 500), 0),
            ),
        )

    def test_pack_definition(self):
        exercise = Exercise(
            name    = "Aerial touch",
            initial = self.initial,
            stages  = [
                Stage(LeaveSurface(), within_ticks=120),
                Stage(
                    BallTouch(surface=ContactSurface.FRONT, min_height=400),
                    within_ticks = 240,
                    fail_on      = [SurfaceContact()],
                ),
            ],
        )
        pack = TrainingPack(name="Aerials", exercises=[exercise])

        self.assertEqual(pack.exercises, [exercise])
        self.assertEqual(exercise.stages[1].fail_on, [SurfaceContact()])
        self.assertEqual(exercise.stages[1].within_ticks, 240)
        self.assertEqual(exercise.initial.ego.boost, Uniform(60, 100))

    def test_airborne_physics(self):
        car = CarState(
            placement        = InAir((0, 0, 800), Euler(pitch=Uniform(-0.5, 0.5))),
            velocity         = (100, 200, -300),
            angular_velocity = (Uniform(-1, 1), 0, 0),
        )

        self.assertEqual(car.placement.position, (0, 0, 800))
        self.assertEqual(car.velocity, (100, 200, -300))
        self.assertEqual(car.angular_velocity[0], Uniform(-1, 1))

    def test_optional_stage_settings(self):
        stage = Stage(FlipReset())

        self.assertIsNone(stage.within_ticks)
        self.assertEqual(stage.fail_on, [])

    def test_mutable_definitions_and_independent_defaults(self):
        first = Stage(FlipReset())
        second = Stage(FlipReset())

        first.within_ticks = 120
        first.fail_on.append(SurfaceContact())

        self.assertEqual(first.within_ticks, 120)
        self.assertEqual(second.fail_on, [])

    def test_basic_validation(self):
        for ticks in (0, -1):
            with self.subTest(ticks=ticks), self.assertRaises(ValueError):
                Stage(BallTouch(), within_ticks=ticks)
        with self.assertRaises(ValueError):
            Uniform(10, 0)
        with self.assertRaises(ValueError):
            Exercise(name="Empty", initial=self.initial, stages=[])
        with self.assertRaises(ValueError):
            TrainingPack(name="Empty", exercises=[])
