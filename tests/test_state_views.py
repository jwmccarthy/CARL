import carl
import torch

from carl.gymnasium.state import CARLBall, CARLCar, CARLCars


def assert_physics_view(raw, view, start, end):
    assert torch.equal(view, raw[..., start:end])
    view.fill_(-1)
    assert (raw[..., start:end] == -1).all()


def test_physics_views_share_storage_and_preserve_layout():
    ball = torch.arange(2 * carl.OBS_BALL, dtype=torch.float32).reshape(2, carl.OBS_BALL)
    car = torch.arange(2 * carl.OBS_PER_CAR, dtype=torch.float32).reshape(2, carl.OBS_PER_CAR)
    cars = torch.arange(6 * carl.OBS_PER_CAR, dtype=torch.float32).reshape(2, 3, carl.OBS_PER_CAR)

    for raw, state in (
        (ball, CARLBall.from_tensor(ball)),
        (car, CARLCar.from_tensor(car)),
        (cars, CARLCars.from_tensor(cars, 3)),
    ):
        assert_physics_view(raw, state.position, carl.OBS_POS, carl.OBS_VEL)
        assert_physics_view(raw, state.velocity, carl.OBS_VEL, carl.OBS_ANG)
        assert_physics_view(raw, state.angular_velocity, carl.OBS_ANG, carl.OBS_FORWARD)


def test_cars_inherit_car_fields_and_ego():
    raw = torch.arange(2 * 3 * carl.OBS_PER_CAR, dtype=torch.float32).reshape(
        2, 3, carl.OBS_PER_CAR
    )
    cars = CARLCars.from_tensor(raw, 3)

    assert cars.n_cars == 3
    assert torch.equal(cars.forward, raw[..., carl.OBS_FORWARD:carl.OBS_UP])
    assert torch.equal(cars.boost, raw[..., carl.OBS_BOOST])
    assert torch.equal(cars.ego.position, raw[:, 0, carl.OBS_POS:carl.OBS_VEL])
    assert torch.equal(cars.on_ground, raw[..., carl.OBS_ON_GROUND].bool())
