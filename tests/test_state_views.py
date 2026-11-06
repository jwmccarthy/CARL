import carl
import torch

from carl.gymnasium.state import CARLBall, CARLCar, CARLCars


def test_physics_views_share_storage_and_preserve_layout():
    for shape, wrap in (
        ((2, carl.OBS_BALL), CARLBall.from_tensor),
        ((2, carl.OBS_PER_CAR), CARLCar.from_tensor),
        ((2, 3, carl.OBS_PER_CAR), lambda x: CARLCars.from_tensor(x, 3)),
    ):
        raw = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
        state = wrap(raw)
        for name, start, end in (
            ("position", carl.OBS_POS, carl.OBS_VEL),
            ("velocity", carl.OBS_VEL, carl.OBS_ANG),
            ("angular_velocity", carl.OBS_ANG, carl.OBS_FORWARD),
        ):
            view = getattr(state, name)
            assert torch.equal(view, raw[..., start:end])
            view.fill_(-1)
            assert (raw[..., start:end] == -1).all()


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
