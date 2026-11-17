from __future__ import annotations

from dataclasses import replace

import torch as th

from carl import SOCCAR_BACK_WALL, SOCCAR_CEILING, SOCCAR_SIDE_WALL, _carl
from carl.gymnasium.state import CARLBall, CARLCars, CARLResetState

from .definitions import Exercise, TrainingPack
from .initial import (
    Box2D, Box3D, Euler, InAir, InitialState, OnSurface, Scalar, Uniform,
    Vector2, Vector3,
)
from ..events import ArenaSurface


PhysicsValue = Scalar | Vector2 | Vector3 | Euler | th.Tensor | None


def _check_side(side: int) -> None:
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")


def _surface_axes(
    surface: ArenaSurface,
    side:    int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if surface == ArenaSurface.FLOOR:
        return (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    if surface == ArenaSurface.CEILING:
        return (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)
    if surface == ArenaSurface.SIDE_WALL:
        return (-float(side), 0.0, 0.0), (0.0, 1.0, 0.0)
    if surface == ArenaSurface.BACK_WALL:
        return (0.0, -float(side), 0.0), (1.0, 0.0, 0.0)

    raise ValueError("unsupported arena surface")


def _surface_position(
    batch:    ResetBatch,
    surface:  ArenaSurface,
    position: Vector2 | th.Tensor,
    side:     int,
    rest:     float,
) -> th.Tensor:
    tangent = batch._vector(position, dimensions=2)
    if surface == ArenaSurface.FLOOR:
        depth = tangent.new_full((batch.count, 1), rest)

        return th.cat((tangent, depth), dim=-1)
    if surface == ArenaSurface.CEILING:
        depth = tangent.new_full((batch.count, 1), SOCCAR_CEILING - rest)

        return th.cat((tangent, depth), dim=-1)
    if surface == ArenaSurface.SIDE_WALL:
        x = tangent.new_full((batch.count, 1), side * (SOCCAR_SIDE_WALL - rest))

        return th.cat((x, tangent), dim=-1)
    if surface == ArenaSurface.BACK_WALL:
        y = tangent.new_full((batch.count, 1), side * (SOCCAR_BACK_WALL - rest))

        return th.cat((tangent[:, :1], y, tangent[:, 1:]), dim=-1)

    raise ValueError("unsupported arena surface")


def _surface_forward(
    batch:   ResetBatch,
    normal:  Vector3,
    tangent: Vector3,
    heading: Scalar | th.Tensor,
) -> th.Tensor:
    angle = batch._scalar(heading)
    normal = batch._vector(normal)
    tangent = batch._vector(tangent)
    bitangent = th.linalg.cross(normal, tangent, dim=-1)
    return tangent * angle.cos()[:, None] + bitangent * angle.sin()[:, None]


class ResetSampler:

    def __init__(
        self,
        n_sim:  int,
        seed:   int = 0,
        device: str | th.device = "cuda:0",
    ) -> None:
        self.seed = seed & 0xFFFFFFFF
        self.device = th.device(device)
        self.generator = th.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        self.reset_counts = th.zeros(n_sim, dtype=th.int64, device=self.device)
        # Resolve aliases such as "cuda" to "cuda:0" for reset-mask validation.
        self.device = self.reset_counts.device

    def select(self, selected: th.Tensor) -> "ResetBatch":
        if selected.device != self.reset_counts.device or selected.ndim != 1:
            raise ValueError("reset selection must be one-dimensional on the sampler device")

        if selected.dtype == th.bool:
            if selected.shape != self.reset_counts.shape:
                raise ValueError("reset mask must have one entry per simulation")
            indices = selected.nonzero(as_tuple=True)[0]
        elif selected.dtype == th.int64:
            indices = selected
        else:
            raise TypeError("reset selection must be a bool mask or int64 indices")

        counts = self.reset_counts[indices].clone()
        self.reset_counts[indices] += 1

        return ResetBatch(indices, counts, self.seed, self.generator)


class ExerciseResetProvider:

    def __init__(
        self,
        exercise: Exercise,
        n_sim:    int,
        seed:     int = 0,
        device:   str | th.device = "cuda:0",
    ) -> None:
        if n_sim < 1:
            raise ValueError("n_sim must be positive")

        self.exercise = exercise
        self.sampler = ResetSampler(n_sim, seed=seed, device=device)
        self.indices = th.arange(n_sim, dtype=th.int64, device=device)

    def __call__(self, reset_mask: th.Tensor) -> CARLResetState:
        if reset_mask.dtype != th.bool or reset_mask.shape != self.sampler.reset_counts.shape:
            raise ValueError("reset_mask must be a per-simulation boolean mask")

        if reset_mask.device != self.sampler.device:
            raise ValueError("reset_mask must be on the sampler device")

        self.sampler.reset_counts.add_(reset_mask)
        batch = ResetBatch(self.indices, self.sampler.reset_counts, self.sampler.seed, self.sampler.generator)

        return replace(batch.sample(self.exercise.initial), simulation_indices=None, reset_mask=reset_mask)


class TrainingPackResetProvider:

    """Sample exercises randomly, or cycle every simulation through all variants.

    ``balanced=True`` uses each simulation's own reset counter, so partial
    resets cannot skip a mirrored side or favor a shorter exercise.
    """

    def __init__(
        self,
        pack:     TrainingPack,
        n_sim:    int,
        seed:     int = 0,
        device:   str | th.device = "cuda:0",
        *,
        balanced: bool = False,
    ) -> None:
        if n_sim < 1:
            raise ValueError("n_sim must be positive")

        if not pack.exercises:
            raise ValueError("a training pack needs at least one exercise")

        self.pack = pack
        self.balanced = balanced
        self.sampler = ResetSampler(n_sim, seed=seed, device=device)
        self.indices = th.arange(n_sim, dtype=th.int64, device=device)
        self.exercise_index = th.zeros(n_sim, dtype=th.int64, device=device)
        self._initial = _InitialStateTable(pack, self.sampler.device)

    def __call__(self, reset_mask: th.Tensor) -> CARLResetState:
        if reset_mask.dtype != th.bool or reset_mask.shape != self.exercise_index.shape:
            raise ValueError("reset_mask must be a per-simulation boolean mask")

        if reset_mask.device != self.sampler.device:
            raise ValueError("reset_mask must be on the sampler device")

        # Per-simulation cycling guarantees both mirrored variants are visited,
        # including when there are fewer simulations than exercises.
        if self.balanced:
            choices = (self.indices + self.sampler.reset_counts) % len(self.pack.exercises)
        else:
            choices = th.randint(
                len(self.pack.exercises), self.exercise_index.shape,
                generator = self.sampler.generator,
                device    = self.sampler.device,
            )
        self.exercise_index = th.where(reset_mask, choices, self.exercise_index)
        self.sampler.reset_counts.add_(reset_mask)

        state = self._initial.sample(self.exercise_index, self.indices, self.sampler.generator)
        return replace(state, simulation_indices=None, reset_mask=reset_mask)


class _InitialStateTable:
    """Compile initial distributions once, then sample all selected exercises together.

    Work per call is O(n_sim), not O(n_sim * exercises). Fixed-size masked rows
    avoid nonzero() and host synchronization even when no simulations reset.
    """

    def __init__(
        self,
        pack:   TrainingPack,
        device: th.device,
    ) -> None:
        rows, axes, surfaces, grounded = [], [], [], []

        def scalar(value: Scalar) -> tuple[float, float]:
            return (value.low, value.high) if isinstance(value, Uniform) else (value, value)

        def vector(
            value:      Vector2 | Vector3,
            dimensions: int = 3,
        ) -> list[tuple[float, float]]:
            if isinstance(value, (Box2D, Box3D)):
                result = list(zip(value.low, value.high))
            else:
                result = [scalar(component) for component in value]
            if len(result) != dimensions:
                raise ValueError(f"expected {dimensions} vector components")
            return result

        for exercise in pack.exercises:
            car, ball = exercise.initial.ego, exercise.initial.ball
            placement = car.placement
            surface = isinstance(placement, OnSurface)
            normal = tangent = (0.0, 0.0, 0.0)
            on_ground = False
            if surface:
                _check_side(placement.side)
                normal, tangent = _surface_axes(placement.surface, placement.side)
                first, second = vector(placement.position, 2)
                if placement.surface == ArenaSurface.FLOOR:
                    position = [first, second, scalar(_carl.CAR_REST_Z)]
                    on_ground = True
                elif placement.surface == ArenaSurface.CEILING:
                    position = [first, second, scalar(SOCCAR_CEILING - _carl.CAR_REST_Z)]
                elif placement.surface == ArenaSurface.SIDE_WALL:
                    position = [scalar(placement.side * (SOCCAR_SIDE_WALL - _carl.CAR_REST_Z)), first, second]
                else:
                    position = [first, scalar(placement.side * (SOCCAR_BACK_WALL - _carl.CAR_REST_Z)), second]
                angles = [scalar(0), scalar(0), scalar(placement.heading)]
            elif isinstance(placement, InAir):
                position = vector(placement.position)
                orientation = placement.orientation
                angles = [scalar(orientation.roll), scalar(orientation.pitch), scalar(orientation.yaw)]
            else:
                raise TypeError("unsupported car placement")
            rows.append(position + angles + vector(car.velocity) + vector(car.angular_velocity) + [scalar(car.boost)]
                        + vector(ball.position) + vector(ball.velocity) + vector(ball.angular_velocity))
            axes.append((*normal, *tangent))
            surfaces.append(surface)
            grounded.append(on_ground)

        bounds = th.tensor(rows, dtype=th.float32, device=device)
        self.minimum = bounds[..., 0].contiguous()
        self.extent = bounds[..., 1] - self.minimum
        self.axes = th.tensor(axes, dtype=th.float32, device=device)
        self.surface = th.tensor(surfaces, dtype=th.bool, device=device)
        self.grounded = th.tensor(grounded, dtype=th.float32, device=device)

    def sample(
        self,
        exercise:  th.Tensor,
        indices:   th.Tensor,
        generator: th.Generator,
    ) -> CARLResetState:
        count = exercise.numel()
        device = exercise.device
        noise = th.rand((count, self.minimum.shape[1]), generator=generator, device=device)
        values = self.minimum[exercise] + self.extent[exercise] * noise
        roll, pitch, yaw = values[:, 3:6].unbind(-1)
        cr, sr = roll.cos(), roll.sin()
        cp, sp = pitch.cos(), pitch.sin()
        cy, sy = yaw.cos(), yaw.sin()
        forward = th.stack((cy * cp, sy * cp, -sp), dim=-1)
        up = th.stack((cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr), dim=-1)
        normal, tangent = self.axes[exercise].split(3, dim=-1)
        surface_forward = tangent * cy[:, None] + th.linalg.cross(normal, tangent, dim=-1) * sy[:, None]
        surface = self.surface[exercise, None]

        cars = CARLCars.from_tensor(th.zeros((count, 1, _carl.OBS_PER_CAR), device=device), 1)
        ego = cars.ego
        ego.position.copy_(values[:, :3])
        ego.forward.copy_(th.where(surface, surface_forward, forward))
        ego.up.copy_(th.where(surface, normal, up))
        ego.velocity.copy_(values[:, 6:9])
        ego.angular_velocity.copy_(values[:, 9:12])
        ego.boost.copy_(values[:, 12])
        cars[:, 0, _carl.OBS_ON_GROUND] = self.grounded[exercise]
        ball = CARLBall.from_tensor(th.zeros((count, _carl.OBS_BALL), device=device))
        ball.position.copy_(values[:, 13:16])
        ball.velocity.copy_(values[:, 16:19])
        ball.angular_velocity.copy_(values[:, 19:22])
        internal = th.zeros((count, 1, _carl.CAR_INTERNAL_SIZE), device=device)
        internal[..., _carl.CAR_INT_ON_GROUND] = cars[..., _carl.OBS_ON_GROUND]

        return CARLResetState(simulation_indices=indices, ball=ball, cars=cars, car_internal_state=internal)


class ResetBatch:

    def __init__(
        self,
        simulation_indices: th.Tensor,
        reset_counts:       th.Tensor | None = None,
        seed:               int = 0,
        generator:          th.Generator | None = None,
    ) -> None:
        self.simulation_indices = simulation_indices
        self.device = simulation_indices.device
        self.count = simulation_indices.numel()
        self.seed = seed & 0xFFFFFFFF
        self.generator = generator

        if self.generator is None:
            self.generator = th.Generator(device=self.device)
            self.generator.manual_seed(self.seed)
        self.reset_counts = (
            th.zeros_like(simulation_indices) if reset_counts is None else reset_counts
        )
        self._ball = CARLBall.from_tensor(
            th.zeros((self.count, _carl.OBS_BALL), device=self.device, dtype=th.float32)
        )
        self._cars = CARLCars.from_tensor(
            th.zeros((self.count, 1, _carl.OBS_PER_CAR), device=self.device, dtype=th.float32), 1
        )
        self.ego = CarReset(self)
        self.ball = BallReset(self)
        self.ego.on_floor(position=(0, -1000), boost=100)
        self.ball.on_floor()

    def uniform(
        self,
        low:   float = 0.0,
        high:  float = 1.0,
        shape: tuple[int, ...] = (),
    ) -> th.Tensor:
        values = th.rand(
            (self.count, *shape),
            generator=self.generator,
            device=self.device,
            dtype=th.float32,
        )

        return low + (high - low) * values

    def choice(self, values: th.Tensor | list[float] | tuple[float, ...]) -> th.Tensor:
        values = th.as_tensor(values, device=self.device)

        if values.shape[0] == 0:
            raise ValueError("choice needs at least one value")

        indices = (self.uniform() * values.shape[0]).long()
        return values[indices]

    def _scalar(self, value: Scalar | th.Tensor) -> th.Tensor:
        if isinstance(value, Uniform):
            return self.uniform(value.low, value.high)

        if not isinstance(value, th.Tensor):
            return th.full((self.count,), value, device=self.device, dtype=th.float32)

        value = th.as_tensor(value, device=self.device, dtype=th.float32)

        return th.broadcast_to(value, (self.count,))

    def _vector(
        self,
        value:      Vector2 | Vector3 | th.Tensor,
        dimensions: int = 3,
    ) -> th.Tensor:
        if isinstance(value, (Box2D, Box3D)):
            low = self._vector(value.low, dimensions)
            high = self._vector(value.high, dimensions)

            return low + (high - low) * self.uniform(shape=(dimensions,))

        if isinstance(value, th.Tensor):
            value = value.to(device=self.device, dtype=th.float32)

            return th.broadcast_to(value, (self.count, dimensions))

        if len(value) != dimensions:
            raise ValueError(f"expected {dimensions} vector components")

        return th.stack([self._scalar(component) for component in value], dim=-1)

    def sample(self, initial: InitialState) -> CARLResetState:
        placement = initial.ego.placement
        if isinstance(placement, OnSurface):
            self.ego.on_surface(
                placement.surface, placement.position, placement.heading, placement.side
            )
        elif isinstance(placement, InAir):
            self.ego.in_air(placement.position, placement.orientation)
        else:
            raise TypeError("unsupported car placement")

        self.ego.set(
            velocity=initial.ego.velocity,
            angular_velocity=initial.ego.angular_velocity,
            boost=initial.ego.boost,
        )
        self.ball.set(
            position=initial.ball.position,
            velocity=initial.ball.velocity,
            angular_velocity=initial.ball.angular_velocity,
        )

        return self.build()

    def build(self) -> CARLResetState:
        internal = th.zeros(
            (self.count, 1, _carl.CAR_INTERNAL_SIZE), device=self.device, dtype=th.float32
        )
        internal[..., _carl.CAR_INT_ON_GROUND] = self._cars[..., _carl.OBS_ON_GROUND]

        return CARLResetState(
            simulation_indices = self.simulation_indices,
            ball               = self._ball,
            cars               = self._cars,
            car_internal_state = internal,
        )


class CarReset:

    def __init__(self, batch: ResetBatch) -> None:
        self._batch = batch
        self._car = batch._cars.ego

    def set(
        self,
        *,
        position:         Vector3 | th.Tensor | None = None,
        orientation:      Euler | None = None,
        velocity:         Vector3 | th.Tensor | None = None,
        angular_velocity: Vector3 | th.Tensor | None = None,
        boost:            Scalar | th.Tensor | None = None,
    ) -> CarReset:
        batch = self._batch

        if position is not None:
            self._car.position.copy_(batch._vector(position))
            self._car[..., _carl.OBS_ON_GROUND] = 0

        if orientation is not None:
            self._car[..., _carl.OBS_ON_GROUND] = 0
            roll = batch._scalar(orientation.roll)
            pitch = batch._scalar(orientation.pitch)
            yaw = batch._scalar(orientation.yaw)
            cr, sr = roll.cos(), roll.sin()
            cp, sp = pitch.cos(), pitch.sin()
            cy, sy = yaw.cos(), yaw.sin()

            self._car.forward.copy_(th.stack((cy * cp, sy * cp, -sp), dim=-1))
            up = th.stack((
                cy * sp * cr + sy * sr,
                sy * sp * cr - cy * sr,
                cp * cr,
            ), dim=-1)
            self._car.up.copy_(up)

        if velocity is not None:
            self._car.velocity.copy_(batch._vector(velocity))
        if angular_velocity is not None:
            self._car.angular_velocity.copy_(batch._vector(angular_velocity))
        if boost is not None:
            self._car.boost.copy_(batch._scalar(boost))

        return self

    def on_surface(
        self,
        surface:   ArenaSurface = ArenaSurface.FLOOR,
        position:  Vector2 | th.Tensor = (0, 0),
        heading:   Scalar | th.Tensor = 0.0,
        side:      int = 1,
        **physics: PhysicsValue,
    ) -> CarReset:
        _check_side(side)
        normal, tangent = _surface_axes(surface, side)
        position = _surface_position(self._batch, surface, position, side, _carl.CAR_REST_Z)
        forward = _surface_forward(self._batch, normal, tangent, heading)

        self._car.position.copy_(position)
        self._car.forward.copy_(forward)
        self._car.up.copy_(self._batch._vector(normal))
        self._car[..., _carl.OBS_ON_GROUND] = int(surface == ArenaSurface.FLOOR)

        return self.set(**physics)

    def on_floor(
        self,
        position:  Vector2 | th.Tensor = (0, 0),
        heading:   Scalar | th.Tensor = 0.0,
        **physics: PhysicsValue,
    ) -> CarReset:
        return self.on_surface(ArenaSurface.FLOOR, position, heading, **physics)

    def on_ceiling(
        self,
        position:  Vector2 | th.Tensor = (0, 0),
        heading:   Scalar | th.Tensor = 0.0,
        **physics: PhysicsValue,
    ) -> CarReset:
        return self.on_surface(ArenaSurface.CEILING, position, heading, **physics)

    def on_side_wall(
        self,
        position:  Vector2 | th.Tensor = (0, 0),
        heading:   Scalar | th.Tensor = 0.0,
        side:      int = 1,
        **physics: PhysicsValue,
    ) -> CarReset:
        return self.on_surface(ArenaSurface.SIDE_WALL, position, heading, side, **physics)

    def on_back_wall(
        self,
        position:  Vector2 | th.Tensor = (0, 0),
        heading:   Scalar | th.Tensor = 0.0,
        side:      int = 1,
        **physics: PhysicsValue,
    ) -> CarReset:
        return self.on_surface(ArenaSurface.BACK_WALL, position, heading, side, **physics)

    def in_air(
        self,
        position:    Vector3 | th.Tensor,
        orientation: Euler | None = None,
        **physics:   PhysicsValue,
    ) -> CarReset:
        orientation = Euler() if orientation is None else orientation
        return self.set(position=position, orientation=orientation, **physics)

    def local_vector(
        self,
        forward: Scalar | th.Tensor = 0.0,
        left:    Scalar | th.Tensor = 0.0,
        up:      Scalar | th.Tensor = 0.0,
    ) -> th.Tensor:
        batch = self._batch
        side = th.linalg.cross(self._car.up, self._car.forward, dim=-1)

        return (
            self._car.forward * batch._scalar(forward)[:, None]
            + side * batch._scalar(left)[:, None]
            + self._car.up * batch._scalar(up)[:, None]
        )

    def local_point(
        self,
        forward: Scalar | th.Tensor = 0.0,
        left:    Scalar | th.Tensor = 0.0,
        up:      Scalar | th.Tensor = 0.0,
    ) -> th.Tensor:
        return self._car.position + self.local_vector(forward, left, up)


class BallReset:

    def __init__(self, batch: ResetBatch) -> None:
        self._batch = batch
        self._ball = batch._ball

    def set(
        self,
        *,
        position:         Vector3 | th.Tensor | None = None,
        velocity:         Vector3 | th.Tensor | None = None,
        angular_velocity: Vector3 | th.Tensor | None = None,
    ) -> BallReset:
        values = (
            ("position",         position),
            ("velocity",         velocity),
            ("angular_velocity", angular_velocity),
        )

        for name, value in values:
            if value is not None:
                getattr(self._ball, name).copy_(self._batch._vector(value))

        return self

    def on_surface(
        self,
        surface:   ArenaSurface = ArenaSurface.FLOOR,
        position:  Vector2 | th.Tensor = (0, 0),
        side:      int = 1,
        **physics: PhysicsValue,
    ) -> BallReset:
        _check_side(side)
        position = _surface_position(self._batch, surface, position, side, _carl.BALL_REST_Z)
        return self.set(position=position, **physics)

    def on_floor(
        self,
        position:  Vector2 | th.Tensor = (0, 0),
        **physics: PhysicsValue,
    ) -> BallReset:
        return self.on_surface(ArenaSurface.FLOOR, position, **physics)

    def on_ceiling(
        self,
        position:  Vector2 | th.Tensor = (0, 0),
        **physics: PhysicsValue,
    ) -> BallReset:
        return self.on_surface(ArenaSurface.CEILING, position, **physics)

    def on_side_wall(
        self,
        position:  Vector2 | th.Tensor = (0, 0),
        side:      int = 1,
        **physics: PhysicsValue,
    ) -> BallReset:
        return self.on_surface(ArenaSurface.SIDE_WALL, position, side, **physics)

    def on_back_wall(
        self,
        position:  Vector2 | th.Tensor = (0, 0),
        side:      int = 1,
        **physics: PhysicsValue,
    ) -> BallReset:
        return self.on_surface(ArenaSurface.BACK_WALL, position, side, **physics)
