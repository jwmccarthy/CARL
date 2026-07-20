from itertools import product

import torch as th
import torch.nn as nn


ACTION_NVECS = (3, 3, 3, 2, 2, 3, 2)
SELF_BOOST_INDEX = 9 + 15
SELF_ON_GROUND_INDEX = 9 + 16
SELF_HAS_FLIPPED_INDEX = 9 + 18
SELF_HAS_DOUBLE_JUMPED_INDEX = 9 + 19


class CARLActionCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        actions = th.tensor(
            list(product(*(range(size) for size in ACTION_NVECS))),
            dtype=th.int64,
        )
        multipliers = []
        for index in range(len(ACTION_NVECS)):
            multiplier = 1
            for size in ACTION_NVECS[index + 1 :]:
                multiplier *= size
            multipliers.append(multiplier)

        self.register_buffer("actions", actions, persistent=False)
        self.register_buffer(
            "multipliers",
            th.tensor(multipliers, dtype=th.int64),
            persistent=False,
        )

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (len(ACTION_NVECS),)

    def mask(self, observation: th.Tensor) -> th.Tensor:
        on_ground = observation[..., SELF_ON_GROUND_INDEX].bool()
        has_boost = observation[..., SELF_BOOST_INDEX].gt(0)
        has_flipped = observation[..., SELF_HAS_FLIPPED_INDEX].bool()
        has_double_jumped = observation[..., SELF_HAS_DOUBLE_JUMPED_INDEX].bool()

        pitch = self.actions[:, 1].ne(0)
        reverse = self.actions[:, 2].eq(2)
        powerslide = self.actions[:, 3].ne(0)
        boost = self.actions[:, 4].ne(0)
        air_roll = self.actions[:, 5].ne(0)
        jump = self.actions[:, 6].ne(0)

        jump_available = on_ground | (~has_flipped & ~has_double_jumped)
        incompatible = (
            (on_ground[..., None] & (pitch | air_roll))
            | (~on_ground[..., None] & (reverse | powerslide))
            | (~has_boost[..., None] & boost)
            | (~jump_available[..., None] & jump)
        )
        return ~incompatible

    def encode(self, action: th.Tensor) -> th.Tensor:
        return (action.to(th.int64) * self.multipliers).sum(dim=-1)

    def decode(self, index: th.Tensor) -> th.Tensor:
        return self.actions[index]


__all__ = ["ACTION_NVECS", "CARLActionCodec"]
