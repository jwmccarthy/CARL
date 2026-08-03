import torch as th
import torch.nn as nn


ACTION_NVECS = (3, 3, 3, 2, 2, 3, 2)
ACTION_LOGITS = sum(ACTION_NVECS)
SELF_BOOST_INDEX = 9 + 15
SELF_ON_GROUND_INDEX = 9 + 16
SELF_HAS_FLIPPED_INDEX = 9 + 18
SELF_HAS_DOUBLE_JUMPED_INDEX = 9 + 19


class CARLActionCodec(nn.Module):
    
    @property
    def action_shape(self) -> tuple[int, ...]:
        return (len(ACTION_NVECS),)

    def mask(self, observation: th.Tensor) -> th.Tensor:
        on_ground = observation[..., SELF_ON_GROUND_INDEX].bool()
        has_boost = observation[..., SELF_BOOST_INDEX].gt(0)
        has_flipped = observation[..., SELF_HAS_FLIPPED_INDEX].bool()
        has_double_jumped = observation[..., SELF_HAS_DOUBLE_JUMPED_INDEX].bool()
        jump_available = on_ground | (~has_flipped & ~has_double_jumped)

        mask = th.ones(
            (*observation.shape[:-1], ACTION_LOGITS),
            dtype=th.bool,
            device=observation.device,
        )
        
        mask[..., 4:6] = ~on_ground[..., None]    # Pitch
        mask[..., 10] = on_ground                 # Powerslide
        mask[..., 12] = has_boost                 # Boost
        mask[..., 14:16] = ~on_ground[..., None]  # Air roll
        mask[..., 17] = jump_available            # Jump

        return mask


__all__ = ["ACTION_NVECS", "CARLActionCodec"]
