import torch as th
import torch.nn as nn

import carl
from carl import ACTION_NVECS


class CARLActionCodec(nn.Module):
    
    @property
    def action_shape(self) -> tuple[int, ...]:
        return (len(ACTION_NVECS),)

    def mask(self, observation: th.Tensor) -> th.Tensor:
        on_ground = observation[..., carl.OBS_EGO_ON_GROUND_INDEX].bool()
        has_boost = observation[..., carl.OBS_EGO_BOOST_INDEX].gt(0)

        flip_available = observation[..., carl.OBS_EGO_FLIP_INDEX].bool()
        jump_available = on_ground | (flip_available & (
            observation[..., carl.OBS_EGO_DODGE_TIME_INDEX] > carl.PHYS_DT
        ))

        mask = th.ones(
            (*observation.shape[:-1], carl.ACTION_LOGITS),
            dtype=th.bool,
            device=observation.device,
        )
        
        mask[..., carl.ACTION_PITCH_LOGITS] = ~on_ground[..., None]
        mask[..., carl.ACTION_POWERSLIDE_LOGIT] = on_ground
        mask[..., carl.ACTION_BOOST_LOGIT] = has_boost
        mask[..., carl.ACTION_AIR_ROLL_LOGITS] = ~on_ground[..., None]
        mask[..., carl.ACTION_JUMP_LOGIT] = jump_available

        return mask


__all__ = ["ACTION_NVECS", "CARLActionCodec"]
