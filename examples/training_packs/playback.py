"""Checkpoint adapter for Goddard's training-pack viewer.

The attention policy consumes physical observations and the runtime's raw skill
features, not Goddard's legacy stage-one-hot MLP input. Playback does not restore
the optimizer, RNG, or training sampler, and supports checkpoints predating the
optional viewer metadata.
"""

from __future__ import annotations

from typing import Any

import torch as th

from carl.training import POLICY_FEATURE_DIM, TrainingPack, TrainingPackRuntime, policy_features
from .pack import aerial_pack
from .ppo import CHECKPOINT_FORMAT, Agent, PPOConfig, _autocast, _pack_signature, _plain


def checkpoint_pack(payload: dict[str, Any]) -> TrainingPack:
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("feature_dim") != POLICY_FEATURE_DIM:
        raise ValueError("incompatible attention PPO checkpoint or policy feature schema")
    for version in (3, 2, 1):
        pack = aerial_pack(version)
        if payload.get("pack_signature") == _pack_signature(pack):
            return pack
    raise ValueError("checkpoint uses a different training pack")


class PackPolicy(Agent):
    """The original attention network with the viewer's get_action interface."""

    def __init__(
        self,
        observation_size: int,
        config:           PPOConfig,
        runtime:          TrainingPackRuntime,
    ) -> None:
        super().__init__(observation_size, config)
        self.runtime = runtime
        self.window_size = config.window_size
        self.bf16 = config.bf16

    @th.inference_mode()
    def get_action(
        self,
        observation: th.Tensor,
        sample:      bool = False,
    ) -> th.Tensor:
        observation = _plain(observation)
        features = policy_features(self.runtime, K=self.window_size)
        with _autocast(observation.device, self.bf16):
            actions, _, _, _ = self.get_action_and_value(
                observation, features.window, features.failures, deterministic=not sample,
            )

        return actions


def checkpoint_policy(
    payload:          dict[str, Any],
    runtime:          TrainingPackRuntime,
    observation_size: int,
) -> PackPolicy:
    # Checkpoints predating mixed precision were trained and evaluated in FP32.
    config = PPOConfig(**dict(payload["config"], bf16=payload["config"].get("bf16", False)))
    expected_size = payload["agent"]["observation.0.weight"].shape[1]
    if observation_size != expected_size or payload.get("observation_size", expected_size) != expected_size:
        raise ValueError("checkpoint observation layout differs from the viewer environment")
    policy = PackPolicy(observation_size, config, runtime).to(runtime.active.device)
    policy.load_state_dict(payload["agent"], strict=True)
    policy.eval()

    return policy
