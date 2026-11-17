"""GPU PPO conditioned on an attention encoding of the current training skill.

Run from the repository root with:
    uv run --extra training python -m examples.training_packs.ppo --n-sim 1024

Rollouts store raw skill features, not detached embeddings: the attention
encoder is optimized jointly with the actor and critic. No exercise ID enters
the network. Checkpoints resume learning but restart physical episodes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import ExitStack
from copy import copy
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import torch as th
from torch import nn
from torch.nn import functional as F

import carl
from carl.events import CarlEvents
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import CARLActionCodec
from carl.gymnasium.state import CarlState, StepContext
from carl.training import (
    POLICY_FEATURE_DIM, POLICY_FEATURE_SLICES, CompiledTrainingPack, PolicyFeatures,
    TrainingPack, TrainingPackResetProvider, TrainingPackRuntime,
    TrainingPackTerminationManager, TrainingStep, policy_features,
)

if __package__:
    from .pack import aerial_pack
    from .reporting import log_metrics, open_writer, progress_bar
else:
    from pack import aerial_pack
    from reporting import log_metrics, open_writer, progress_bar


CHECKPOINT_FORMAT = "carl-training-pack-ppo-v1"
EPSILON = 1e-8


@dataclass
class PPOConfig:
    n_sim:                   int   = 1024
    total_timesteps:         int   = 10_000_000
    rollout_steps:           int   = 128
    minibatch_size:          int   = 16_384
    update_epochs:           int   = 4
    frameskip:               int   = 8
    max_ticks:               int   = 15 * carl.PHYS_TICKS_PER_SECOND
    window_size:             int   = 3
    pack_version:            int   = 3
    embedding_size:          int   = 64
    attention_heads:         int   = 4
    attention_layers:        int   = 2
    hidden_size:             int   = 256
    learning_rate:           float = 3e-4
    gamma:                   float = 0.99
    gae_lambda:              float = 0.95
    clip_coef:               float = 0.2
    entropy_coef:            float = 0.01
    value_coef:              float = 0.5
    max_grad_norm:           float = 0.5
    target_kl:               float = 0.03
    stage_reward_base:       float = 1.0
    stage_reward_multiplier: float = 2.0
    bf16:                    bool  = True
    compile:                 bool  = True
    seed:                    int   = 0
    save_every:              int   = 10
    output_dir:              str   = "runs/training-pack-ppo"
    resume:                  str | None = None

    def __post_init__(self) -> None:
        for name in (
            "n_sim", "total_timesteps", "rollout_steps", "minibatch_size", "update_epochs",
            "frameskip", "max_ticks", "window_size", "pack_version", "embedding_size", "attention_heads",
            "attention_layers", "hidden_size", "save_every",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.embedding_size % self.attention_heads:
            raise ValueError("embedding_size must be divisible by attention_heads")
        if self.pack_version not in (1, 2, 3):
            raise ValueError("pack_version must be 1, 2, or 3")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_coef < 1:
            raise ValueError("invalid discount, GAE, or PPO clipping coefficient")
        if min(self.entropy_coef, self.value_coef, self.target_kl) < 0:
            raise ValueError("loss coefficients and target_kl must be nonnegative")
        if self.stage_reward_base <= 0 or self.stage_reward_multiplier < 1:
            raise ValueError("stage rewards require a positive base and multiplier >= 1")
        if not isinstance(self.bf16, bool) or not isinstance(self.compile, bool):
            raise ValueError("bf16 and compile must be booleans")


class SkillEncoder(nn.Module):
    """Ordered stage tokens and unordered failure tokens, pooled by attention."""

    def __init__(
        self,
        window_size:    int,
        embedding_size: int,
        heads:          int,
        layers:         int,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(POLICY_FEATURE_DIM, embedding_size)
        self.pool = nn.Parameter(th.zeros(1, 1, embedding_size))
        self.positions = nn.Parameter(th.empty(1, window_size, embedding_size))
        self.roles = nn.Parameter(th.empty(2, embedding_size))
        nn.init.normal_(self.positions, std=0.02)
        nn.init.normal_(self.roles, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=heads, dim_feedforward=4 * embedding_size,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(embedding_size), enable_nested_tensor=False,
        )

    def forward(
        self,
        window:   th.Tensor,
        failures: th.Tensor,
    ) -> th.Tensor:
        valid_index = POLICY_FEATURE_SLICES["valid"].start
        stage_valid = window[..., valid_index] > 0
        failure_valid = failures[..., valid_index] > 0
        stages = self.projection(window.masked_fill(~stage_valid.unsqueeze(-1), 0))
        stages = stages + self.positions + self.roles[0]
        failed = self.projection(failures.masked_fill(~failure_valid.unsqueeze(-1), 0)) + self.roles[1]
        tokens = th.cat((self.pool.expand(window.shape[0], -1, -1), stages, failed), dim=1)
        padding = th.cat((
            th.zeros((window.shape[0], 1), dtype=th.bool, device=window.device),
            ~stage_valid, ~failure_valid,
        ), dim=1)

        # The pool token is always valid, even for an entirely padded skill.
        return self.attention(tokens, src_key_padding_mask=padding)[:, 0]


class Agent(nn.Module):

    def __init__(
        self,
        observation_size: int,
        config:           PPOConfig,
    ) -> None:
        super().__init__()
        self.skill = SkillEncoder(config.window_size, config.embedding_size,
                                  config.attention_heads, config.attention_layers)
        self.observation = nn.Sequential(nn.Linear(observation_size, config.hidden_size), nn.Tanh())
        self.body = nn.Sequential(
            nn.Linear(config.hidden_size + config.embedding_size, config.hidden_size), nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh(),
        )
        self.action_sizes = tuple(carl.ACTION_NVECS)
        self.action_codec = CARLActionCodec()
        self.actor = nn.Linear(config.hidden_size, sum(self.action_sizes))
        self.critic = nn.Linear(config.hidden_size, 1)
        for module in (*self.observation, *self.body, self.actor, self.critic):
            if isinstance(module, nn.Linear):
                gain = 0.01 if module is self.actor else 1.0 if module is self.critic else math.sqrt(2)
                nn.init.orthogonal_(module.weight, gain)
                nn.init.zeros_(module.bias)

    def _hidden(
        self,
        observations: th.Tensor,
        window:       th.Tensor,
        failures:     th.Tensor,
    ) -> th.Tensor:
        skill = self.skill(window, failures)
        return self.body(th.cat((self.observation(observations), skill), dim=-1))

    def get_value(
        self,
        observations: th.Tensor,
        window:       th.Tensor,
        failures:     th.Tensor,
    ) -> th.Tensor:
        return self.critic(self._hidden(observations, window, failures)).squeeze(-1).float()

    def get_action_and_value(
        self,
        observations:  th.Tensor,
        window:        th.Tensor,
        failures:      th.Tensor,
        actions:       th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        hidden = self._hidden(observations, window, failures)
        logits = self.actor(hidden).float().masked_fill(~self.action_codec.mask(observations), -1e9)
        width = max(self.action_sizes)
        logits = th.stack([
            F.pad(head, (0, width - size), value=-1e9)
            for head, size in zip(logits.split(self.action_sizes, dim=-1), self.action_sizes)
        ], dim=-2)
        log_probabilities = logits.log_softmax(dim=-1)
        if actions is None:
            # Gumbel-max samples the same categorical distribution without multinomial's overhead.
            noise = 0 if deterministic else th.empty_like(logits).exponential_().log()
            actions = (logits - noise).argmax(dim=-1)
        log_prob = log_probabilities.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=(-1, -2))

        return actions, log_prob, entropy, self.critic(hidden).squeeze(-1).float()


class StageRewardManager(TrainingPackTerminationManager):

    def __init__(
        self,
        runtime:  TrainingPackRuntime,
        provider: TrainingPackResetProvider,
        config:   PPOConfig,
    ) -> None:
        super().__init__(runtime, provider)
        self.window_size = config.window_size
        self.width = runtime.pack.kind.shape[1]
        rewards = [config.stage_reward_base * config.stage_reward_multiplier ** stage for stage in range(self.width)]
        if any(not math.isfinite(value) or value > th.finfo(th.float32).max for value in rewards):
            raise ValueError("compounding stage rewards exceed float32 range")
        self.rewards = th.tensor(rewards, device=runtime.active.device)
        self.last_reward = th.zeros(runtime.n_sim, device=runtime.active.device)
        self._transition_runtime = copy(runtime)
        self.transition_exercise = runtime.exercise_index.clone()
        self.stage_passes = th.zeros(runtime.pack.kind.numel(), dtype=th.int64, device=runtime.active.device)
        self.episodes = th.zeros(len(provider.pack.exercises), dtype=th.int64, device=runtime.active.device)
        self.successes = th.zeros_like(self.episodes)
        self.timeouts = th.zeros_like(self.episodes)

    def step(
        self,
        events:    CarlEvents,
        frameskip: int,
        state:     CarlState | None = None,
    ) -> TrainingStep:
        stage = self.runtime.stage_index.clone()
        self.transition_exercise = self.runtime.exercise_index.clone()
        result = super().step(events, frameskip, state=state)
        # Snapshot before same-step autoreset changes either the skill or its progress.
        self._transition_runtime = copy(self.runtime)
        self.last_reward = self.rewards[stage] * result.advanced.float()
        self.stage_passes.scatter_add_(0, self.transition_exercise * self.width + stage, result.advanced.long())

        return result

    def reward(self, context: StepContext) -> th.Tensor:
        return self.last_reward.unsqueeze(-1)

    @property
    def transition_features(self) -> PolicyFeatures:
        # Runtime updates replace their tensors; this shallow snapshot therefore
        # survives autoreset without encoding every transition a second time.
        return policy_features(self._transition_runtime, K=self.window_size)

    def record_outcomes(
        self,
        done:      th.Tensor,
        succeeded: th.Tensor,
        timed_out: th.Tensor,
    ) -> None:
        self.episodes.scatter_add_(0, self.transition_exercise, done.long())
        self.successes.scatter_add_(0, self.transition_exercise, (done & succeeded).long())
        self.timeouts.scatter_add_(0, self.transition_exercise, (done & timed_out).long())

    def metrics(self) -> dict[str, float]:
        counts = th.stack((self.episodes, self.successes, self.timeouts), dim=-1).cpu().tolist()
        passes = self.stage_passes.reshape(-1, self.width).sum(dim=0).cpu().tolist()
        metrics = {f"stage/{stage + 1}/completions": float(count) for stage, count in enumerate(passes)}
        for index, exercise in enumerate(self.provider.pack.exercises):
            episodes, successes, timeouts = counts[index]
            metrics[f"exercise/{exercise.name}/episodes"] = float(episodes)
            metrics[f"exercise/{exercise.name}/success"] = successes / max(episodes, 1)
            metrics[f"exercise/{exercise.name}/timeout"] = timeouts / max(episodes, 1)
        # with_mirrors() produces consecutive pairs; neither side can hide the other.
        for index in range(0, len(counts) - 1, 2):
            rates = [successes / max(episodes, 1) for episodes, successes, _ in counts[index:index + 2]]
            name = self.provider.pack.exercises[index].name
            metrics[f"skill/{name}/both_sides_success"] = min(rates)
        self.stage_passes.zero_()
        self.episodes.zero_()
        self.successes.zero_()
        self.timeouts.zero_()

        return metrics


@dataclass
class Rollout:
    observations:   th.Tensor
    window:         th.Tensor
    failures:       th.Tensor
    actions:        th.Tensor
    log_probs:      th.Tensor
    values:         th.Tensor
    rewards:        th.Tensor
    dones:          th.Tensor
    timeout_values: th.Tensor
    advantages:     th.Tensor
    returns:        th.Tensor

    @classmethod
    def allocate(
        cls,
        config:           PPOConfig,
        observation_size: int,
        failure_slots:    int,
        device:           th.device,
    ) -> Rollout:
        shape = config.rollout_steps, config.n_sim

        return cls(
            observations=th.empty((*shape, observation_size), device=device),
            window=th.empty((*shape, config.window_size, POLICY_FEATURE_DIM), device=device),
            failures=th.empty((*shape, failure_slots, POLICY_FEATURE_DIM), device=device),
            actions=th.empty((*shape, len(carl.ACTION_NVECS)), dtype=th.int64, device=device),
            log_probs=th.empty(shape, device=device), values=th.empty(shape, device=device),
            rewards=th.empty(shape, device=device), dones=th.empty(shape, dtype=th.bool, device=device),
            timeout_values=th.empty(shape, device=device), advantages=th.empty(shape, device=device),
            returns=th.empty(shape, device=device),
        )

    @th.no_grad()
    def finish(
        self,
        next_value: th.Tensor,
        gamma:      float,
        gae_lambda: float,
    ) -> None:
        advantage = th.zeros_like(next_value)
        for step in reversed(range(self.rewards.shape[0])):
            continuing = (~self.dones[step]).float()
            bootstrap = continuing * next_value + self.timeout_values[step]
            delta = self.rewards[step] + gamma * bootstrap - self.values[step]
            advantage = delta + gamma * gae_lambda * continuing * advantage
            self.advantages[step].copy_(advantage)
            next_value = self.values[step]
        self.returns.copy_(self.advantages + self.values)


def _plain(tensor: th.Tensor) -> th.Tensor:
    return tensor if type(tensor) is th.Tensor else tensor.as_subclass(th.Tensor)


def _autocast(
    device:  th.device,
    enabled: bool,
) -> th.autocast:
    return th.autocast(device.type, dtype=th.bfloat16, enabled=enabled and device.type == "cuda")


@dataclass(frozen=True)
class TrainingFunctions:
    policy:   Callable[..., tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]
    value:    Callable[..., th.Tensor]
    loss:     Callable[..., tuple[th.Tensor, th.Tensor]]
    gae:      Callable[..., None]
    features: Callable[..., PolicyFeatures]


def compile_functions(
    agent:  Agent,
    config: PPOConfig,
) -> TrainingFunctions:
    functions = (agent.get_action_and_value, agent.get_value, ppo_loss, Rollout.finish, policy_features)
    if config.compile:
        functions = tuple(th.compile(function, fullgraph=True, dynamic=False) for function in functions)
    return TrainingFunctions(*functions)


@th.no_grad()
def collect(
    env:         CARLTorchVectorEnv,
    agent:       Agent,
    manager:     StageRewardManager,
    rollout:     Rollout,
    observation: th.Tensor,
    config:      PPOConfig | None = None,
    functions:   TrainingFunctions | None = None,
) -> th.Tensor:
    policy = agent.get_action_and_value if functions is None else functions.policy
    feature_function = policy_features if functions is None else functions.features
    mixed_precision = config is not None and config.bf16
    for step in range(rollout.rewards.shape[0]):
        features = feature_function(manager.runtime, K=manager.window_size)
        rollout.observations[step].copy_(observation)
        rollout.window[step].copy_(features.window)
        rollout.failures[step].copy_(features.failures)
        with _autocast(observation.device, mixed_precision):
            actions, log_prob, _, value = policy(observation, features.window, features.failures)
        rollout.actions[step].copy_(actions)
        rollout.log_probs[step].copy_(log_prob)
        rollout.values[step].copy_(value)

        observation, reward, terminated, truncated, info = env.step(actions)
        observation = _plain(observation)
        done = terminated | truncated
        rollout.rewards[step].copy_(reward)
        rollout.dones[step].copy_(done)
        rollout.timeout_values[step].zero_()
        if "final_obs" in info:
            # A skill deadline is a failed finite-horizon objective, not a time-limit bootstrap.
            bootstrap = truncated & ~terminated & ~info["exercise_timeout"]
            selected = bootstrap.nonzero(as_tuple=True)[0]
            if selected.numel():
                final_skill = manager.transition_features
                # Rare, variable-sized truncations use eager inference to avoid recompilation.
                with _autocast(observation.device, mixed_precision):
                    final_value = agent.get_value(
                        _plain(info["final_obs"])[selected], final_skill.window[selected], final_skill.failures[selected],
                    )
                rollout.timeout_values[step, selected] = final_value
        manager.record_outcomes(done, info["exercise_success"], truncated)

    return observation


def ppo_loss(
    agent:         Agent,
    observations:  th.Tensor,
    window:        th.Tensor,
    failures:      th.Tensor,
    actions:       th.Tensor,
    old_log_probs: th.Tensor,
    advantages:    th.Tensor,
    returns:       th.Tensor,
    old_values:    th.Tensor,
    clip_coef:     float,
    entropy_coef:  float,
    value_coef:    float,
) -> tuple[th.Tensor, th.Tensor]:
    _, log_prob, entropy, value = agent.get_action_and_value(observations, window, failures, actions)
    log_ratio = log_prob - old_log_probs
    ratio = log_ratio.exp()
    policy_loss = th.maximum(-advantages * ratio, -advantages * ratio.clamp(1 - clip_coef, 1 + clip_coef)).mean()
    clipped_value = old_values + (value - old_values).clamp(-clip_coef, clip_coef)
    value_loss = 0.5 * th.maximum((value - returns).square(), (clipped_value - returns).square()).mean()
    entropy = entropy.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    kl = ((ratio - 1) - log_ratio).mean()
    clipped_fraction = ((ratio - 1).abs() > clip_coef).float().mean()
    metrics = th.stack((policy_loss.detach(), value_loss.detach(), entropy.detach(), kl.detach(), clipped_fraction.detach()))

    return loss, metrics


def update_ppo(
    agent:     Agent,
    optimizer: th.optim.Optimizer,
    rollout:   Rollout,
    config:    PPOConfig,
    functions: TrainingFunctions | None = None,
) -> dict[str, float]:
    observations = rollout.observations.flatten(0, 1)
    windows = rollout.window.flatten(0, 1)
    failures = rollout.failures.flatten(0, 1)
    actions = rollout.actions.flatten(0, 1)
    old_log_probs = rollout.log_probs.flatten()
    old_values = rollout.values.flatten()
    returns = rollout.returns.flatten()
    advantages = rollout.advantages.flatten()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(EPSILON)
    totals = th.zeros(5, device=observations.device)
    examples = 0
    optimizer_steps = 0
    loss_function = ppo_loss if functions is None else functions.loss

    for epoch in range(config.update_epochs):
        epoch_kl = th.zeros((), device=observations.device)
        for indices in th.randperm(len(observations), device=observations.device).split(config.minibatch_size):
            with _autocast(observations.device, config.bf16):
                loss, metrics = loss_function(
                    agent, observations[indices], windows[indices], failures[indices], actions[indices],
                    old_log_probs[indices], advantages[indices], returns[indices], old_values[indices],
                    config.clip_coef, config.entropy_coef, config.value_coef,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm, foreach=True)
            # Check before Adam without a GPU-to-host synchronization on every minibatch.
            th._assert_async(th.isfinite(norm), "non-finite PPO gradients")
            optimizer.step()
            with th.no_grad():
                totals += metrics * len(indices)
                epoch_kl += metrics[3] * len(indices)
            examples += len(indices)
            optimizer_steps += 1
        if config.target_kl > 0 and (epoch_kl / len(observations)).item() > config.target_kl:
            break

    metrics = dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"),
                       (totals / examples).cpu().tolist()))
    target_variance = returns.var(unbiased=False)
    explained = 1 - (returns - old_values).var(unbiased=False) / target_variance.clamp_min(EPSILON)
    metrics["explained_variance"] = th.where(target_variance > EPSILON, explained, 0).item()
    metrics["ppo/optimizer_steps"] = float(optimizer_steps)
    metrics["ppo/epochs"] = float(epoch + 1)
    metrics["ppo/minibatches_per_epoch"] = float(math.ceil(len(observations) / config.minibatch_size))

    return metrics


def _pack_signature(pack: TrainingPack) -> str:
    return hashlib.sha256(repr(pack).encode()).hexdigest()


def save_checkpoint(
    path:      Path,
    agent:     Agent,
    optimizer: th.optim.Optimizer,
    manager:   StageRewardManager,
    config:    PPOConfig,
    steps:     int,
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "algorithm": "attention-pack-ppo",
        "observation_size": agent.observation[0].in_features,
        "feature_dim": POLICY_FEATURE_DIM,
        "pack_signature": _pack_signature(manager.provider.pack),
        "config": asdict(config),
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": steps,
        "torch_rng": th.get_rng_state(),
        "cuda_rng": th.cuda.get_rng_state(manager.runtime.active.device),
        "sampler_rng": manager.provider.sampler.generator.get_state(),
        "reset_counts": manager.provider.sampler.reset_counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    th.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path:      Path,
    agent:     Agent,
    optimizer: th.optim.Optimizer,
    manager:   StageRewardManager,
    config:    PPOConfig,
) -> int:
    payload = th.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("feature_dim") != POLICY_FEATURE_DIM:
        raise ValueError("incompatible PPO checkpoint or policy feature schema")
    if payload.get("pack_signature") != _pack_signature(manager.provider.pack):
        version = payload["config"].get("pack_version", 1)
        raise ValueError(f"checkpoint uses a different training pack; this checkpoint requires --pack-version {version}")
    for name in ("n_sim", "window_size", "embedding_size", "attention_heads", "attention_layers", "hidden_size",
                 "frameskip", "gamma", "stage_reward_base", "stage_reward_multiplier"):
        if payload["config"][name] != getattr(config, name):
            raise ValueError(f"checkpoint configuration differs: {name}")
    agent.load_state_dict(payload["agent"])
    # Preserve the optimizer implementation requested by this launch when
    # resuming older unfused Adam checkpoints, including placement of step state.
    for saved, current in zip(payload["optimizer"]["param_groups"], optimizer.param_groups):
        for name in ("fused", "foreach", "capturable"):
            if name in current:
                saved[name] = current[name]
    optimizer.load_state_dict(payload["optimizer"])
    th.set_rng_state(payload["torch_rng"])
    th.cuda.set_rng_state(payload["cuda_rng"], manager.runtime.active.device)
    manager.provider.sampler.generator.set_state(payload["sampler_rng"])
    manager.provider.sampler.reset_counts.copy_(payload["reset_counts"])

    return int(payload["steps"])


def train(config: PPOConfig) -> Path:
    if not th.cuda.is_available():
        raise RuntimeError("CARL PPO training requires a CUDA GPU")
    if config.bf16 and not th.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support bf16; use --no-bf16")
    th.manual_seed(config.seed)
    pack = aerial_pack(config.pack_version)
    compiled = CompiledTrainingPack.compile(pack)
    provider = TrainingPackResetProvider(pack, config.n_sim, seed=config.seed, balanced=True)
    runtime = TrainingPackRuntime(compiled, config.n_sim)
    manager = StageRewardManager(runtime, provider, config)
    env = CARLTorchVectorEnv(
        n_sim=config.n_sim, n_blue=1, n_orange=0, seed=config.seed,
        frameskip=config.frameskip, max_ticks=config.max_ticks,
        normalize=True, discrete_actions=True, capture_events=True,
        capture_ball_touches=compiled.needs_touch,
        reset_state_provider=provider, termination_manager=manager, reward_funcs=[manager.reward],
    )
    with ExitStack() as resources:
        resources.callback(env.close)
        observation_size = env.single_observation_space.shape[0]
        agent = Agent(observation_size, config).to(env.device)
        optimizer = th.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5, fused=True)
        rollout = Rollout.allocate(config, observation_size, compiled.failure_kind.shape[-1], env.device)
        steps = load_checkpoint(Path(config.resume), agent, optimizer, manager, config) if config.resume else 0
        directory = Path(config.output_dir) / f"{datetime.now():%Y%m%d-%H%M%S-%f}-seed{config.seed}"
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = directory / "latest.pt"
        (directory / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
        writer = open_writer(directory, asdict(config))
        if writer is not None:
            resources.callback(writer.close)
        observation = _plain(env.reset())
        iteration = 0
        print(f"Training {len(pack.exercises) // 2} mirrored skills; checkpoints: {directory}", flush=True)
        if config.compile:
            print("First update compiles policy, skill features, GAE, and PPO; later updates reuse those graphs.", flush=True)
        progress, task = progress_bar(config.total_timesteps, steps)
        if progress is not None:
            resources.callback(progress.stop)
        functions = compile_functions(agent, config)
        begin, collected, prepared, updated = (th.cuda.Event(enable_timing=True) for _ in range(4))
        with (directory / "metrics.jsonl").open("a") as log:
            while steps < config.total_timesteps:
                started = time.perf_counter()
                begin.record()
                optimizer.param_groups[0]["lr"] = config.learning_rate * (1 - steps / config.total_timesteps)
                observation = collect(env, agent, manager, rollout, observation, config, functions)
                collected.record()
                features = functions.features(runtime, K=config.window_size)
                with th.no_grad(), _autocast(env.device, config.bf16):
                    next_value = functions.value(observation, features.window, features.failures)
                functions.gae(rollout, next_value, config.gamma, config.gae_lambda)
                prepared.record()
                metrics = update_ppo(agent, optimizer, rollout, config, functions)
                updated.record()
                updated.synchronize()
                steps += rollout.rewards.numel()
                iteration += 1
                checkpoint_seconds = 0.0
                if iteration % config.save_every == 0 or steps >= config.total_timesteps:
                    saving = time.perf_counter()
                    save_checkpoint(checkpoint, agent, optimizer, manager, config, steps)
                    checkpoint_seconds = time.perf_counter() - saving
                metrics.update(manager.metrics())
                metrics["steps"] = steps
                metrics["reward_mean"] = rollout.rewards.mean().item()
                metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
                metrics["time/rollout_seconds"] = begin.elapsed_time(collected) / 1000
                metrics["time/gae_seconds"] = collected.elapsed_time(prepared) / 1000
                metrics["time/ppo_seconds"] = prepared.elapsed_time(updated) / 1000
                metrics["time/checkpoint_seconds"] = checkpoint_seconds
                metrics["time/update_seconds"] = time.perf_counter() - started
                metrics["transitions_per_second"] = rollout.rewards.numel() / metrics["time/update_seconds"]
                metrics["runtime/compilation_update"] = float(config.compile and iteration == 1)
                log.write(json.dumps(metrics, allow_nan=False) + "\n")
                log.flush()
                log_metrics(writer, steps, metrics, progress, task)
        # A resumed checkpoint may already satisfy the requested transition budget.
        if not checkpoint.exists():
            save_checkpoint(checkpoint, agent, optimizer, manager, config, steps)

    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for option in fields(PPOConfig):
        arguments = {"default": option.default, "help": option.name.replace("_", " ")}
        if isinstance(option.default, bool):
            arguments["action"] = argparse.BooleanOptionalAction
        else:
            arguments["type"] = str if option.default is None else type(option.default)
        parser.add_argument("--" + option.name.replace("_", "-"), **arguments)
    config = PPOConfig(**vars(parser.parse_args()))
    train(config)


if __name__ == "__main__":
    main()
