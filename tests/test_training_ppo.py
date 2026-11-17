from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch as th
from torch import nn

import carl
from carl.events import BallEvents, CarlEvents, CarEvents, Demoed, Event, Goal, Jump
from carl.gymnasium import CARLTorchVectorEnv
from carl.training import (
    POLICY_FEATURE_DIM, POLICY_FEATURE_SLICES, BallState, CarState,
    CompiledTrainingPack, Exercise, InitialState, OnSurface, Stage, TargetKind,
    TrainingPack, TrainingPackResetProvider, TrainingPackRuntime, policy_features,
)
from examples.training_packs.pack import aerial_pack
from examples.training_packs.ppo import (
    Agent, PPOConfig, Rollout, SkillEncoder, StageRewardManager, collect, train,
    update_ppo,
)


def config(**changes: int | float | str | None) -> PPOConfig:
    return replace(PPOConfig(n_sim=2, rollout_steps=3, minibatch_size=3, update_epochs=1,
                             embedding_size=16, attention_heads=2, attention_layers=1, hidden_size=32,
                             bf16=False, compile=False), **changes)


def features(batch: int) -> tuple[th.Tensor, th.Tensor]:
    window = th.zeros(batch, 3, POLICY_FEATURE_DIM)
    failures = th.zeros(batch, 2, POLICY_FEATURE_DIM)
    window[:, 0, POLICY_FEATURE_SLICES["kind"].start + TargetKind.LEAVE - 1] = 1
    window[:, 1, POLICY_FEATURE_SLICES["kind"].start + TargetKind.TOUCH - 1] = 1
    window[:, :2, POLICY_FEATURE_SLICES["entity"].start] = 1
    window[:, :2, POLICY_FEATURE_SLICES["valid"]] = 1
    failures[:, 0, POLICY_FEATURE_SLICES["kind"].start] = 1
    failures[:, 0, POLICY_FEATURE_SLICES["valid"]] = 1

    return window, failures


def observation_size() -> int:
    return max(carl.OBS_EGO_BOOST_INDEX, carl.OBS_EGO_ON_GROUND_INDEX,
               carl.OBS_EGO_FLIP_INDEX, carl.OBS_EGO_DODGE_TIME_INDEX) + 1


def events(flags: list[int]) -> CarlEvents:
    cars = th.tensor(flags, dtype=th.int32).unsqueeze(-1)
    zeros = th.zeros(len(flags), dtype=th.int32)

    return CarlEvents(
        cars=CarEvents(cars, surface_contacted=th.zeros_like(cars), surface_left=th.zeros_like(cars)),
        ball=BallEvents(zeros.clone(), surface_entered=zeros, surface_left=zeros, goal_side=zeros.clone()),
    )


class TrainingPPOTests(unittest.TestCase):

    def test_aerial_pack_has_independent_pairs_for_every_skill(self) -> None:
        pack = aerial_pack()
        self.assertEqual(len(pack.exercises), 12)
        compiled = CompiledTrainingPack.compile(pack, device="cpu")
        self.assertTrue(compiled.needs_touch)
        self.assertTrue(compiled.needs_touch_speed)
        self.assertTrue(compiled.needs_state)
        for original, mirrored in zip(pack.exercises[::2], pack.exercises[1::2]):
            self.assertEqual(mirrored, original.mirrored())
            self.assertEqual(mirrored.target_goal, original.target_goal)
            self.assertIsNot(original.stages, mirrored.stages)

    def test_attention_masks_padding_and_is_invariant_to_failure_order(self) -> None:
        encoder = SkillEncoder(3, 16, 2, 1)
        window, failures = features(2)
        expected = encoder(window, failures)
        changed_window = window.clone()
        changed_window[:, 2] = 123
        changed_window[:, 2, POLICY_FEATURE_SLICES["valid"]] = 0
        changed_failures = failures.clone()
        changed_failures[:, 1] = -100
        changed_failures[:, 1, POLICY_FEATURE_SLICES["valid"]] = 0
        th.testing.assert_close(encoder(changed_window, changed_failures), expected)
        th.testing.assert_close(encoder(window, failures.flip(1)), expected)
        self.assertTrue(th.isfinite(encoder(th.zeros_like(window), th.zeros_like(failures))).all().item())
        self.assertFalse(th.allclose(encoder(window.flip(1), failures), expected))

    def test_action_masks_log_probability_round_trip_and_encoder_gradients(self) -> None:
        agent = Agent(observation_size(), config())
        observation = th.zeros(2, observation_size())
        window, failures = features(2)
        with th.no_grad():
            actions, old_log_prob, _, _ = agent.get_action_and_value(observation, window, failures)
        _, log_prob, entropy, values = agent.get_action_and_value(observation, window, failures, actions)
        th.testing.assert_close(log_prob, old_log_prob)
        mask = agent.action_codec.mask(observation)
        offset = 0
        for index, size in enumerate(carl.ACTION_NVECS):
            self.assertTrue(((actions[:, index] >= 0) & (actions[:, index] < size)).all().item())
            self.assertTrue(mask.gather(1, (actions[:, index] + offset).unsqueeze(-1)).all().item())
            offset += size
        (-log_prob.mean() - 0.01 * entropy.mean() + values.square().mean()).backward()
        self.assertGreater(agent.skill.projection.weight.grad.abs().sum().item(), 0)
        attention_gradient = agent.skill.attention.layers[0].self_attn.in_proj_weight.grad
        self.assertGreater(attention_gradient.abs().sum().item(), 0)

    def test_fused_categorical_matches_torch_distribution(self) -> None:
        agent = Agent(observation_size(), config())
        observation = th.zeros(8, observation_size())
        observation[::2, carl.OBS_EGO_BOOST_INDEX] = 1
        window, failures = features(8)
        with th.no_grad():
            actions, log_prob, entropy, _ = agent.get_action_and_value(
                observation, window, failures, deterministic=True,
            )
            hidden = agent._hidden(observation, window, failures)
            logits = agent.actor(hidden).masked_fill(~agent.action_codec.mask(observation), -1e9)
        expected_log_prob = th.zeros(8)
        expected_entropy = th.zeros(8)
        for index, head in enumerate(logits.split(agent.action_sizes, dim=-1)):
            distribution = th.distributions.Categorical(logits=head)
            expected_log_prob += distribution.log_prob(actions[:, index])
            expected_entropy += distribution.entropy()
            th.testing.assert_close(actions[:, index], head.argmax(dim=-1))
        th.testing.assert_close(log_prob, expected_log_prob)
        th.testing.assert_close(entropy, expected_entropy)

    def test_gae_stops_at_episode_boundaries_and_adds_only_explicit_bootstraps(self) -> None:
        rollout = Rollout.allocate(config(n_sim=1), observation_size(), 1, th.device("cpu"))
        rollout.values[:, 0] = th.tensor([1, 2, 3])
        rollout.rewards[:] = 1
        rollout.dones[:, 0] = th.tensor([False, True, False])
        rollout.timeout_values.zero_()
        rollout.finish(th.tensor([4.]), gamma=0.5, gae_lambda=1)
        th.testing.assert_close(rollout.advantages[:, 0], th.tensor([0.5, -1, 0]))
        th.testing.assert_close(rollout.returns[:, 0], th.tensor([1.5, 1, 3]))
        rollout.timeout_values[1, 0] = 10
        rollout.finish(th.tensor([4.]), gamma=0.5, gae_lambda=1)
        th.testing.assert_close(rollout.returns[:, 0], th.tensor([4., 6, 3]))

    def test_feature_construction_compiles_as_one_graph_and_tracks_runtime_changes(self) -> None:
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(aerial_pack(), device="cpu"), 2)
        runtime.reset(th.ones(2, dtype=th.bool), th.tensor([0, 1]))
        compiled = th.compile(policy_features, backend="eager", fullgraph=True)
        for stage in (0, 1):
            runtime.stage_index[:] = stage
            expected = policy_features(runtime)
            actual = compiled(runtime)
            th.testing.assert_close(actual.window, expected.window)
            th.testing.assert_close(actual.failures, expected.failures)

    def test_ppo_update_retrains_attention_from_stored_raw_features(self) -> None:
        settings = config()
        agent = Agent(observation_size(), settings)
        optimizer = th.optim.Adam(agent.parameters(), lr=settings.learning_rate)
        rollout = Rollout.allocate(settings, observation_size(), 2, th.device("cpu"))
        window, failures = features(2)
        rollout.observations.zero_()
        rollout.window[:] = window
        rollout.failures[:] = failures
        with th.no_grad():
            for step in range(settings.rollout_steps):
                actions, log_prob, _, values = agent.get_action_and_value(rollout.observations[step], window, failures)
                rollout.actions[step].copy_(actions)
                rollout.log_probs[step].copy_(log_prob)
                rollout.values[step].copy_(values)
        rollout.rewards[:] = th.tensor([[0, 1], [1, 0], [2, 1]])
        rollout.dones.zero_()
        rollout.timeout_values.zero_()
        rollout.finish(th.zeros(2), settings.gamma, settings.gae_lambda)
        before = agent.skill.projection.weight.detach().clone()
        metrics = update_ppo(agent, optimizer, rollout, settings)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertFalse(th.equal(agent.skill.projection.weight, before))

    def test_compounding_stage_rewards_are_once_only_and_failures_get_zero(self) -> None:
        initial = InitialState(CarState(OnSurface()), BallState((0, 0, 500)))
        pack = TrainingPack("stages", [Exercise("sequence", initial,
                            [Stage(Jump()), Stage(Jump()), Stage(Goal())], fail_on=[Demoed()])])
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack, device="cpu"), 2)
        provider = TrainingPackResetProvider(pack, 2, device="cpu")
        manager = StageRewardManager(runtime, provider, config())
        runtime.reset(th.ones(2, dtype=th.bool), th.zeros(2, dtype=th.long))
        manager.step(events([Event.JUMP] * 2), 1)
        self.assertEqual(manager.last_reward.tolist(), [1, 1])
        manager.step(events([Event.JUMP] * 2), 1)
        self.assertEqual(manager.last_reward.tolist(), [2, 2])
        manager.step(events([0, 0]), 1)
        self.assertEqual(manager.last_reward.tolist(), [0, 0])
        goal = events([0, Event.DEMOED])
        goal.ball.flags[:] = int(Event.GOAL)
        goal.ball.goal_side[:] = 1
        result = manager.step(goal, 1)
        self.assertEqual(result.succeeded.tolist(), [True, False])
        self.assertEqual(result.failed.tolist(), [False, True])
        self.assertEqual(manager.last_reward.tolist(), [4, 0])
        manager.step(goal, 1)
        self.assertEqual(manager.last_reward.tolist(), [0, 0])

    def test_repeats_do_not_farm_stage_rewards_and_both_sides_are_reported(self) -> None:
        initial = InitialState(CarState(OnSurface()), BallState((0, 0, 500)))
        pack = TrainingPack("repeats", [Exercise("repeat", initial,
                            [Stage(Goal(), repeat_on=Jump(), max_repeats=3)])]).with_mirrors()
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack, device="cpu"), 2)
        provider = TrainingPackResetProvider(pack, 2, device="cpu", balanced=True)
        manager = StageRewardManager(runtime, provider, config())
        runtime.reset(th.ones(2, dtype=th.bool), th.tensor([0, 1]))
        self.assertTrue(manager.step(events([Event.JUMP] * 2), 1).repeated.all().item())
        self.assertEqual(manager.last_reward.tolist(), [0, 0])
        manager.record_outcomes(th.tensor([True, True]), th.tensor([True, False]), th.tensor([False, True]))
        metrics = manager.metrics()
        self.assertEqual(metrics["exercise/repeat/success"], 1)
        self.assertEqual(metrics["exercise/repeat (mirrored)/success"], 0)
        self.assertEqual(metrics["skill/repeat/both_sides_success"], 0)

    def test_lazy_transition_skill_snapshot_survives_autoreset(self) -> None:
        initial = InitialState(CarState(OnSurface()), BallState((0, 0, 500)))
        pack = TrainingPack("snapshot", [Exercise("jump", initial, [Stage(Jump()), Stage(Goal())]),
                                          Exercise("goal", initial, [Stage(Goal())])])
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack, device="cpu"), 2)
        provider = TrainingPackResetProvider(pack, 2, device="cpu")
        manager = StageRewardManager(runtime, provider, config())
        runtime.reset(th.ones(2, dtype=th.bool), th.zeros(2, dtype=th.long))
        with mock.patch("examples.training_packs.ppo.policy_features", side_effect=AssertionError("duplicate features")):
            manager.step(events([Event.JUMP] * 2), 1)
        expected = manager.transition_features
        runtime.reset(th.ones(2, dtype=th.bool), th.zeros(2, dtype=th.long))
        actual = manager.transition_features
        th.testing.assert_close(actual.window, expected.window)
        th.testing.assert_close(actual.failures, expected.failures)
        self.assertFalse(th.equal(actual.window, policy_features(runtime).window))

    def test_invalid_configuration(self) -> None:
        for changes in ({"n_sim": 0}, {"rollout_steps": True}, {"learning_rate": float("nan")},
                        {"attention_heads": 3}, {"gamma": 1.1}, {"stage_reward_multiplier": 0.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)

    @unittest.skipUnless(th.cuda.is_available(), "requires a CUDA GPU")
    def test_time_limit_bootstraps_transition_skill_not_reset_skill(self) -> None:
        class FeatureValueAgent(nn.Module):

            def get_value(
                self,
                observation: th.Tensor,
                window:      th.Tensor,
                failures:    th.Tensor,
            ) -> th.Tensor:
                self.last_window = window.clone()
                self.last_observation = observation.clone()
                kinds = window[:, 0, POLICY_FEATURE_SLICES["kind"]]
                return 7 * kinds[:, TargetKind.EVENT - 1] + 11 * kinds[:, TargetKind.GOAL - 1]

            def get_action_and_value(
                self,
                observation: th.Tensor,
                window:      th.Tensor,
                failures:    th.Tensor,
            ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
                zeros = th.zeros(len(observation), device=observation.device)

                return (th.zeros((len(observation), len(carl.ACTION_NVECS)), dtype=th.long, device=observation.device),
                        zeros, zeros, self.get_value(observation, window, failures))

        for deadline, expected in ((200, 7), (1, 0)):
            with self.subTest(deadline=deadline):
                initial = InitialState(CarState(OnSurface()), BallState((100, 0, 500)))
                other = InitialState(CarState(OnSurface()), BallState((-100, 0, 500)))
                pack = TrainingPack("bootstrap", [Exercise("old", initial, [Stage(Jump(), within_ticks=deadline)]),
                                                   Exercise("new", other, [Stage(Goal())])])
                settings = config(n_sim=1, rollout_steps=1, frameskip=1, max_ticks=1)
                runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack), 1)
                provider = TrainingPackResetProvider(pack, 1, balanced=True)
                manager = StageRewardManager(runtime, provider, settings)
                env = CARLTorchVectorEnv(1, 1, 0, frameskip=1, max_ticks=1, discrete_actions=True,
                                        no_touch_timeout_ticks=1,
                                        capture_events=True, reset_state_provider=provider,
                                        termination_manager=manager, reward_funcs=[manager.reward])
                self.addCleanup(env.close)
                observation = env.reset().as_subclass(th.Tensor)
                rollout = Rollout.allocate(settings, observation.shape[-1], 1, env.device)
                agent = FeatureValueAgent().to(env.device)
                after = collect(env, agent, manager, rollout, observation)
                self.assertEqual(rollout.timeout_values.item(), expected)
                self.assertEqual(agent.last_observation[0, 0].item(), 100)
                self.assertEqual(after[0, 0].item(), -100)
                self.assertEqual(agent.last_window[0, 0, POLICY_FEATURE_SLICES["kind"]].argmax().item(), TargetKind.EVENT - 1)
                live = policy_features(runtime)
                self.assertEqual(live.window[0, 0, POLICY_FEATURE_SLICES["kind"]].argmax().item(), TargetKind.GOAL - 1)
                rollout.finish(th.tensor([11.], device=env.device), gamma=0.5, gae_lambda=1)
                self.assertEqual(rollout.returns.item(), expected / 2)
                env.close()

    @unittest.skipUnless(th.cuda.is_available(), "requires a CUDA GPU")
    def test_gpu_training_and_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp/opencode", prefix="carl-pack-ppo-test-") as directory:
            settings = config(n_sim=12, total_timesteps=48, rollout_steps=4, minibatch_size=24,
                              frameskip=4, max_ticks=8, save_every=1, output_dir=directory)
            checkpoint = train(settings)
            payload = th.load(checkpoint, weights_only=True, map_location="cpu")
            self.assertEqual(payload["steps"], 48)
            self.assertEqual(payload["algorithm"], "attention-pack-ppo")
            self.assertEqual(payload["feature_dim"], POLICY_FEATURE_DIM)
            self.assertTrue(payload["optimizer"]["state"])
            # Older checkpoints have neither viewer metadata nor precision flags,
            # and store unfused Adam's step counters on CPU.
            payload.pop("algorithm")
            payload.pop("observation_size")
            payload["config"].pop("bf16")
            payload["config"].pop("compile")
            for group in payload["optimizer"]["param_groups"]:
                group["fused"] = None
            legacy = checkpoint.with_name("legacy.pt")
            th.save(payload, legacy)
            resumed = train(replace(settings, total_timesteps=96, resume=str(legacy)))
            continued = th.load(resumed, weights_only=True, map_location="cpu")
            self.assertEqual(continued["steps"], 96)
            self.assertTrue((continued["reset_counts"] > payload["reset_counts"]).all().item())
            self.assertTrue(continued["optimizer"]["param_groups"][0]["fused"])
            metrics = (Path(resumed).parent / "metrics.jsonl").read_text()
            self.assertIn("both_sides_success", metrics)
            self.assertIn("time/checkpoint_seconds", metrics)

    @unittest.skipUnless(th.cuda.is_available() and th.cuda.is_bf16_supported(), "requires CUDA bf16")
    def test_compiled_bf16_training_updates_attention_and_logs_finite_metrics(self) -> None:
        import json

        with tempfile.TemporaryDirectory(dir="/tmp/opencode", prefix="carl-pack-compiled-test-") as directory:
            settings = config(n_sim=12, total_timesteps=96, rollout_steps=4, minibatch_size=24,
                              frameskip=4, max_ticks=8, save_every=2, output_dir=directory,
                              bf16=True, compile=True)
            checkpoint = train(settings)
            metrics = [json.loads(line) for line in (checkpoint.parent / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics), 2)
            for row in metrics:
                self.assertTrue(all(math.isfinite(value) for value in row.values()))
                self.assertEqual(row["ppo/optimizer_steps"], 2)
            self.assertEqual(metrics[0]["time/checkpoint_seconds"], 0)
            self.assertGreater(metrics[1]["time/checkpoint_seconds"], 0)
            self.assertEqual(metrics[1]["runtime/compilation_update"], 0)
            payload = th.load(checkpoint, map_location="cpu", weights_only=True)
            projection_shape = (settings.embedding_size, POLICY_FEATURE_DIM)
            gradients = [state["exp_avg"] for state in payload["optimizer"]["state"].values()
                         if tuple(state["exp_avg"].shape) == projection_shape]
            self.assertEqual(len(gradients), 1)
            self.assertGreater(gradients[0].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
