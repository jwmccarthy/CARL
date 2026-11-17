from dataclasses import asdict, replace
import unittest

import torch as th

import carl
from carl.training import POLICY_FEATURE_DIM, CompiledTrainingPack, TrainingPackRuntime, policy_features
from examples.training_packs.pack import aerial_pack
from examples.training_packs.playback import checkpoint_pack, checkpoint_policy
from examples.training_packs.ppo import CHECKPOINT_FORMAT, Agent, PPOConfig, _pack_signature


class TrainingPlaybackTests(unittest.TestCase):

    def setUp(self) -> None:
        self.pack = aerial_pack()
        self.runtime = TrainingPackRuntime(CompiledTrainingPack.compile(self.pack, device="cpu"), 1)
        self.config = PPOConfig(n_sim=16_384, embedding_size=16, attention_heads=2,
                                attention_layers=1, hidden_size=32, bf16=False, compile=False)
        self.observation_size = (carl.OBS_BALL + carl.OBS_PER_CAR + carl.OBS_BOOST_PADS
                                 + carl.OBS_RELATIVE_EGO_BALL + carl.OBS_RELATIVE_GOALS + carl.OBS_EGO_DODGE)
        self.agent = Agent(self.observation_size, self.config).eval()
        self.payload = {
            "format": CHECKPOINT_FORMAT,
            "feature_dim": POLICY_FEATURE_DIM,
            "pack_signature": _pack_signature(self.pack),
            "config": asdict(self.config),
            "agent": self.agent.state_dict(),
            "steps": 100,
        }

    def test_old_checkpoint_restores_exact_attention_policy_for_all_variants(self) -> None:
        self.assertEqual(checkpoint_pack(self.payload), self.pack)
        policy = checkpoint_policy(self.payload, self.runtime, self.observation_size)
        observation = th.zeros(1, self.observation_size)
        for index in range(len(self.pack.exercises)):
            self.runtime.reset(th.ones(1, dtype=th.bool), th.tensor([index]))
            features = policy_features(self.runtime, K=self.config.window_size)
            with th.inference_mode():
                expected = self.agent.get_action_and_value(
                    observation, features.window, features.failures, deterministic=True,
                )[0]
            th.testing.assert_close(policy.get_action(observation), expected)
            sampled = policy.get_action(observation, sample=True)
            offsets = th.tensor((0, *carl.ACTION_NVECS[:-1])).cumsum(0)
            self.assertTrue(policy.action_codec.mask(observation).gather(1, sampled + offsets).all().item())
        for key, value in self.agent.state_dict().items():
            th.testing.assert_close(policy.state_dict()[key], value)
        self.assertEqual(policy.runtime.n_sim, 1)

    def test_new_metadata_is_optional_but_observation_layout_must_match(self) -> None:
        self.payload["config"].pop("bf16")
        self.assertFalse(checkpoint_policy(self.payload, self.runtime, self.observation_size).bf16)
        self.payload.update(algorithm="attention-pack-ppo", observation_size=self.observation_size)
        checkpoint_policy(self.payload, self.runtime, self.observation_size)
        with self.assertRaisesRegex(ValueError, "observation layout"):
            checkpoint_policy(self.payload, self.runtime, self.observation_size + 1)
        self.payload["observation_size"] += 1
        with self.assertRaisesRegex(ValueError, "observation layout"):
            checkpoint_policy(self.payload, self.runtime, self.observation_size)

    def test_rejects_incompatible_schema_or_pack(self) -> None:
        for changes in ({"format": "goddard-leanrl-v1"}, {"feature_dim": POLICY_FEATURE_DIM + 1},
                        {"pack_signature": "different"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                checkpoint_pack(dict(self.payload, **changes))

    def test_v1_pack_is_reconstructed_from_old_checkpoint_signature(self) -> None:
        original = aerial_pack(1)
        self.payload["pack_signature"] = _pack_signature(original)
        self.payload["config"].pop("pack_version")
        self.assertEqual(checkpoint_pack(self.payload), original)

    def test_v2_checkpoint_retains_its_original_ball_heights(self) -> None:
        original = aerial_pack(2)
        self.payload["pack_signature"] = _pack_signature(original)
        self.payload["config"]["pack_version"] = 2
        self.assertEqual(checkpoint_pack(self.payload), original)
        self.assertEqual(_pack_signature(original), "fd5b987f508a84d4185e734703f3cedf879874d16bc25fbfc16ab4fb0e7b6ead")

    @unittest.skipUnless(th.cuda.is_available() and th.cuda.is_bf16_supported(), "requires CUDA bf16")
    def test_bf16_viewer_policy_with_inference_fast_path(self) -> None:
        self.payload["config"] = asdict(replace(self.config, bf16=True))
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(self.pack), 1)
        runtime.reset(th.ones(1, device="cuda", dtype=th.bool), th.zeros(1, device="cuda", dtype=th.long))
        policy = checkpoint_policy(self.payload, runtime, self.observation_size)
        observation = th.zeros(1, self.observation_size, device="cuda")
        for sample in (False, True):
            actions = policy.get_action(observation, sample=sample)
            self.assertEqual(tuple(actions.shape), (1, len(carl.ACTION_NVECS)))


if __name__ == "__main__":
    unittest.main()
