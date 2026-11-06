import unittest

import torch as th

from carl.events import Event, Goal, Jump, StateCondition
from carl.gymnasium import CARLTorchVectorEnv
from carl.training import (
    BallState, CarState, CompiledTrainingPack, Exercise, InitialState, OnSurface,
    Stage, TargetKind, TrainingPack, TrainingPackResetProvider, TrainingPackRuntime,
    TrainingPackTerminationManager,
)


class TargetDrivenPolicy(th.nn.Module):

    def __init__(self):
        super().__init__()
        self.jump_embedding = th.nn.Embedding(2, 1)

        with th.no_grad():
            self.jump_embedding.weight[:, 0] = th.tensor([0.0, 1.0])

    def forward(self, observation, target):
        rows = th.arange(observation.shape[0], device=observation.device)
        kind = target.stages.kind[rows, target.stage_index]
        event = target.stages.event_mask[rows, target.stage_index]
        jump = ((kind == TargetKind.CAR_EVENT) & ((event & int(Event.JUMP)) != 0)).long()

        actions = th.zeros((observation.shape[0], 7), device=observation.device)
        actions[:, -1] = self.jump_embedding(jump).squeeze(-1)

        return actions


@unittest.skipUnless(th.cuda.is_available(), "requires a CUDA GPU")
class TrainingIntegrationTests(unittest.TestCase):

    def test_goal_is_not_failed_by_reset_ball_height(self):
        initial = InitialState(ego=CarState(placement=OnSurface(position=(0, -1000))),
                               ball=BallState(position=(0, 5300, 200)))
        pack = TrainingPack(name="goal", exercises=[Exercise(
            name="goal", initial=initial,
            stages=[Stage(Goal(), fail_on=[StateCondition("ball_height", maximum=95.15)])],
        )])
        provider = TrainingPackResetProvider(pack, n_sim=1, seed=1)
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack), n_sim=1)
        manager = TrainingPackTerminationManager(runtime, provider)
        env = CARLTorchVectorEnv(n_sim=1, n_blue=1, n_orange=0, frameskip=4,
                                 capture_events=True, reset_state_provider=provider,
                                 termination_manager=manager)
        self.addCleanup(env.close)
        env.reset()
        _, _, terminated, _, info = env.step(th.zeros((1, 7), device="cuda"))
        self.assertTrue(terminated[0].item())
        self.assertTrue(info["exercise_success"][0].item())
        self.assertFalse(info["exercise_failure"][0].item())

    def test_target_driven_mixed_pack_loop(self):
        def start(ball_position):
            return InitialState(
                ego  = CarState(placement=OnSurface(position=(0, -1000))),
                ball = BallState(position=ball_position),
            )

        pack = TrainingPack(name="mixed loop", exercises=[
            Exercise(name="two stages", initial=start((0, 0, 500)), stages=[
                Stage(Jump(), within_ticks=12),
                Stage(StateCondition("car_height", minimum=25), within_ticks=48),
            ]),
            Exercise(name="state failure", initial=start((0, 0, 200)), stages=[
                Stage(Jump(), fail_on=[StateCondition("ball_height", maximum=300)]),
            ]),
            Exercise(name="native goal", initial=start((0, 5300, 200)), stages=[Stage(Goal())]),
            Exercise(name="timeout", initial=start((0, 0, 500)), stages=[Stage(Goal(), within_ticks=8)]),
        ])
        n_sim = 64
        provider = TrainingPackResetProvider(pack, n_sim=n_sim, seed=13)
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack), n_sim=n_sim)
        manager = TrainingPackTerminationManager(runtime, provider)
        reward_events = []
        reward_scores = []

        def reward(context):
            reward_events.append(context.events.done.clone())
            reward_scores.append(context.events.score_delta.clone())
            return context.events.done[:, None].float()

        env = CARLTorchVectorEnv(
            n_sim                = n_sim,
            n_blue               = 1,
            n_orange             = 0,
            frameskip            = 4,
            capture_events       = True,
            reset_state_provider = provider,
            termination_manager  = manager,
            reward_funcs         = [reward],
        )
        self.addCleanup(env.close)
        policy = TargetDrivenPolicy().to("cuda")
        observation = env.reset()
        seen = th.zeros(5, device="cuda", dtype=th.bool)
        initial_target = runtime.target_sequence()
        self.assertEqual(initial_target.exercise_index.unique().numel(), 4)

        with th.no_grad():
            for _ in range(5):
                before = runtime.target_sequence()
                counts = provider.sampler.reset_counts.clone()
                actions = policy(observation, before)
                observation, rewards, terminated, truncated, info = env.step(actions)
                after = runtime.target_sequence()
                done = terminated | truncated

                first_stage = (before.exercise_index == 0) & (before.stage_index == 0)
                second_stage = (before.exercise_index == 0) & (before.stage_index == 1)
                expected_failure = before.exercise_index == 1
                expected_goal = before.exercise_index == 2
                expected_timeout = (before.exercise_index == 3) & (before.ticks_remaining <= 4)

                self.assertTrue(th.equal(info["exercise_failure"], expected_failure))
                self.assertTrue(th.equal(info["exercise_timeout"], expected_timeout))
                self.assertTrue((info["exercise_success"][expected_goal]).all().item())
                self.assertTrue((info["exercise_success"][second_stage]).all().item())
                self.assertFalse(done[first_stage].any().item())
                self.assertTrue((terminated[expected_goal]).all().item())
                self.assertTrue((reward_scores[-1][expected_goal] == 1).all().item())
                self.assertTrue(th.equal(rewards.bool(), done))
                self.assertTrue(th.equal(reward_events[-1], done))
                self.assertTrue(th.equal(info["_final_obs"], done))
                self.assertTrue(th.equal(provider.sampler.reset_counts, counts + done.long()))
                self.assertTrue(th.equal(after.exercise_index, provider.exercise_index))
                self.assertTrue((after.stage_index[done] == 0).all().item())
                self.assertTrue((after.stage_index[first_stage] == 1).all().item())
                self.assertTrue(th.equal(after.exercise_index[~done], before.exercise_index[~done]))
                self.assertTrue(th.isfinite(observation).all().item())
                self.assertTrue((info["final_obs"].ball_position[expected_failure, 2] < 300).all().item())

                reset_ball = observation.ball_position[done]
                reset_exercise = after.exercise_index[done]
                expected_y = th.where(reset_exercise == 2, 5300.0, 0.0)
                expected_z = th.where((reset_exercise == 1) | (reset_exercise == 2), 200.0, 500.0)
                th.testing.assert_close(reset_ball[:, 1], expected_y)
                th.testing.assert_close(reset_ball[:, 2], expected_z)

                seen |= th.stack((first_stage.any(), second_stage.any(), expected_failure.any(),
                                  expected_goal.any(), expected_timeout.any()))

        self.assertTrue(seen.all().item())

    def test_state_failure_uses_transition_before_reset(self):
        initial = InitialState(
            ego  = CarState(placement=OnSurface(position=(0, -1000))),
            ball = BallState(position=(0, 0, 500)),
        )
        exercise = Exercise(
            name    = "jump above ball height",
            initial = initial,
            stages  = [Stage(Jump(), fail_on=[StateCondition("ball_height", maximum=300)])],
        )
        pack = TrainingPack(name="state", exercises=[exercise])
        provider = TrainingPackResetProvider(pack, n_sim=2, seed=7)
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack), n_sim=2)
        manager = TrainingPackTerminationManager(runtime, provider)
        env = CARLTorchVectorEnv(
            n_sim                = 2,
            n_blue               = 1,
            n_orange             = 0,
            frameskip            = 4,
            capture_events       = True,
            reset_state_provider = provider,
            termination_manager  = manager,
        )
        self.addCleanup(env.close)
        env.reset()

        env.set_ball(
            position           = th.tensor([[0, 0, 200]], device="cuda", dtype=th.float32),
            velocity           = th.zeros((1, 3), device="cuda"),
            angular_velocity   = th.zeros((1, 3), device="cuda"),
            simulation_indices = th.tensor([0], device="cuda"),
        )
        actions = th.zeros((2, 7), device="cuda")
        actions[:, -1] = 1
        observation, _, terminated, truncated, info = env.step(actions)

        self.assertEqual(terminated.tolist(), [True, True])
        self.assertEqual(truncated.tolist(), [False, False])
        self.assertEqual(info["exercise_failure"].tolist(), [True, False])
        self.assertEqual(info["exercise_success"].tolist(), [False, True])
        self.assertLess(info["final_obs"].ball_position[0, 2].item(), 300)
        self.assertEqual(observation.ball_position[0, 2].item(), 500)

    def test_parallel_endings_and_resets(self):
        initial = InitialState(
            ego  = CarState(placement=OnSurface(position=(0, -1000))),
            ball = BallState(position=(0, 0, 500)),
        )
        pack = TrainingPack(name="outcomes", exercises=[
            Exercise(name="jump", initial=initial, stages=[Stage(Jump())]),
            Exercise(name="fail", initial=initial, stages=[Stage(Goal(), fail_on=[Jump()])]),
            Exercise(name="timeout", initial=initial, stages=[Stage(Goal(), within_ticks=4)]),
            Exercise(name="continue", initial=initial, stages=[Stage(Goal())]),
            Exercise(name="goal", initial=initial, stages=[Stage(Goal())]),
        ])
        provider = TrainingPackResetProvider(pack, n_sim=5, seed=7)
        runtime = TrainingPackRuntime(CompiledTrainingPack.compile(pack), n_sim=5)
        manager = TrainingPackTerminationManager(runtime, provider)
        env = CARLTorchVectorEnv(
            n_sim                = 5,
            n_blue               = 1,
            n_orange             = 0,
            frameskip            = 4,
            capture_events       = True,
            reset_state_provider = provider,
            termination_manager  = manager,
        )
        self.addCleanup(env.close)
        env.reset()
        provider.exercise_index = th.arange(5, device="cuda")
        manager.reset(th.ones(5, dtype=th.bool, device="cuda"))

        env.set_ball(
            position           = th.tensor([[0, 5300, 200]], device="cuda", dtype=th.float32),
            velocity           = th.zeros((1, 3), device="cuda"),
            angular_velocity   = th.zeros((1, 3), device="cuda"),
            simulation_indices = th.tensor([4], device="cuda"),
        )
        actions = th.zeros((5, 7), device="cuda")
        actions[:2, -1] = 1
        observation, _, terminated, truncated, info = env.step(actions)

        self.assertEqual(terminated.tolist(), [True, True, False, False, True])
        self.assertEqual(truncated.tolist(), [False, False, True, False, False])
        self.assertEqual(info["exercise_success"].tolist(), [True, False, False, False, True])
        self.assertEqual(info["exercise_failure"].tolist(), [False, True, False, False, False])
        self.assertEqual(info["exercise_timeout"].tolist(), [False, False, True, False, False])
        self.assertEqual(info["_final_obs"].tolist(), [True, True, True, False, True])
        self.assertEqual(provider.sampler.reset_counts.tolist(), [2, 2, 2, 1, 2])
        self.assertTrue(runtime.active.all().item())
        self.assertEqual(runtime.exercise_index.tolist(), provider.exercise_index.tolist())
        self.assertTrue(th.isfinite(observation).all().item())
        self.assertGreater(info["final_obs"].car_position[0, 0, 2].item(),
                           observation.car_position[0, 0, 2].item())


if __name__ == "__main__":
    unittest.main()
