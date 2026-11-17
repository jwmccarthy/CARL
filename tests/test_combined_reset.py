"""GPU regression coverage for the complete selected-reset transaction."""

import unittest

import carl
import torch
from carl.gymnasium.state import CARLBall, CARLCars, CARLMatchReset, CARLResetState
from carl.gymnasium.torch import CARLTorchVectorEnv


def reset_inputs(n_selected, n_cars, *, match=True, internal=True):
    def zeros(*shape):
        return torch.zeros(shape, device="cuda", dtype=torch.float32)

    ball_position = zeros(n_selected, 3)
    ball_position[:, 2] = 800
    ball_velocity = zeros(n_selected, 3)
    ball_velocity[:, 0] = 170
    car_position = zeros(n_selected, n_cars, 3)
    car_position[..., 0] = torch.arange(n_cars, device="cuda") * 250 - 500
    car_position[..., 1] = -1000
    car_position[..., 2] = 17
    car_rotation = zeros(n_selected, n_cars, 4)
    car_rotation[..., 3] = 1
    demoed = torch.zeros((n_selected, n_cars), device="cuda", dtype=torch.bool)
    demoed[:, -1] = True
    result = {
        "ball_position": ball_position,
        "ball_velocity": ball_velocity,
        "ball_angular_velocity": zeros(n_selected, 3),
        "car_position": car_position,
        "car_rotation": car_rotation,
        "car_velocity": zeros(n_selected, n_cars, 3),
        "car_angular_velocity": zeros(n_selected, n_cars, 3),
        "car_demoed": demoed,
        "car_boost": zeros(n_selected, n_cars).fill_(61),
    }
    if internal:
        values = zeros(n_selected, n_cars, carl.CAR_INTERNAL_SIZE)
        # CAR_INT_HAS_JUMPED, CAR_INT_LAST_JUMP: a held jump must not retrigger.
        values[..., 3] = 1
        values[..., 5] = 1
        result["car_internal_state"] = values
    if match:
        result.update(
            blue_score=torch.full((n_selected,), 2, device="cuda", dtype=torch.int32),
            orange_score=torch.full((n_selected,), 1, device="cuda", dtype=torch.int32),
            episode_ticks=torch.full((n_selected,), 23, device="cuda", dtype=torch.int32),
        )
    return result


def legacy_reset(env, values, indices):
    env.reset_boost_pads(indices)
    env.set_ball(
        values["ball_position"], values["ball_velocity"],
        values["ball_angular_velocity"], simulation_indices=indices,
    )
    env.set_car(
        values["car_position"], values["car_rotation"], values["car_velocity"],
        values["car_angular_velocity"], values["car_demoed"],
        boost=values.get("car_boost"), internal_state=values.get("car_internal_state"),
        simulation_indices=indices,
    )
    if "blue_score" in values:
        env.set_match_state(
            values["blue_score"], values["orange_score"], values["episode_ticks"],
            simulation_indices=indices,
        )


def snapshot(env, names):
    # Native output capsules are non-owning; explicitly order copies on Torch.
    torch.cuda.current_stream().wait_stream(torch.cuda.ExternalStream(env.stream))
    result = {
        name: torch.from_dlpack(getattr(env, "get_" + name)()).clone()
        for name in names
    }
    torch.cuda.synchronize()
    return result


CURRENT = ("obs", "state", "score_difference", "episode_ticks", "overtime")
TRANSITION = (
    "transition_obs", "transition_state", "transition_score_difference",
    "transition_episode_ticks", "transition_overtime", "rewards", "dones",
)


class CombinedResetTests(unittest.TestCase):
    def assert_equal(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_matches_setters_and_continuation(self):
        for team_size in (1, 2, 3):
            for selection in ([], [2, 0], None):
                for optional in (False, True):
                    with self.subTest(team_size=team_size, selection=selection, optional=optional):
                        old = carl.Env(3, team_size, team_size, 7, previous_controls=True)
                        new = carl.Env(3, team_size, team_size, 7, previous_controls=True)
                        indices = None if selection is None else torch.tensor(
                            selection, device="cuda", dtype=torch.int64,
                        )
                        count = 3 if indices is None else len(indices)
                        values = reset_inputs(count, 2 * team_size, match=optional, internal=optional)
                        if not optional:
                            values.pop("car_boost")
                            values["car_demoed"] = values["car_demoed"].int()
                        actions = torch.zeros((3, 2 * team_size, 7), device="cuda")
                        actions[..., 6] = 1  # held jump
                        # Populate terminal buffers deterministically before snapshotting.
                        old.no_touch_timeout_ticks = new.no_touch_timeout_ticks = 1
                        old.step_continuous(actions)
                        new.step_continuous(actions)
                        before = snapshot(new, CURRENT + TRANSITION)
                        old.no_touch_timeout_ticks = new.no_touch_timeout_ticks = 0
                        for _ in range(2):
                            legacy_reset(old, values, indices)
                            new.apply_reset_state(**values, simulation_indices=indices)
                            self.assert_equal(snapshot(new, TRANSITION), {
                                key: before[key] for key in TRANSITION
                            })
                            self.assert_equal(snapshot(new, ("obs", "state")),
                                              snapshot(old, ("obs", "state")))
                        after = snapshot(new, CURRENT)
                        untouched = sorted(set(range(3)) - set(range(3) if selection is None else selection))
                        for name in CURRENT:
                            self.assert_equal(after[name][untouched], before[name][untouched])
                        if optional and count:
                            selected = slice(None) if indices is None else indices
                            self.assert_equal(after["score_difference"][selected], values["blue_score"] - values["orange_score"])
                            self.assert_equal(after["episode_ticks"][selected], values["episode_ticks"])
                            self.assertFalse(after["overtime"][selected].any().item())
                        for _ in range(8):
                            old.step_continuous(actions)
                            new.step_continuous(actions)
                            self.assert_equal(snapshot(new, CURRENT + TRANSITION),
                                              snapshot(old, CURRENT + TRANSITION))
                        del old, new

    def test_contacts_use_complete_scene_without_physics(self):
        env = carl.Env(1, 1, 1, 7, previous_controls=True)
        values = reset_inputs(1, 2)
        # Top car's wheels rest on the bottom car, above all arena geometry.
        values["car_position"][0] = torch.tensor(
            [[0, 0, 257], [0, 0, 200]], device="cuda",
        )
        values["car_demoed"].zero_()
        env.apply_reset_state(**values)
        state = snapshot(env, ("state",))["state"]
        cars = state[:, carl.OBS_BALL:carl.OBS_BALL + 2 * carl.STATE_PER_CAR].view(
            1, 2, carl.STATE_PER_CAR,
        )
        self.assert_equal(cars[..., :3], values["car_position"])
        self.assert_equal(cars[..., 3:6], values["car_velocity"])
        self.assertTrue(cars[0, 0, carl.OBS_ON_GROUND].item())
        self.assertFalse(cars[0, 1, carl.OBS_ON_GROUND].item())
        self.assertFalse(cars[..., carl.STATE_BALL_TOUCH].any().item())
        # Demoed restored cars must have no grounded contact state.
        values["car_demoed"][:, 0] = True
        env.apply_reset_state(**values)
        obs = snapshot(env, ("obs",))["obs"]
        self.assertFalse(obs[0, 0, carl.OBS_BALL + carl.OBS_ON_GROUND].item())

    def test_rejects_invalid_input_before_mutation(self):
        env = carl.Env(2, 1, 1, 7)
        before = snapshot(env, CURRENT)
        values = reset_inputs(2, 2)
        bad_inputs = (
            {"episode_ticks": None},
            {"episode_ticks": values["episode_ticks"].long()},
            {"car_internal_state": torch.zeros((2, 2, 1), device="cuda")},
            {"car_position": values["car_position"].transpose(0, 1)},
            {"ball_velocity": torch.utils.dlpack.to_dlpack(values["ball_velocity"].cpu())},
        )
        for bad in bad_inputs:
            with self.subTest(fields=list(bad)):
                with self.assertRaises((ValueError, TypeError)):
                    env.apply_reset_state(**(values | bad))
                self.assert_equal(snapshot(env, CURRENT), before)

    def test_restored_match_clock_enters_overtime_or_truncates(self):
        env = carl.Env(2, 1, 1, 7)
        env.max_ticks = 4
        env.overtime_timeout_ticks = 2
        values = reset_inputs(2, 2)
        values["orange_score"][0] = 2
        values["episode_ticks"].fill_(3)
        env.apply_reset_state(**values)
        actions = torch.zeros((2, 2, 7), device="cuda")
        env.step_continuous(actions)
        outputs = snapshot(env, CURRENT + TRANSITION)
        self.assert_equal(outputs["dones"], torch.tensor([False, True], device="cuda"))
        self.assert_equal(outputs["overtime"], torch.tensor([True, False], device="cuda"))
        self.assert_equal(outputs["transition_episode_ticks"],
                          torch.tensor([4, 4], device="cuda", dtype=torch.int32))
        for _ in range(2):
            env.step_continuous(actions)
        outputs = snapshot(env, CURRENT + TRANSITION)
        self.assert_equal(outputs["dones"], torch.tensor([True, False], device="cuda"))
        self.assertTrue(outputs["transition_overtime"][0].item())
        self.assertFalse(outputs["overtime"].any().item())

    def test_alternate_stream_temporary_inputs_and_capsules(self):
        def run(capsules):
            env = carl.Env(3, 1, 1, 7)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                values = reset_inputs(2, 2)
                values["simulation_indices"] = torch.tensor([2, 0], device="cuda")
                for i in range(4):
                    # Delay producers; temporary owners must survive native reads.
                    torch.cuda._sleep(1_000_000)
                    values["ball_position"][:, 0] = i * 100
                    temporary = {name: value.clone() for name, value in values.items()}
                    if capsules:
                        native = torch.cuda.ExternalStream(env.stream)
                        native.wait_stream(stream)
                        temporary = {name: torch.utils.dlpack.to_dlpack(value)
                                     for name, value in temporary.items()}
                    env.apply_reset_state(**temporary)
                    del temporary
                    scratch = [torch.empty_like(value).fill_(-999) for value in values.values()]
                    del scratch
            torch.cuda.current_stream().wait_stream(stream)
            return snapshot(env, CURRENT)

        self.assert_equal(run(capsules=True), run(capsules=False))

    def test_wrapper_preserves_terminal_successors(self):
        for typed in (False, True):
            for goal in (False, True):
                with self.subTest(typed=typed, goal=goal):
                    control = CARLTorchVectorEnv(2, 1, 1, frameskip=1, no_touch_timeout_ticks=1)
                    candidate = CARLTorchVectorEnv(2, 1, 1, frameskip=1, no_touch_timeout_ticks=1)
                    try:
                        control.reset()
                        candidate.reset()
                        if goal:
                            position = torch.tensor([[0, 5300, 100], [0, 5300, 100]],
                                                    device="cuda", dtype=torch.float32)
                            velocity = torch.zeros_like(position)
                            control.set_ball(position, velocity, velocity)
                            candidate.set_ball(position, velocity, velocity)

                        def provider(mask, typed=typed):
                            indices = mask.nonzero().flatten()
                            values = reset_inputs(len(indices), 2)
                            if not typed:
                                return values | {"simulation_indices": indices}
                            ball = CARLBall.from_tensor(torch.cat((
                                values["ball_position"], values["ball_velocity"],
                                values["ball_angular_velocity"],
                            ), dim=-1))
                            cars = CARLCars.from_tensor(torch.zeros(
                                (len(indices), 2, carl.OBS_PER_CAR), device="cuda",
                            ), 2)
                            cars.position.copy_(values["car_position"])
                            cars.forward[..., 0] = 1
                            cars.up[..., 2] = 1
                            cars.boost.copy_(values["car_boost"])
                            cars[..., carl.OBS_DEMOED] = values["car_demoed"]
                            return CARLResetState(
                                indices, ball, cars, values["car_internal_state"],
                                CARLMatchReset(values["blue_score"], values["orange_score"],
                                               values["episode_ticks"]),
                            )

                        candidate.reset_state_provider = provider
                        actions = torch.zeros(control.action_space.shape, device="cuda")
                        for completion in range(2):
                            expected = control.step(actions)
                            actual = candidate.step(actions)
                            # Rewards, goal/truncation classification, final obs and episode stats.
                            self.assert_equal(actual[1:4], expected[1:4])
                            self.assert_equal(actual[4], expected[4])
                            is_goal = goal and completion == 0
                            self.assertTrue(actual[2].all().item() if is_goal else actual[3].all().item())
                            self.assert_equal(
                                snapshot(candidate._env, TRANSITION),
                                snapshot(control._env, TRANSITION),
                            )
                            self.assertTrue((actual[0].ball.position[:, 2] == 800).all().item())
                            # Re-align starting states before another completion.
                            candidate.reset_state_provider = None
                            candidate.reset()
                            control.reset()
                            candidate.reset_state_provider = provider
                    finally:
                        candidate.close()
                        control.close()


if __name__ == "__main__":
    unittest.main()
