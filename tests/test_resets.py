"""GPU regression coverage for native and Gymnasium environment resets."""

import unittest
from dataclasses import replace
from functools import partial
from itertools import product
from unittest import mock

import carl
import torch
from carl.gymnasium.state import CARLBall, CARLCars, CARLMatchReset, CARLResetState
from carl.gymnasium.torch import CARLTorchVectorEnv
from torch.utils._python_dispatch import TorchDispatchMode

CURRENT = ("obs", "state", "score_difference", "episode_ticks", "overtime")
TRANSITION = (
    "transition_obs",
    "transition_state",
    "transition_score_difference",
    "transition_episode_ticks",
    "transition_overtime",
    "rewards",
    "dones",
)


def zeros(*shape, device="cuda"):
    return torch.zeros(shape, device=device, dtype=torch.float32)


def reset_inputs(n_selected, n_cars, *, match=True, internal=True, device="cuda"):
    empty = partial(zeros, device=device)
    ball_position = empty(n_selected, 3)
    ball_position[:, 2] = 800

    ball_velocity = empty(n_selected, 3)
    ball_velocity[:, 0] = 170

    car_position = empty(n_selected, n_cars, 3)
    car_position[..., 0] = torch.arange(n_cars, device=device) * 250 - 500
    car_position[..., 1] = -1000
    car_position[..., 2] = 17

    car_rotation = empty(n_selected, n_cars, 4)
    car_rotation[..., 3] = 1

    demoed = torch.zeros((n_selected, n_cars), device=device, dtype=torch.bool)
    demoed[:, -1] = True

    result = {
        "ball_position": ball_position,
        "ball_velocity": ball_velocity,
        "ball_angular_velocity": empty(n_selected, 3),
        "car_position": car_position,
        "car_rotation": car_rotation,
        "car_velocity": empty(n_selected, n_cars, 3),
        "car_angular_velocity": empty(n_selected, n_cars, 3),
        "car_demoed": demoed,
        "car_boost": empty(n_selected, n_cars).fill_(61),
    }

    if internal:
        values = empty(n_selected, n_cars, carl.CAR_INTERNAL_SIZE)

        # CAR_INT_HAS_JUMPED, CAR_INT_LAST_JUMP: a held jump must not retrigger.
        values[..., 3] = 1
        values[..., 5] = 1
        result["car_internal_state"] = values

    if match:
        result.update(
            blue_score=torch.full((n_selected,), 2, device=device, dtype=torch.int32),
            orange_score=torch.full((n_selected,), 1, device=device, dtype=torch.int32),
            episode_ticks=torch.full((n_selected,), 23, device=device, dtype=torch.int32),
        )

    return result


def sequential_reset(env, values, indices):
    env.reset_boost_pads(indices)

    env.set_ball(
        values["ball_position"],
        values["ball_velocity"],
        values["ball_angular_velocity"],
        simulation_indices=indices,
    )

    env.set_car(
        values["car_position"],
        values["car_rotation"],
        values["car_velocity"],
        values["car_angular_velocity"],
        values["car_demoed"],
        boost=values.get("car_boost"),
        internal_state=values.get("car_internal_state"),
        simulation_indices=indices,
    )

    if "blue_score" in values:
        env.set_match_state(
            values["blue_score"],
            values["orange_score"],
            values["episode_ticks"],
            simulation_indices=indices,
        )


def reset_provider(mask, *, typed, masked=False):
    indices = None if masked else mask.nonzero().flatten()
    count = len(mask) if masked else len(indices)
    values = reset_inputs(count, 2, device=mask.device)
    selection = {"reset_mask": mask} if masked else {"simulation_indices": indices}

    if not typed:
        return values | selection

    return typed_reset_state(values, indices, reset_mask=mask if masked else None)


def typed_reset_state(values, indices=None, *, reset_mask=None):
    ball = CARLBall.from_tensor(torch.cat((
        values["ball_position"],
        values["ball_velocity"],
        values["ball_angular_velocity"],
    ), dim=-1))

    count, n_cars = values["car_position"].shape[:2]
    device = values["car_position"].device
    cars = CARLCars.from_tensor(zeros(count, n_cars, carl.OBS_PER_CAR, device=device), n_cars)
    cars.position.copy_(values["car_position"])
    cars.forward[..., 0] = 1
    cars.up[..., 2] = 1
    cars.boost.copy_(values["car_boost"])
    cars[..., carl.OBS_DEMOED] = values["car_demoed"]

    match = CARLMatchReset(
        values["blue_score"], values["orange_score"], values["episode_ticks"],
    )

    return CARLResetState(
        indices, ball, cars, values["car_internal_state"], match,
        reset_mask=reset_mask,
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


def reset_on_alternate_stream(*, capsules, masked=False):
    env = carl.Env(3, 1, 1, 7)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        values = reset_inputs(3 if masked else 2, 2)

        if masked:
            values["reset_mask"] = torch.tensor([True, False, True], device="cuda")
        else:
            values["simulation_indices"] = torch.tensor([2, 0], device="cuda")

        for i in range(4):
            # Delay producers; temporary owners must survive native reads.
            torch.cuda._sleep(1_000_000)
            values["ball_position"][:, 0] = i * 100
            temporary = {name: value.clone() for name, value in values.items()}

            if capsules:
                native = torch.cuda.ExternalStream(env.stream)
                native.wait_stream(stream)

                temporary = {
                    name: torch.utils.dlpack.to_dlpack(value)
                    for name, value in temporary.items()
                }

            env.apply_reset_state(**temporary)
            del temporary

            scratch = [
                torch.empty_like(value).fill_(-999) for value in values.values()
            ]
            del scratch

    torch.cuda.current_stream().wait_stream(stream)
    return snapshot(env, CURRENT)


class ResetTests(unittest.TestCase):

    def make_env(self, **kwargs):
        env = CARLTorchVectorEnv(
            n_sim=2,
            n_blue=1,
            n_orange=1,
            frameskip=1,
            no_touch_timeout_ticks=1,
            **kwargs,
        )
        self.addCleanup(env.close)
        return env

    def assert_equal(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_combined_reset_matches_setters_and_continuation(self):
        cases = product((1, 2, 3), ([], [2, 0], None), (False, True))

        for team_size, selection, optional in cases:
            with self.subTest(
                team_size=team_size, selection=selection, optional=optional,
            ):
                self.check_setters_and_continuation(team_size, selection, optional)

    def check_setters_and_continuation(self, team_size, selection, optional):
        reference = carl.Env(3, team_size, team_size, 7, previous_controls=True)
        env = carl.Env(3, team_size, team_size, 7, previous_controls=True)

        indices = None
        if selection is not None:
            indices = torch.tensor(selection, device="cuda", dtype=torch.int64)

        count = 3 if indices is None else len(indices)
        values = reset_inputs(count, 2 * team_size, match=optional, internal=optional)

        if not optional:
            values.pop("car_boost")
            values["car_demoed"] = values["car_demoed"].int()

        actions = zeros(3, 2 * team_size, 7)
        actions[..., 6] = 1  # held jump

        # Populate terminal buffers deterministically before snapshotting.
        reference.no_touch_timeout_ticks = env.no_touch_timeout_ticks = 1
        reference.step_continuous(actions)
        env.step_continuous(actions)

        before = snapshot(env, CURRENT + TRANSITION)
        transition = {name: before[name] for name in TRANSITION}
        reference.no_touch_timeout_ticks = env.no_touch_timeout_ticks = 0

        for _ in range(2):
            sequential_reset(reference, values, indices)
            env.apply_reset_state(**values, simulation_indices=indices)

            self.assert_equal(snapshot(env, TRANSITION), transition)
            self.assert_equal(
                snapshot(env, ("obs", "state")),
                snapshot(reference, ("obs", "state")),
            )

        after = snapshot(env, CURRENT)
        selected = range(3) if selection is None else selection
        untouched = sorted(set(range(3)) - set(selected))

        for name in CURRENT:
            self.assert_equal(after[name][untouched], before[name][untouched])

        if optional and count:
            selected = slice(None) if indices is None else indices

            self.assert_equal(
                after["score_difference"][selected],
                values["blue_score"] - values["orange_score"],
            )
            self.assert_equal(after["episode_ticks"][selected], values["episode_ticks"])
            self.assertFalse(after["overtime"][selected].any().item())

        for _ in range(8):
            reference.step_continuous(actions)
            env.step_continuous(actions)

            self.assert_equal(
                snapshot(env, CURRENT + TRANSITION),
                snapshot(reference, CURRENT + TRANSITION),
            )

    def test_contacts_use_complete_scene_without_physics(self):
        for masked in (False, True):
            with self.subTest(masked=masked):
                self.check_restored_contacts(masked)

    def check_restored_contacts(self, masked):
        env = carl.Env(1, 1, 1, 7, previous_controls=True)
        values = reset_inputs(1, 2)

        if masked:
            values["reset_mask"] = torch.ones(1, device="cuda", dtype=torch.bool)

        # Top car's wheels rest on the bottom car, above all arena geometry.
        values["car_position"][0] = torch.tensor(
            [[0, 0, 257], [0, 0, 200]], device="cuda",
        )
        values["car_demoed"].zero_()
        env.apply_reset_state(**values)

        state = snapshot(env, ("state",))["state"]
        car_end = carl.OBS_BALL + 2 * carl.STATE_PER_CAR
        cars = state[:, carl.OBS_BALL:car_end].view(1, 2, carl.STATE_PER_CAR)

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
            {"reset_mask": torch.ones(2, device="cuda")},
            {"reset_mask": torch.ones(3, device="cuda", dtype=torch.bool)},
            {"reset_mask": torch.ones(2, device="cuda", dtype=torch.bool),
             "simulation_indices": torch.arange(2, device="cuda")},
            {"reset_mask": torch.ones(2, device="cuda", dtype=torch.bool),
             "ball_position": values["ball_position"][:1]},
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

        actions = zeros(2, 2, 7)
        env.step_continuous(actions)
        outputs = snapshot(env, CURRENT + TRANSITION)

        self.assert_equal(outputs["dones"], torch.tensor([False, True], device="cuda"))
        self.assert_equal(outputs["overtime"], torch.tensor([True, False], device="cuda"))
        self.assert_equal(
            outputs["transition_episode_ticks"],
            torch.tensor([4, 4], device="cuda", dtype=torch.int32),
        )

        for _ in range(2):
            env.step_continuous(actions)

        outputs = snapshot(env, CURRENT + TRANSITION)

        self.assert_equal(outputs["dones"], torch.tensor([True, False], device="cuda"))
        self.assertTrue(outputs["transition_overtime"][0].item())
        self.assertFalse(outputs["overtime"].any().item())

    def test_alternate_stream_temporary_inputs_and_capsules(self):
        expected = reset_on_alternate_stream(capsules=False)

        for capsules, masked in product((False, True), repeat=2):
            with self.subTest(capsules=capsules, masked=masked):
                self.assert_equal(
                    reset_on_alternate_stream(capsules=capsules, masked=masked),
                    expected,
                )

    def test_masked_reset_matches_indexed_reset(self):
        masks = ([False, False, False], [True, False, True], [True, True, True])

        for team_size, mask in product((1, 2, 3), masks):
            with self.subTest(team_size=team_size, mask=mask):
                self.check_masked_reset(team_size, mask)

    def check_masked_reset(self, team_size, mask):
        reference = carl.Env(3, team_size, team_size, 7, previous_controls=True)
        env = carl.Env(3, team_size, team_size, 7, previous_controls=True)
        actions = zeros(3, 2 * team_size, 7)
        values = reset_inputs(3, 2 * team_size)
        mask = torch.tensor(mask, device="cuda")

        reference.no_touch_timeout_ticks = env.no_touch_timeout_ticks = 1
        reference.step_continuous(actions)
        env.step_continuous(actions)
        before = snapshot(env, TRANSITION)
        reference.no_touch_timeout_ticks = env.no_touch_timeout_ticks = 0

        for repeat in range(3):
            mask = mask.roll(1)
            indices = mask.nonzero().flatten()
            values["ball_position"][:, 0] = torch.arange(3, device="cuda") * 100 + repeat
            compact = {name: value[indices].contiguous() for name, value in values.items()}

            reference.apply_reset_state(**compact, simulation_indices=indices)
            env.apply_reset_state(**values, reset_mask=mask)

            self.assert_equal(snapshot(env, CURRENT), snapshot(reference, CURRENT))
            self.assert_equal(snapshot(env, TRANSITION), before)

        for _ in range(8):
            reference.step_continuous(actions)
            env.step_continuous(actions)

            self.assert_equal(
                snapshot(env, CURRENT + TRANSITION),
                snapshot(reference, CURRENT + TRANSITION),
            )

    def test_close_waits_for_pending_masked_inputs(self):
        env = carl.Env(3, 1, 1, 7)
        stream = torch.cuda.ExternalStream(env.stream)
        complete = torch.cuda.Event()

        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000)

        env.apply_reset_state(
            **reset_inputs(3, 2),
            reset_mask=torch.ones(3, device="cuda", dtype=torch.bool),
        )
        complete.record(stream)
        del env

        self.assertTrue(complete.query())

    def test_wrapper_preserves_terminal_successors(self):
        for typed, goal, masked in product((False, True), repeat=3):
            with self.subTest(typed=typed, goal=goal, masked=masked):
                self.check_wrapper_terminal_successors(typed, goal, masked)

    def check_wrapper_terminal_successors(self, typed, goal, masked):
        control = self.make_env()
        candidate = self.make_env(masked_resets=masked)
        control.reset()
        candidate.reset()

        if goal:
            position = torch.tensor(
                [[0, 5300, 100], [0, 5300, 100]], device="cuda", dtype=torch.float32,
            )
            velocity = torch.zeros_like(position)

            control.set_ball(position, velocity, velocity)
            candidate.set_ball(position, velocity, velocity)

        provider = partial(reset_provider, typed=typed, masked=masked)
        candidate.reset_state_provider = provider
        actions = zeros(*control.action_space.shape)

        for completion in range(2):
            expected = control.step(actions)
            actual = candidate.step(actions)

            # Rewards, termination flags, final observations and episode stats.
            self.assert_equal(actual[1:4], expected[1:4])
            self.assert_equal(actual[4], expected[4])

            is_goal = goal and completion == 0
            done = actual[2] if is_goal else actual[3]
            self.assertTrue(done.all().item())
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

        candidate.close()
        control.close()


class FixedShapeOnly(TorchDispatchMode):
    """Fail on scalar readback or dynamic compaction in the masked wrapper path."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        forbidden = (
            torch.ops.aten._local_scalar_dense.default,
            torch.ops.aten.nonzero.default,
            torch.ops.aten.masked_select.default,
            torch.ops.aten._unique2.default,
        )

        if func in forbidden:
            raise AssertionError(f"host-dependent reset operation: {func}")

        if func == torch.ops.aten.index.Tensor:
            boolean_indices = (
                index is not None and index.dtype in (torch.bool, torch.uint8)
                for index in args[1]
            )
            if any(boolean_indices):
                raise AssertionError("boolean reset compaction")

        return func(*args, **(kwargs or {}))


class ResetValidationTests(unittest.TestCase):
    """CPU checks for fixed-shape provider dispatch and metadata validation."""

    def make_env(self, provider):
        env = CARLTorchVectorEnv.__new__(CARLTorchVectorEnv)
        env.n_sim = 3
        env.n_cars = 2
        env.masked_resets = True
        env.reset_state_provider = provider
        env._env = mock.Mock()
        return env

    def test_masked_provider_never_compacts_or_reads_scalars(self):
        masks = ([False, False, False], [True, False, True], [True, True, True])

        for typed, eligible in product((False, True), masks):
            with self.subTest(typed=typed, eligible=eligible):
                self.check_fixed_shape_provider(typed, eligible)

    def check_fixed_shape_provider(self, typed, eligible):
        values = reset_inputs(3, 2, device="cpu")
        mask = torch.tensor([True, True, False])
        eligible = torch.tensor(eligible)
        state = values | {"reset_mask": mask}

        if typed:
            state = typed_reset_state(values, reset_mask=mask)

        provider = mock.Mock(return_value=state)
        env = self.make_env(provider)

        with FixedShapeOnly():
            env._apply_reset_state(eligible)

        provider.assert_called_once_with(eligible)
        env._env.apply_reset_state.assert_called_once()
        args, kwargs = env._env.apply_reset_state.call_args

        self.assertIsNone(kwargs["simulation_indices"])
        self.assertEqual(args[0].shape, (3, 3))
        self.assertEqual(args[3].shape, (3, 2, 3))
        torch.testing.assert_close(kwargs["reset_mask"], mask & eligible)
        torch.testing.assert_close(args[4], values["car_rotation"])

    def test_masked_provider_validates_selection_metadata(self):
        invalid = (
            {"reset_mask": torch.ones(3)},
            {"reset_mask": torch.ones(2, dtype=torch.bool)},
            {"reset_mask": torch.ones(3, dtype=torch.bool),
             "simulation_indices": torch.arange(3)},
            {"simulation_indices": torch.arange(3)},
        )

        for typed, selection in product((False, True), invalid):
            with self.subTest(typed=typed, selection=list(selection)):
                self.check_invalid_selection(typed, selection)

    def check_invalid_selection(self, typed, selection):
        values = reset_inputs(3, 2, device="cpu")
        state = values | selection

        if typed:
            state = typed_reset_state(
                values, selection.get("simulation_indices"),
                reset_mask=selection.get("reset_mask"),
            )

        env = self.make_env(mock.Mock(return_value=state))
        eligible = torch.ones(3, dtype=torch.bool)

        with FixedShapeOnly(), self.assertRaises(ValueError):
            env._apply_reset_state(eligible)

        env._env.apply_reset_state.assert_not_called()

    def test_normalized_masked_inputs_remain_fixed_shape(self):
        values = reset_inputs(3, 2, device="cpu")
        mask = torch.tensor([True, False, True])
        state = typed_reset_state(values, reset_mask=mask)
        state = replace(state, normalized=True)
        position_scale = torch.tensor(carl.OBS_POSITION_SCALE)

        state.ball.position.div_(position_scale)
        state.ball.velocity.div_(carl.BALL_MAX_SPEED)
        state.ball.angular_velocity.div_(carl.BALL_MAX_ANG_SPEED)
        state.cars.position.div_(position_scale)
        state.cars.velocity.div_(carl.CAR_MAX_SPEED)
        state.cars.angular_velocity.div_(carl.CAR_MAX_ANG_SPEED)
        state.cars.boost.div_(carl.BOOST_MAX)

        env = self.make_env(mock.Mock(return_value=state))

        with FixedShapeOnly():
            env._apply_reset_state(mask)

        args, kwargs = env._env.apply_reset_state.call_args
        torch.testing.assert_close(args[0], values["ball_position"])
        torch.testing.assert_close(args[1], values["ball_velocity"])
        torch.testing.assert_close(args[3], values["car_position"])
        torch.testing.assert_close(kwargs["car_boost"], values["car_boost"])

    def test_indexed_providers_remain_supported(self):
        for typed in (False, True):
            with self.subTest(typed=typed):
                self.check_indexed_provider(typed)

    def check_indexed_provider(self, typed):
        env = self.make_env(partial(reset_provider, typed=typed))
        env.masked_resets = False
        eligible = torch.tensor([True, False, True])

        env._apply_reset_state(eligible)
        args, kwargs = env._env.apply_reset_state.call_args

        self.assertEqual(args[0].shape, (2, 3))
        self.assertIsNone(kwargs["reset_mask"])
        torch.testing.assert_close(kwargs["simulation_indices"], torch.tensor([0, 2]))


if __name__ == "__main__":
    unittest.main()
