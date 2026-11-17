"""Native graph replay must preserve the ordinary launch path exactly."""

import unittest
from functools import partial
from itertools import product

import carl
import torch
from carl.gymnasium import CARLTorchVectorEnv
from test_resets import CURRENT, TRANSITION, reset_inputs, reset_provider

EVENT_OUTPUTS = (
    "car_flags", "car_surface_contacted", "car_surface_left", "ball_flags",
    "ball_surface_entered", "ball_surface_left", "goal_side", "ball_speed_change",
)
TOUCH_OUTPUTS = ("ball_touch_surfaces", "ball_touch_tick_surfaces",
                 "ball_touch_tick_speed", "ball_touch_tick_speed_before")


@unittest.skipUnless(torch.cuda.is_available() and not torch.version.hip, "requires CUDA")
class NativeGraphTests(unittest.TestCase):
    def snapshot(self, env):
        env.events.acquire(torch.cuda.current_stream().cuda_stream)
        result = {name: torch.from_dlpack(getattr(env, f"get_{name}")()).clone()
                  for name in (*CURRENT, *TRANSITION)}
        names = EVENT_OUTPUTS if env.events.enabled else ()
        if env.events.touches_enabled:
            names += TOUCH_OUTPUTS
        result.update({name: torch.from_dlpack(getattr(env.events, f"get_{name}")()).clone()
                       for name in names})
        return result

    def actions(self, env, generator, discrete):
        shape = (env.n_sim, env.n_cars)
        if discrete:
            return torch.stack([torch.randint(size, shape, device="cuda", generator=generator)
                                for size in carl.ACTION_NVECS], -1).int()
        return torch.rand((*shape, 7), device="cuda", generator=generator) * 2 - 1

    def scene(self, team):
        values = reset_inputs(8, team * 2)
        values["car_demoed"].zero_()
        values["car_internal_state"].zero_()
        values["ball_position"][:] = torch.tensor([
            [0., 5300., 200.], [0., 0., 93.15], [4090., 0., 500.],
            [0., 0., 2000.], [-350., -1000., 100.], [0., 0., 800.],
            [3500., 4500., 100.], [0., -5300., 200.],
        ], device="cuda")
        return values

    def test_physics_outputs_and_events_match(self):
        for team, discrete, capture, touches in product((1, 2, 3), (False, True),
                                                        (False, True), (False, True)):
            with self.subTest(team=team, discrete=discrete, capture=capture, touches=touches):
                self.check_trajectory(team, discrete, capture, touches)

    def check_trajectory(self, team, discrete, capture, touches):
        options = {"frameskip": 4, "previous_controls": True,
                   "capture_events": capture, "capture_ball_touches": touches}
        plain = carl.Env(8, team, team, 17, **options)
        graph = carl.Env(8, team, team, 17, native_cuda_graphs=True, **options)
        self.assertTrue(graph.native_cuda_graphs)
        self.assertEqual(graph.native_graph_nodes, 0)
        generator = torch.Generator(device="cuda").manual_seed(23)
        values = self.scene(team)
        for env in (plain, graph):
            env.no_touch_timeout_ticks = 24
            env.apply_reset_state(**values)

        saw_done = False
        for step in range(24):
            if step in (5, 8, 13):
                mask = torch.arange(8, device="cuda") % 2 == 0
                if step == 8:
                    mask.zero_()
                if step == 13:
                    mask.fill_(True)
                for env in (plain, graph):
                    env.apply_reset_state(**values, reset_mask=mask)
            if step == 18:
                for env in (plain, graph):
                    env.reset_selected(torch.arange(8, device="cuda") % 2 == 1)

            actions = self.actions(plain, generator, discrete)
            for env in (plain, graph):
                (env.step if discrete else env.step_continuous)(actions)
            expected, actual = self.snapshot(plain), self.snapshot(graph)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            saw_done |= bool(actual["dones"].any())

        self.assertTrue(saw_done)
        self.assertGreater(graph.native_graph_nodes, 0)
        self.assertEqual(plain.native_graph_nodes, 0)

    def test_control_modes_resize_reset_and_validation(self):
        plain = carl.Env(2, 1, 1, 7, capture_events=True, capture_ball_touches=True)
        graph = carl.Env(2, 1, 1, 7, capture_events=True, capture_ball_touches=True,
                         native_cuda_graphs=True)
        with self.assertRaises(ValueError):
            graph.step(torch.zeros((1, 2, 7), device="cuda", dtype=torch.int32))
        self.assertEqual(graph.native_graph_nodes, 0)
        generator = torch.Generator(device="cuda").manual_seed(29)

        for frameskip in (1, 8, 2, 4):
            plain.frameskip = graph.frameskip = frameskip
            self.assertEqual(graph.native_graph_nodes, 0)
            for discrete in (False, True, False, True):
                before = graph.native_graph_nodes
                actions = self.actions(plain, generator, discrete)
                for env in (plain, graph):
                    (env.step if discrete else env.step_continuous)(actions)
                torch.testing.assert_close(self.snapshot(graph), self.snapshot(plain), rtol=0, atol=0)
                self.assertGreaterEqual(graph.native_graph_nodes, before)
            nodes = graph.native_graph_nodes
            graph.frameskip = frameskip
            self.assertEqual(graph.native_graph_nodes, nodes)
            plain.reset()
            graph.reset()
            self.assertEqual(graph.native_graph_nodes, nodes)
            with self.assertRaises(ValueError):
                graph.frameskip = 0
            self.assertEqual(graph.native_graph_nodes, nodes)

    def test_alternate_streams_temporary_actions_and_close(self):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        plain = carl.Env(4, 1, 1, 7, frameskip=8, capture_events=True, capture_ball_touches=True)
        graph = carl.Env(4, 1, 1, 7, frameskip=8, capture_events=True, capture_ball_touches=True,
                         native_cuda_graphs=True)
        generator = torch.Generator(device="cuda").manual_seed(31)
        snapshots = []
        for _ in range(5):
            with torch.cuda.stream(producer):
                torch.cuda._sleep(2_000_000)
                actions = self.actions(plain, generator, True)
                plain.step(actions.clone())
                graph.step(actions.clone())
                del actions
            with torch.cuda.stream(consumer):
                torch.cuda._sleep(2_000_000)
                snapshots.append((self.snapshot(plain), self.snapshot(graph)))
        torch.cuda.current_stream().wait_stream(consumer)
        for expected, actual in snapshots:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.cuda.stream(producer):
            graph.step(torch.zeros((4, 2, 7), device="cuda", dtype=torch.int32))
        del graph
        torch.cuda.synchronize()

    def test_wrapper_masked_resets_and_deferred_reports(self):
        envs = [CARLTorchVectorEnv(
            3, 1, 1, frameskip=4, no_touch_timeout_ticks=4, discrete_actions=True,
            capture_events=True, capture_ball_touches=True,
            reset_state_provider=partial(reset_provider, typed=True, masked=True),
            masked_resets=True, defer_episode_reporting=True, native_cuda_graphs=enabled,
        ) for enabled in (False, True)]
        for env in envs:
            self.addCleanup(env.close)
        torch.testing.assert_close(envs[0].reset(), envs[1].reset(), rtol=0, atol=0)
        for _ in range(5):
            actions = torch.zeros(envs[0].action_space.shape, device="cuda", dtype=torch.int32)
            expected, actual = (env.step(actions) for env in envs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(envs[0].drain_episode_reports(), envs[1].drain_episode_reports())


if __name__ == "__main__":
    unittest.main()
