from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from examples.training_packs.reporting import JsonlBridge, log_metrics, open_writer, progress_bar


@unittest.skipUnless(importlib.util.find_spec("tensorboard"), "requires the training extra")
class TrainingReportingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(dir="/tmp/opencode", prefix="carl-reporting-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def scalars(self, name: str) -> list[tuple[int, float]]:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = EventAccumulator(str(self.path / "tensorboard")).Reload()
        return [(event.step, event.value) for event in events.Scalars(name)]

    def test_native_summaries_and_configuration(self) -> None:
        writer = open_writer(self.path, {"n_sim": 16_384}, required=True)
        with redirect_stdout(io.StringIO()):
            log_metrics(writer, 12, {"steps": 12, "reward_mean": 0.25, "time/ppo_seconds": 1.5})
        writer.close()
        self.assertEqual(self.scalars("reward_mean"), [(12, 0.25)])
        self.assertEqual(self.scalars("time/ppo_seconds"), [(12, 1.5)])
        with self.assertRaisesRegex(ValueError, "native TensorBoard"):
            JsonlBridge(self.path / "metrics.jsonl")

    def test_bridge_waits_for_partial_line_and_deduplicates_restarts(self) -> None:
        path = self.path / "metrics.jsonl"
        (self.path / "config.json").write_text(json.dumps({"n_sim": 8192}))
        path.write_bytes(b'{"steps": 8, "reward_mean": 0.5}\n{"steps": 16, "reward_mean":')
        bridge = JsonlBridge(path)
        try:
            self.assertEqual(bridge.sync(), 1)
            self.assertEqual(bridge.sync(), 0)
            with path.open("ab") as stream:
                stream.write(b' 0.75}\n')
            self.assertEqual(bridge.sync(), 1)
            self.assertEqual(bridge.sync(), 0)
        finally:
            bridge.close()
        bridge = JsonlBridge(path)
        try:
            self.assertEqual(bridge.sync(), 0)
            with path.open("ab") as stream:
                stream.write(b'{"steps": 24, "reward_mean": 1}\n')
            self.assertEqual(bridge.sync(), 1)
        finally:
            bridge.close()
        self.assertEqual(self.scalars("reward_mean"), [(8, 0.5), (16, 0.75), (24, 1)])

    def test_bridge_handles_rewritten_log(self) -> None:
        path = self.path / "metrics.jsonl"
        path.write_text('{"steps": 8, "reward_mean": 0.5}\n\n')
        bridge = JsonlBridge(path)
        try:
            self.assertEqual(bridge.sync(), 1)
            path.write_text('{"steps":16,"reward_mean":1}\n')
            self.assertEqual(bridge.sync(), 1)
        finally:
            bridge.close()
        self.assertEqual(self.scalars("reward_mean"), [(8, 0.5), (16, 1)])


@unittest.skipUnless(importlib.util.find_spec("rich"), "requires the training extra")
class TrainingProgressTests(unittest.TestCase):

    def test_progress_supports_resume_and_caps_overshooting_rollout(self) -> None:
        with redirect_stdout(io.StringIO()):
            progress, task = progress_bar(total=100, initial=40)
            try:
                self.assertEqual(progress.tasks[task].completed, 40)
                log_metrics(None, 120, {"transitions_per_second": 12345}, progress, task)
                self.assertEqual(progress.tasks[task].completed, 100)
                self.assertEqual(progress.tasks[task].fields["rate"], "12,345 steps/s")
            finally:
                progress.stop()


if __name__ == "__main__":
    unittest.main()
