"""Rich/TensorBoard reporting, plus a JSONL bridge for already-running trainers.

Backfill or follow existing runs without restarting their training process:
    python -m examples.training_packs.reporting --logdir runs/training-pack-ppo --watch
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID
    from torch.utils.tensorboard import SummaryWriter


def open_writer(
    directory: Path,
    config:    Mapping[str, object],
    *,
    suffix:    str = ".native",
    required:  bool = False,
) -> SummaryWriter | None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        message = "TensorBoard logging requires `uv sync --extra training`; JSONL logging is still available"
        if required:
            raise RuntimeError(message) from error
        warnings.warn(message, stacklevel=2)
        return None

    writer = SummaryWriter(str(directory / "tensorboard"), flush_secs=5, filename_suffix=suffix)
    writer.add_text("configuration", "```json\n" + json.dumps(dict(config), indent=2) + "\n```", 0)
    writer.flush()

    return writer


def progress_bar(
    total:   int,
    initial: int = 0,
) -> tuple[Progress | None, TaskID | None]:
    try:
        from rich import progress as rich_progress
    except ImportError:
        warnings.warn("Rich progress requires `uv sync --extra training`", stacklevel=2)
        return None, None

    progress = rich_progress.Progress(
        rich_progress.TextColumn("[bold]{task.description}"),
        rich_progress.BarColumn(bar_width=None),
        rich_progress.TaskProgressColumn(),
        rich_progress.MofNCompleteColumn(),
        rich_progress.TextColumn("{task.fields[rate]}", justify="right"),
        rich_progress.TimeElapsedColumn(),
        rich_progress.TimeRemainingColumn(),
    )
    task = progress.add_task("Aerial-pack PPO", total=total, completed=min(initial, total), rate="")
    progress.start()

    return progress, task


def log_metrics(
    writer:   SummaryWriter | None,
    step:     int,
    metrics:  Mapping[str, float | int],
    progress: Progress | None = None,
    task:     TaskID | None = None,
) -> None:
    if writer is not None:
        for name, value in metrics.items():
            if name != "steps":
                writer.add_scalar(name, value, step)
        writer.flush()
    rate = metrics.get("transitions_per_second", 0)
    if progress is not None and task is not None:
        progress.update(task, completed=min(step, progress.tasks[task].total), rate=f"{rate:,.0f} steps/s")
    else:
        print(f"step={step:,} reward={metrics.get('reward_mean', 0):.3f} steps/s={rate:,.0f}", flush=True)


class JsonlBridge:
    """Incrementally import complete JSONL records; restarts do not duplicate steps."""

    def __init__(self, path: Path) -> None:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        self.path = path
        self.offset = 0
        directory = path.parent
        event_directory = directory / "tensorboard"
        if any(event_directory.glob("events.out.tfevents.*.native")):
            raise ValueError("this run already writes native TensorBoard summaries")
        self.last_step = -1
        if any(event_directory.glob("events.out.tfevents.*")):
            events = EventAccumulator(str(event_directory), size_guidance={"scalars": 1}).Reload()
            self.last_step = max((events.Scalars(tag)[-1].step for tag in events.Tags()["scalars"]), default=-1)
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.writer = open_writer(directory, config, suffix=".jsonl", required=True)

    def sync(self) -> int:
        imported = 0
        if self.path.stat().st_size < self.offset:
            self.offset = 0
        with self.path.open("rb") as source:
            source.seek(self.offset)
            while line := source.readline():
                if not line.endswith(b"\n"):
                    break  # A live writer may not have completed this record yet.
                if line.strip():
                    metrics = json.loads(line)
                    step = int(metrics["steps"])
                    if step > self.last_step:
                        for name, value in metrics.items():
                            if name != "steps":
                                self.writer.add_scalar(name, value, step)
                        self.last_step = step
                        imported += 1
                self.offset = source.tell()
        self.writer.flush()

        return imported

    def close(self) -> None:
        self.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, default=Path("runs/training-pack-ppo"))
    parser.add_argument("--watch", action="store_true", help="follow old trainers and discover new runs")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be positive and finite")
    bridges: dict[Path, JsonlBridge] = {}

    try:
        while True:
            for path in args.logdir.glob("*/metrics.jsonl"):
                if any((path.parent / "tensorboard").glob("events.out.tfevents.*.native")):
                    continue
                if path not in bridges:
                    bridges[path] = JsonlBridge(path)
                imported = bridges[path].sync()
                if imported:
                    print(f"TensorBoard: {path.parent.name}: imported {imported} updates", flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        for bridge in bridges.values():
            bridge.close()


if __name__ == "__main__":
    main()
