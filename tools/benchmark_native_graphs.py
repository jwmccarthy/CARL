"""Benchmark native dispatch using a physical replay-reset bank saved by torch.save.

The bank is a mapping of apply_reset_state keyword arguments with one row per
arena. Run with the candidate package on PYTHONPATH and --reset-bank PATH.
Final-output differences are reported against the first trial: parallel contact
insertion can make ordinary launches nondeterministic on large replay banks.
Submission time includes driver backpressure; stream elapsed time includes idle
gaps. Device memory deltas include lazy driver allocations on the first trial.
"""

import argparse
import json
import time
from pathlib import Path

import carl
import torch


def measure(bank, actions, args, graph):
    n_sim, n_cars = bank["car_position"].shape[:2]
    free_before, _ = torch.cuda.mem_get_info()
    env = carl.Env(n_sim, n_cars // 2, n_cars // 2, args.seed,
                   frameskip=args.frameskip, native_cuda_graphs=graph,
                   capture_events=args.capture_events,
                   capture_ball_touches=args.capture_ball_touches)
    env.apply_reset_state(**bank)
    stream = torch.cuda.ExternalStream(env.stream)
    for action in actions[:args.warmup]:
        env.step(action)
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.nvtx.range_push("native_graph" if graph else "ordinary_launches")
    start.record(stream)
    begin = time.perf_counter()
    for action in actions[args.warmup:]:
        env.step(action)
    submitted = time.perf_counter()
    end.record(stream)
    end.synchronize()
    finished = time.perf_counter()
    torch.cuda.nvtx.range_pop()

    row = {
        "native_cuda_graphs": graph,
        "graph_nodes": env.native_graph_nodes,
        "native_device_bytes": free_before - free_after,
        "host_submission_ms_per_step": (submitted - begin) * 1000 / args.steps,
        "stream_elapsed_ms_per_step": start.elapsed_time(end) / args.steps,
        "wall_ms_per_step": (finished - begin) * 1000 / args.steps,
        "decisions_per_second": args.steps * n_sim * n_cars / (finished - begin),
    }
    state = torch.from_dlpack(env.get_state()).cpu().clone()
    observation = torch.from_dlpack(env.get_obs()).cpu().clone()
    del env
    return row, (state, observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("both", "ordinary", "graph"), default="both")
    parser.add_argument("--frameskip", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--capture-events", action="store_true")
    parser.add_argument("--capture-ball-touches", action="store_true")
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error("steps, warmup and repeats must be positive")
    bank = torch.load(args.reset_bank, map_location="cuda", weights_only=True)
    n_sim, n_cars = bank["car_position"].shape[:2]
    if n_cars % 2:
        parser.error("reset bank must contain equal-sized teams")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    shape = (args.steps + args.warmup, n_sim, n_cars)
    actions = torch.stack([torch.randint(size, shape, device="cuda", generator=generator)
                           for size in carl.ACTION_NVECS], -1).int()
    modes = [False, True] if args.mode == "both" else [args.mode == "graph"]
    rows, reference = [], None
    for repeat in range(args.repeats):
        for graph in modes[::1 if repeat % 2 == 0 else -1]:
            row, final = measure(bank, actions, args, graph)
            if reference is None:
                reference = final
            for name, actual, expected in zip(("state", "observation"), final, reference):
                if not torch.isfinite(actual).all():
                    raise ValueError(f"non-finite {name} in benchmark output")
                row[f"{name}_different_fraction"] = (actual != expected).float().mean().item()
                row[f"{name}_max_abs_difference"] = (actual - expected).abs().max().item()
            if args.require_exact:
                torch.testing.assert_close(final, reference, rtol=0, atol=0)
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
