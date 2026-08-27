import argparse
import os

import torch as th

from carl.gymnasium import CARLTorchVectorEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CARL block sizes.")
    parser.add_argument("--n-sim", type=int, default=8192)
    parser.add_argument("--frameskip", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=(32, 64, 128, 192, 256),
    )
    return parser.parse_args()


def measure(args: argparse.Namespace, block_size: int) -> float:
    os.environ["CARL_BLOCK_SIZE"] = str(block_size)
    env = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        frameskip=args.frameskip,
        max_ticks=1_000_000,
        normalize=True,
    )
    action = th.randint(0, 2, (env.n_envs, 7), dtype=th.int32, device=env.device)

    try:
        env.reset()
        for _ in range(args.warmup):
            env.step(action)

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.steps):
            env.step(action)
        end.record()
        end.synchronize()

        return args.steps / (start.elapsed_time(end) / 1000.0)
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    print("block  env steps/s  physics ticks/s")

    for block_size in args.block_sizes:
        steps_per_second = measure(args, block_size)
        ticks_per_second = steps_per_second * args.n_sim * args.frameskip
        print(f"{block_size:>5}  {steps_per_second:>11.1f}  {ticks_per_second:>15.0f}")


if __name__ == "__main__":
    main()
