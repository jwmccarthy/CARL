# CARL

CARL is a GPU-based Rocket League simulator for reinforcement learning, based on
[RocketSim](https://github.com/ZealanL/RocketSim).

## Install

You need Linux and Python 3.10+. NVIDIA builds (the default) require the CUDA
12.8 toolkit and a compatible C++ compiler such as GCC 12. AMD builds are
opt-in via `CARL_GPU_BACKEND=HIP` and require a ROCm-supported GPU, the ROCm
development toolkit with hipCUB and rocPRIM, and a matching ROCm build of PyTorch.

For the default NVIDIA backend, from the repository root:

```bash
uv sync
```

If `nvidia-smi` cannot detect your GPU, set its CUDA architecture first, for
example `export CARL_CUDA_ARCHITECTURES=89`.

## Run an environment

```python
import torch
from carl.gymnasium import CARLTorchVectorEnv

env = CARLTorchVectorEnv(n_sim=8192, n_blue=1, n_orange=1)
observation = env.reset()

actions = torch.zeros(env.action_space.shape, device=env.device)
observation, reward, terminated, truncated, info = env.step(actions)

env.close()
```

Each car is an actor. Cars are grouped by simulation.

## Custom rewards

Reward functions return a tensor of shape `[n_sim, n_cars]`:

```python
import carl
from carl.gymnasium import CARLTorchVectorEnv

def speed_reward(context):
    return context.current.car_velocity.norm(dim=-1) / carl.CAR_MAX_SPEED

env = CARLTorchVectorEnv(1024, 1, 1, reward_funcs=[speed_reward])
```

Custom rewards replace the default score reward.

## Replay resets

`CARLTorchVectorEnv(reset_state_provider=...)` applies both `CARLResetState`
and mapping-based resets through the native `Env.apply_reset_state` operation.
It restores selected ball/car poses, velocities, demo flags, optional boost and
controller state, resets boost pads, and optionally restores match scores/clock.
Suspension contacts are reconstructed after every selected car pose is installed;
restoration does not advance physics or generate touches.

The native operation accepts contiguous CUDA tensors with the same layouts as
`set_ball` and `set_car`, using `ball_`/`car_` argument prefixes. Optional
`simulation_indices` is an int64 tensor; omitted indices select every arena.
`blue_score`, `orange_score`, and `episode_ticks` must be supplied together.
Omitted boost/controller/match fields retain the standalone setters' semantics.
Current observations, canonical state, and match outputs are packed once after
restoration. Pre-reset transition outputs, rewards, and done flags are preserved.
An empty selection is a no-op. Selected indices must be unique and in range.

Inputs are validated before mutation and retained until one final native stream
synchronization completes. This consolidates the setters' waits; GPU-mask reset
selection and fully deferred reporting are separate follow-up optimizations.
Standalone setters remain available.

## Optional arena metadata

To build the RocketSim metadata exporter:

```bash
git submodule update --init --recursive
uv sync --extra rlgym
cmake -S . -B build -DCARL_BUILD_PYTHON=OFF -DCARL_BUILD_ARENA_METADATA=ON \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --target carl_export_arena_metadata
```

Generate metadata using the meshes installed by RLGym:

```bash
MESHES=$(.venv/bin/python -c 'from importlib.metadata import distribution; print(distribution("rlgym-rocket-league").locate_file("rlgym/rocket_league/sim/collision_meshes"))')
build/carl_export_arena_metadata "$MESHES" native_triangles.txt
.venv/bin/python tools/arena_metadata.py assets/arena.obj native_triangles.txt arena_metadata.txt
export CARL_ARENA_METADATA="$PWD/arena_metadata.txt"
```

Without this file, CARL uses its default arena setup. Normal builds do not need
RocketSim or RLGym.
