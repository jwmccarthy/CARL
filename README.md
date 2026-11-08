# CARL

CARL is a GPU-based Rocket League simulator for reinforcement learning, based on
[RocketSim](https://github.com/ZealanL/RocketSim).

## Install

You need Linux, Python 3.10+, an NVIDIA GPU, the CUDA 12.8 toolkit, and a
compatible C++ compiler such as GCC 12. From the repository root:

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
