# CARL

CUDA-Accelerated Rocket League (CARL) is a Rocket League physics simulator based on [RocketSim](https://github.com/ZealanL/RocketSim). Inspired by [PureJaxRL](https://chrislu.page/blog/meta-disco/), it is a vectorized environment for reinforcement learning in Rocket League that resides entirely on the GPU.

## Requirements

- Linux, Python 3.10+, and an NVIDIA GPU with a CUDA 12.8-compatible driver.
- Source builds require the CUDA 12.8 toolkit and GCC 12.

## Building from Source

Prebuilt wheels are not yet published. With CUDA 12.8 and GCC 12 (or an equally compatible compiler) installed, run `uv sync` from the repository root:

Set `CARL_CUDA_ARCHITECTURES` (for example, `89`) if `nvidia-smi` is unavailable.

### RocketSim metadata exporter

RocketSim is an optional submodule used by the arena metadata exporter and
RocketSim-based parity tooling. Install RLGym, initialize the submodule, and
build the exporter using the same Python environment:

```bash
git submodule update --init --recursive
uv sync --extra rlgym
cmake -S . -B build -DCARL_BUILD_PYTHON=OFF -DCARL_BUILD_ARENA_METADATA=ON \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --target carl_export_arena_metadata
```

The exporter requires RocketSim collision mesh files, which RLGym includes.
If the selected Python environment does not contain those meshes, CMake skips
the exporter and CARL continues to use its default arena setup. Locate the mesh
directory and pass it to the exporter, then match the output to the CARL arena OBJ:

```bash
MESHES=$(.venv/bin/python -c 'from importlib.metadata import distribution; print(distribution("rlgym-rocket-league").locate_file("rlgym/rocket_league/sim/collision_meshes"))')
build/carl_export_arena_metadata "$MESHES" native_triangles.txt
python tools/arena_metadata.py /path/to/arena.obj native_triangles.txt arena_metadata.txt
```

Set `CARL_ARENA_METADATA` to the resulting metadata file when running CARL.
Normal CARL builds do not require the submodule.

## Torch Environment

`CARLTorchVectorEnv` exposes each car as one actor. Cars remain grouped inside each physics simulation.

```python
import torch
from carl.gymnasium import CARLTorchVectorEnv

env = CARLTorchVectorEnv(n_sim=8192, n_blue=1, n_orange=1)

observation = env.reset()

actions = torch.zeros(
    env.action_space.shape,
    dtype=torch.float32,
    device=env.device
)

observation, reward, terminated, truncated, info = env.step(actions)
```

## Custom Rewards

Rewards are customizable via the exposed `RewardContext`.

Reward functions receive a `RewardContext` and return `[n_sim, n_cars]`.

```python
import carl

def speed_reward(context):
    return context.current.car_velocity.norm(dim=-1) / carl.CAR_MAX_SPEED

env = CARLTorchVectorEnv(1024, 1, 1, reward_funcs=[speed_reward])
```

The context contains previous state, current transition state, actions, score events, termination flags, and truncation flags. Reward functions are summed and multiplied by `reward_scale`. Custom rewards replace the default score reward.
