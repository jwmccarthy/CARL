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

### Optional ego-controller observations

Pass `ego_controller=True` to `CARLTorchVectorEnv` (or native `carl.Env`) to
append controller schema v1. This adds 27 values to each actor's observation;
the default observation layout and existing prefix are unchanged. A 1v1 actor
has 166 values with this option, versus 139 without it.

`observation.ego_controller` exposes the appended block. Its offsets are:

| Offset | Field |
|---|---|
| 0–7 | Previous throttle, steer, yaw, pitch, roll, jump, boost, slide |
| 8–11 | Last-jump latch, has-jumped, is-jumping, jump time |
| 12 | Remaining minimum boost duration |
| 13–17 | Is-flipping, flip time, car-relative flip torque XYZ |
| 18 | Handbrake value |
| 19–22 | Four wheel-contact flags |
| 23–25 | Is-auto-flipping, auto-flip timer, auto-flip torque scale |
| 26 | Demo respawn timer |

Named properties include `ego_previous_controls`, `ego_last_jump`,
`ego_wheel_contacts`, and the corresponding timer/flip fields. Times remain in
seconds even with `normalize=True`. Controls and torques are car-local and are
not inverted for orange actors. The discrete action codec adjusts its dodge
window indexing automatically. Native constants `OBS_CONTROLLER_VERSION` and
`OBS_EGO_CONTROLLER` identify the schema and block size.

Restoring cars through `set_car` reconstructs suspension/wheel contacts and the
grounded flag from the restored geometry before returning observations. All car
poses are installed before this query, so car-on-car contacts are independent
of actor order. No settling physics tick is run. A supplied internal-state
last-jump latch also initializes the previous jump control; other previous
controls are reset to zero. Demoed cars expose no wheel contacts.

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
