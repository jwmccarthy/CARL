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

### Ego-controller observations

Every observation includes the 13-value ego-controller schema v2 block. It
retains controls and state available through RLBot V5, without simulator-only
timers, torques, wheel-contact bits or auto-flip state. A 1v1 actor has 152 values;
the preceding 139-value physics/relative/dodge prefix is unchanged. There is no
opt-out flag. Models trained with 139 inputs or the earlier 166-value controller
schema need a new compatible input layer/checkpoint.

`observation.ego_controller` exposes the appended block. Its offsets are:

| Offset | Field |
|---|---|
| 0–7 | Previous throttle, steer, yaw, pitch, roll, jump, boost, slide |
| 8 | Has-jumped |
| 9 | Is-jumping |
| 10 | Is-flipping |
| 11 | Elapsed flip time |
| 12 | Demo respawn timer |

Named properties are `ego_previous_controls`, `ego_has_jumped`, `ego_is_jumping`,
`ego_is_flipping`, `ego_flip_time` and `ego_demo_respawn_time`. Times remain in
seconds even with `normalize=True`. Controls are car-local and are not inverted
for orange actors. The discrete action codec uses the dodge window preceding
the controller block. Native constants `OBS_CONTROLLER_VERSION` (2) and
`OBS_EGO_CONTROLLER` (13) identify the schema and block size.

Restoring cars through `set_car` reconstructs suspension/wheel contacts and the
grounded flag from the restored geometry before returning observations. All car
poses are installed before this query, so car-on-car contacts are independent
of actor order. No settling physics tick is run. A supplied internal-state
last-jump latch also initializes the previous jump control; other previous
controls are reset to zero. Demoed cars retain no internal wheel contacts.
Contact reconstruction and internal jump/boost/flip state restoration still run;
removing their observation fields does not remove the reset fixes.

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
