# CARL: CUDA-Accelerated Rocket League

CARL is a vectorized Rocket League physics simulator based on [RocketSim](https://github.com/ZealanL/RocketSim) and inspired by [PureJaxRL](https://chrislu.page/blog/meta-disco/). Intended for highly-parallel reinforcement learning.

## Features
- Full Rocket League physics simulation (car-{car, ball, arena} collision response, boost impulses, ground-suspension interaction, etc.)
- DLPack I/O for generalized device action & observations (i.e. torch.utils.dlpack, jax.dlpack)
- Manages game state entirely on device - no host-device transfer overhead required

## Usage

`PyTorch`:

```python
import carl
import torch

env = carl.Env(
    n_sim=1024, n_blue=4, n_orange=4, seed=0,
    skip_ticks=8, invert_orange=True
)

actions = torch.zeros((1024, env.n_cars, 7), dtype=torch.int32, device="cuda")
obs = torch.utils.dlpack.from_dlpack(
    env.step(torch.utils.dlpack.to_dlpack(actions))
)

print(f"obs: {obs.shape} {obs.dtype}")  # torch.Size([1024, 8, 177]) torch.float32
```

`JAX`:

```python
import carl
import jax.numpy as jnp
import jax.dlpack

env = carl.Env(n_sim=1024, n_blue=4, n_orange=4, seed=0)

actions = jnp.zeros((1024, env.n_cars, 7), dtype=jnp.int32)
obs = jax.dlpack.from_dlpack(
    env.step(jax.dlpack.to_dlpack(actions))
)
```

### Environment properties

| Property | Type | Set | Description |
|----------|------|-----|-------------|
| n_sim | int | constructor | Number of parallel simulations |
| n_blue | int | constructor | Blue team size per simulation |
| n_orange | int | constructor | Orange team size per simulation |
| seed | int | constructor | Random seed |
| skip_ticks | int | constructor, read/write | Physics ticks per controller input (default 1) |
| invert_orange | bool | constructor | Rotate orange observations into the blue frame (default true) |
| max_ticks | int | constructor | Ticks before episode ends |
| n_cars | int | readonly | Total cars per simulation (n_blue + n_orange) |
| obs_dim | int | readonly | Observation vector length |
| act_dim | int | readonly | Action vector length |
| action_nvec | list | readonly | MultiDiscrete cardinalities, shaped [n_cars, 7] |

### Observation space

Shape: `[n_sim, n_cars, obs_dim]` float32, where `obs_dim = 9 + n_cars * 21`.
For each observer, car blocks are ordered as self, remaining teammates by index,
then opponents by index.

When `invert_orange=True`, orange observers see all positions, velocities,
angular velocities, and orientation vectors rotated 180 degrees around the
vertical axis.

| Field | Dims | Range | Description |
|-------|------|-------|-------------|
| Ball position | [3] | | xyz position |
| Ball velocity | [3] | | xyz linear velocity |
| Ball angular velocity | [3] | | xyz angular velocity |
| Car position | [n_cars, 3] | | xyz position |
| Car velocity | [n_cars, 3] | | xyz linear velocity |
| Car angular velocity | [n_cars, 3] | | xyz angular velocity |
| Car forward | [n_cars, 3] | [-1, 1] | forward unit vector |
| Car up | [n_cars, 3] | [-1, 1] | up unit vector |
| Car boost | [n_cars] | [0, 100] | boost amount |
| Car flags | [n_cars, 5] | {0, 1} | on ground, demoed, has flip, has double jump, is boosting |

### Action space

Shape: `[n_sim, n_cars, 7]` int32. For simulation `s` and car `c`, `actions[s, c]` is read in this order:

```text
[horizontal, vertical, throttle, powerslide, boost, air_roll, jump]
```

| Index | Field | Cardinality | Values | Description |
|-------|-------|-------------|--------|-------------|
| 0 | Horizontal | 3 | 0 none, 1 left, 2 right | ground steer and aerial yaw |
| 1 | Vertical | 3 | 0 none, 1 forward, 2 back | aerial pitch and dodge direction |
| 2 | Throttle | 3 | 0 none, 1 forward, 2 reverse | drive direction |
| 3 | Powerslide | 2 | 0 off, 1 on | powerslide button |
| 4 | Boost | 2 | 0 off, 1 on | boost button |
| 5 | Air roll | 3 | 0 none, 1 left, 2 right | directional air roll |
| 6 | Jump | 2 | 0 off, 1 on | jump button |

Every field is decoded independently. For example, this applies left steering, forward pitch, forward throttle, boost, and jump to car 0 in every simulation:

```python
actions[:, 0] = torch.tensor(
    [1, 1, 1, 0, 1, 0, 1],
    dtype=torch.int32,
    device="cuda",
)
```

Horizontal and vertical are separate fields, so diagonal steering and dodges are possible. Opposing inputs within one field, such as left and right, are mutually exclusive.

Each call to `step` applies the supplied controls before the first physics tick and holds them for all `skip_ticks` ticks. Observations, rewards, and dones describe the final tick; touches indicate whether contact occurred during any aggregated tick. Set `skip_ticks=1` to request controls every physics tick.

Episodes end when either team scores or `max_ticks` is reached. State can be replaced directly from contiguous CUDA tensors:

```python
env.set_ball(position, velocity, angular_velocity)  # [n_sim, 3] float32
env.set_car(
    position,          # [n_sim, n_cars, 3] float32
    rotation,          # [n_sim, n_cars, 4] float32 quaternion (x, y, z, w)
    velocity,          # [n_sim, n_cars, 3] float32
    angular_velocity,  # [n_sim, n_cars, 3] float32
    demoed,            # [n_sim, n_cars] bool or int32
)
```

`env.action_nvec` contains the complete cardinality array, repeated for every car:

```python
assert env.action_nvec == [[3, 3, 3, 2, 2, 3, 2]] * env.n_cars
```

It can be used directly to describe one simulation's action space with Gymnasium:

```python
import gymnasium as gym
import numpy as np

action_space = gym.spaces.MultiDiscrete(
    np.asarray(env.action_nvec, dtype=np.int32),
    dtype=np.int32,
)
```

### Performance

Running `python/test.py` will display the ticks/s given an action vector. Depending on the GPU and environment config, CARL simulates between 5-50M ticks/s. By executing end-to-end on device, we also sidestep host-device transfer latency inherent to CPU-based vectorized simulators.
