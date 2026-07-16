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

env = carl.Env(n_sim=1024, n_blue=4, n_orange=4, seed=0)

actions = torch.zeros((1024, env.act_dim), device="cuda")
obs = torch.utils.dlpack.from_dlpack(
    env.step(torch.utils.dlpack.to_dlpack(actions))
)

print(f"obs: {obs.shape} {obs.dtype}")  # torch.Size([1024, 153]) torch.float32
```

`JAX`:

```python
import carl
import jax.numpy as jnp
import jax.dlpack

env = carl.Env(n_sim=1024, n_blue=4, n_orange=4, seed=0)

actions = jnp.zeros((1024, env.act_dim), dtype=jnp.float32)
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
| max_ticks | int | constructor | Ticks before episode ends |
| n_cars | int | readonly | Total cars per simulation (n_blue + n_orange) |
| obs_dim | int | readonly | Observation vector length |
| act_dim | int | readonly | Action vector length |

### Observation space

Shape: `[n_sim, obs_dim]` float32, where `obs_dim = 9 + n_cars * 18`

| Field | Dims | Range | Description |
|-------|------|-------|-------------|
| Ball position | [3] | | xyz position |
| Ball velocity | [3] | | xyz linear velocity |
| Ball angular velocity | [3] | | xyz angular velocity |
| Car position | [3, n_cars] | | xyz position |
| Car velocity | [3, n_cars] | | xyz linear velocity |
| Car angular velocity | [3, n_cars] | | xyz angular velocity |
| Car rotation | [4, n_cars] | | quaternion |
| Car boost | [1, n_cars] | [0, 100] | boost amount |
| Car flags | [5, n_cars] | {0, 1} | on ground, demoed, has flip, has double jump, is boosting |

### Action space

Shape: `[n_sim, act_dim]` float32, where `act_dim = n_cars * 8`

| Field | Dims | Range | Description |
|-------|------|-------|-------------|
| Throttle | [1, n_cars] | [-1, 1] | forward/reverse |
| Steer | [1, n_cars] | [-1, 1] | left/right |
| Pitch | [1, n_cars] | [-1, 1] | nose up/down in air |
| Yaw | [1, n_cars] | [-1, 1] | nose left/right in air |
| Roll | [1, n_cars] | [-1, 1] | barrel roll in air |
| Jump | [1, n_cars] | {0, 1} | jump button |
| Boost | [1, n_cars] | {0, 1} | boost button |
| Handbrake | [1, n_cars] | {0, 1} | powerslide button |

### Performance

Running `python/test.py` will display the ticks/s given a random action vector. Depending on the GPU and environment config, CARL performs between 5-50M ticks/s. By executing end-to-end on device, we further sidestep host-device transfers typical with CPU-based vectorized simulators.
