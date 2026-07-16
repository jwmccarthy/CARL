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

### Observation space

Shape: `[n_sim, obs_dim]` float32, where `obs_dim = 9 + n_cars * 18`

| Field | Dims | Description |
|-------|------|-------------|
| Ball pos | 3 | xyz position |
| Ball vel | 3 | xyz linear velocity |
| Ball ang | 3 | xyz angular velocity |
| Car pos | 3 | xyz position (per car) |
| Car vel | 3 | xyz linear velocity (per car) |
| Car ang | 3 | xyz angular velocity (per car) |
| Car quat | 4 | rotation as quaternion (per car) |
| Car boost | 1 | boost amount 0-100 (per car) |
| Car flags | 5 | isOnGround, isDemoed, hasFlipped, hasDoubleJumped, isBoosting (per car) |

### Action space

Shape: `[n_sim, act_dim]` float32, where `act_dim = n_cars * 8`

| Field | Range | Description |
|-------|-------|-------------|
| throttle | [-1, 1] | forward/reverse (per car) |
| steer | [-1, 1] | left/right (per car) |
| pitch | [-1, 1] | nose up/down in air (per car) |
| yaw | [-1, 1] | nose left/right in air (per car) |
| roll | [-1, 1] | barrel roll in air (per car) |
| jump | {0, 1} | jump button (per car) |
| boost | {0, 1} | boost button (per car) |
| handbrake | {0, 1} | powerslide button (per car) |