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
| Ball position | [3] | xyz position |
| Ball velocity | [3] | xyz linear velocity |
| Ball angular velocity | [3] | xyz angular velocity |
| Car position | [3, n_cars] | xyz position |
| Car velocity | [3, n_cars] | xyz linear velocity |
| Car angular velocity | [3, n_cars] | xyz angular velocity |
| Car rotation | [4, n_cars] | quaternion |
| Car boost | [1, n_cars] | boost amount 0-100 |
| Car flags | [5, n_cars] | on ground, demoed, has flip, has double jump, is boosting |

### Action space

Shape: `[n_sim, act_dim]` float32, where `act_dim = n_cars * 8`

| Field | Dims | Description |
|-------|------|-------------|
| Throttle | [1, n_cars] | forward/reverse, [-1, 1] |
| Steer | [1, n_cars] | left/right, [-1, 1] |
| Pitch | [1, n_cars] | nose up/down in air, [-1, 1] |
| Yaw | [1, n_cars] | nose left/right in air, [-1, 1] |
| Roll | [1, n_cars] | barrel roll in air, [-1, 1] |
| Jump | [1, n_cars] | jump button, {0, 1} |
| Boost | [1, n_cars] | boost button, {0, 1} |
| Handbrake | [1, n_cars] | powerslide button, {0, 1} |