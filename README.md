# CARL

CUDA-native Rocket League physics simulation for vectorized reinforcement learning

## Features

- Full car physics: suspension, tire friction, controls, jump/dodge/flip
- Collision: car-arena (SAT + clip), car-car (OBB SAT), car-ball, ball-arena
- Game systems: boost pads, demos with respawn selection, goal detection and scoring
- 1024+ parallel simulations on a single GPU
- Python bindings via pybind11 with zero-copy DLPack tensor I/O
- No host-device copies in the simulation loop

## Install

```bash
pip install .
```

For development:

```bash
pip install -e . --no-build-isolation
cmake --build build --parallel && cp build/carl*.so .
```

## Usage

```python
import carl
import torch

env = carl.Env(n_sim=1024, n_blue=4, n_orange=4, seed=0)

actions = torch.zeros((1024, env.act_dim), device="cuda")
obs = torch.utils.dlpack.from_dlpack(
    env.step(torch.utils.dlpack.to_dlpack(actions)))

print(f"obs: {obs.shape} {obs.dtype}")  # torch.Size([1024, 153]) torch.float32
```

JAX:

```python
import carl
import jax.numpy as jnp
import jax.dlpack

env = carl.Env(n_sim=1024, n_blue=4, n_orange=4, seed=0)

actions = jnp.zeros((1024, env.act_dim), dtype=jnp.float32)
obs = jax.dlpack.from_dlpack(
    env.step(jax.dlpack.to_dlpack(actions)))
```

## Observation space

Per simulation: ball (pos, vel, ang) + per car (pos, vel, ang, quat, boost, flags)

## Action space

Per car: throttle, steer, pitch, yaw, roll, jump, boost, handbrake
