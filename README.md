# CARL

CUDA-Accelerated Rocket League (CARL) is a Rocket League physics simulator based on [RocketSim](https://github.com/ZealanL/RocketSim). Inspired by [PureJaxRL](https://chrislu.page/blog/meta-disco/), it is a vectorized environment for reinforcement learning in Rocket League that resides entirely on the GPU.

## Requirements

CARL requires Python 3.10 or newer, Linux, an NVIDIA GPU and driver, and the CUDA Toolkit for source builds.

Install the current source and Torch wrapper with:

```bash
uv sync --extra gymnasium
```

## Torch Environment

`CARLTorchVectorEnv` exposes each car as one actor. Cars remain grouped inside each physics simulation.

```python
import torch
from carl.gymnasium import CARLTorchVectorEnv

env = CARLTorchVectorEnv(
    n_sim=1024,
    n_blue=1,
    n_orange=1,
    frameskip=8,
    max_ticks=4096,
    no_touch_timeout_seconds=30.0,
    normalize=True,
)

observation = env.reset()
actions = torch.zeros((env.n_envs, 7), dtype=torch.int32, device="cuda:0")
observation, reward, terminated, truncated, info = env.step(actions)
```

Actors use simulation order, then car order. Blue cars come before orange cars. `reset()` returns only the observation tensor. `step()` returns the Gymnasium five-value result.

`info["reward"]` and `info["length"]` contain Python lists for completed actors. Lengths count physics ticks. Same-step autoreset also exposes the pre-reset observation as `info["final_obs"]`, masked by `info["_final_obs"]`.

## Actions

The Torch action shape is `[n_envs, 7]`. The native action shape is `[n_sim, n_cars, 7]`.

```text
[horizontal, vertical, throttle, powerslide, boost, air_roll, jump]
```

Horizontal, vertical, throttle, and air roll use `0` for neutral, `1` for the first direction, and `2` for the opposite direction. Powerslide, boost, and jump use `0` for off and `1` for on.

Controls are held for every tick in `frameskip`. Episode completion is checked after those ticks.

`no_touch_timeout_ticks` optionally truncates an episode after that many 120 Hz physics ticks without any car touching the ball. `no_touch_timeout_seconds` provides the same setting in seconds; specify only one. Both default to disabled. Like `max_ticks`, the condition is packed and reset natively with same-step autoreset.

## Observations

The native observation shape is `[n_sim, n_cars, obs_dim]` (all `f32`).

Set `normalize=True` to normalize observations during CUDA packing using arena, ball, car, boost, and angular-speed limits. Reward state and reset setters remain in raw physics units.

Each observation contains ball position, velocity, and angular velocity. It then contains position, velocity, angular velocity, forward direction, up direction, boost, and state flags for every car, followed by boost pad state and distance. These existing fields retain their original ordering.

The appended fields contain self-to-ball relative position and velocity, then self-to-car relative position and velocity for every other car in the same teammate/opponent order, then ball-to-own-goal and ball-to-opponent-goal vectors. Goal identity is relative to the observing team, and orange vectors use the same optional rotation as the base observation.

Car blocks start with the observing car, followed by teammates and opponents. Orange observations can be rotated into the blue frame with `invert_orange=True`.

## Custom Rewards

The environment carries a default, zero-sum reward tied to goals, but rewards are customizable via the exposed `RewardContext`.

Reward functions receive a `RewardContext` and return `[n_sim, n_cars]`.

```python
def speed_reward(context):
    return context.current.car_velocity.norm(dim=-1) / 2300.0

env = CARLTorchVectorEnv(1024, 1, 1, reward_funcs=[speed_reward])
```

The context contains previous state, current transition state, actions, score events, termination flags, and truncation flags. Reward functions are summed and multiplied by `reward_scale`. Custom rewards replace the default score reward.
