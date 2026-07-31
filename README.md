# CARL

CARL is a CUDA Rocket League Soccar physics simulator. Simulation state and native input and output buffers are stored in CUDA memory and exposed through DLPack.

CARL uses ideas and constants from [RocketSim](https://github.com/ZealanL/RocketSim). Its vector environment design was informed by [PureJaxRL](https://chrislu.page/blog/meta-disco/).

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

Actors use simulation order, then car order. Blue cars come before orange cars. `reset()` returns only the observation tensor. `step()` returns the Gymnasium five value result.

Default rewards are relative to each actor. Scoring gives `1`. Conceding gives `-1`. Every car in a completed simulation terminates and resets together.

`info["reward"]` and `info["length"]` contain Python lists for completed actors. Lengths count physics ticks. Same-step autoreset also exposes the pre-reset observation as `info["final_obs"]`, masked by `info["_final_obs"]`.

## Actions

The Torch action shape is `[n_envs, 7]`. The native action shape is `[n_sim, n_cars, 7]`.

```text
[horizontal, vertical, throttle, powerslide, boost, air_roll, jump]
```

Horizontal, vertical, throttle, and air roll use `0` for neutral, `1` for the first direction, and `2` for the opposite direction. Powerslide, boost, and jump use `0` for off and `1` for on.

The action cardinalities are:

```python
[3, 3, 3, 2, 2, 3, 2]
```

`CARLTorchVectorEnv.action_codec` can mask invalid logits for `MultiCategoricalPolicy`. Native `step()` does not enforce this mask.

Controls are held for every tick in `frameskip`. Episode completion is checked after those ticks.

`no_touch_timeout_ticks` optionally truncates an episode after that many 120 Hz physics ticks without any car touching the ball. `no_touch_timeout_seconds` provides the same setting in seconds; specify only one. Both default to disabled. Like `max_ticks`, the condition is packed and reset natively with same-step autoreset.

## Observations

The native observation shape is `[n_sim, n_cars, obs_dim]` with `float32` values.

Set `normalize=True` to normalize observations during CUDA packing using arena, ball, car, boost, and angular-speed limits. Reward state and reset setters remain in raw physics units.

```text
obs_dim = 9 + n_cars * 21 + 68 + 6 + (n_cars - 1) * 6 + 6
        = 83 + n_cars * 27
```

Each observation contains ball position, velocity, and angular velocity. It then contains position, velocity, angular velocity, forward direction, up direction, boost, and state flags for every car, followed by boost pad state and distance. These existing fields retain their original ordering.

The appended fields contain ego-to-ball relative position and velocity, then ego-to-car relative position and velocity for every other car in the same teammate/opponent order, then ball-to-own-goal and ball-to-opponent-goal vectors. Goal identity is relative to the observing team, and orange vectors use the same optional rotation as the base observation.

Car blocks start with the observing car, followed by teammates and opponents. Orange observations can be rotated into the blue frame with `invert_orange=True`.

## Custom Rewards

Reward functions receive a `RewardContext` and return `[n_sim, n_cars]`.

```python
def speed_reward(context):
    return context.current.car_velocity.norm(dim=-1) / 2300.0


env = CARLTorchVectorEnv(1024, 1, 1, reward_funcs=[speed_reward])
```

The context contains previous state, current transition state, actions, score events, termination flags, and truncation flags. Reward functions are summed and multiplied by `reward_scale`. Custom rewards replace the default score reward.

## State Setters

`set_ball()` changes ball position, velocity, and angular velocity. `set_car()` changes car position, rotation, velocity, angular velocity, demo state, and optional boost.

Full updates use these shapes:

```text
ball vectors       [n_sim, 3] float32
car vectors        [n_sim, n_cars, 3] float32
car rotation       [n_sim, n_cars, 4] float32
car demo state     [n_sim, n_cars] bool or int32
car boost          [n_sim, n_cars] float32
```

Rotations use quaternion order `(x, y, z, w)`. Boost is limited to the range from `0` to `100`.

Selected updates pass `simulation_indices` as contiguous `int64` values on `cuda:0`. State tensors then use the selection count as their first dimension. Indices must be unique and valid. Duplicate indices race. Invalid indices are skipped.

Setters do not reset scores, episode time, controls, jump state, contacts, or wrapper statistics. Native setters return `None`. Torch setters return refreshed observations and refresh the custom reward baseline.

## Reset State Provider

`CARLTorchVectorEnv` accepts an optional `reset_state_provider`. The provider receives a Boolean CUDA mask shaped `[n_sim]` and returns compact tensors in `mask.nonzero()` order.

The required mapping keys are `ball_position`, `ball_velocity`, `ball_angular_velocity`, `car_position`, `car_rotation`, `car_velocity`, `car_angular_velocity`, and `car_demoed`. `car_boost` is optional. A provider can return compact `simulation_indices` to override only part of the reset mask. Returning `None` keeps the normal kickoff state.

On explicit reset, every mask value is true. During same step reset, the mask marks completed simulations. Terminal custom rewards use the transition before reset. Returned observations use the provider state.

## Native API

The native environment accepts CUDA arrays through DLPack.

```python
import carl
import torch


env = carl.Env(n_sim=1024, n_blue=1, n_orange=1, seed=0, frameskip=8)
actions = torch.zeros((1024, 2, 7), dtype=torch.int32, device="cuda:0")
observation = torch.from_dlpack(env.step(actions))
```

The native API provides `reset`, `step`, state getters, reward and done getters, state setters, `frameskip`, `max_ticks`, `no_touch_timeout_ticks`, observation dimensions, action dimensions, action cardinalities, simulation count, and car count. Native `no_touch_timeout_ticks=0` disables the timeout.
