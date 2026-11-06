# CARL

CUDA-Accelerated Rocket League (CARL) is a Rocket League physics simulator based on [RocketSim](https://github.com/ZealanL/RocketSim). Inspired by [PureJaxRL](https://chrislu.page/blog/meta-disco/), it is a vectorized environment for reinforcement learning in Rocket League that resides entirely on the GPU.

## Requirements

- Linux, Python 3.10+, and an NVIDIA GPU with a CUDA 12.8-compatible driver.
- Source builds require the CUDA 12.8 toolkit and GCC 12.

## Building from Source

Prebuilt wheels are not yet published. With CUDA 12.8 and GCC 12 (or an equally compatible compiler) installed, run `uv sync` from the repository root:

Set `CARL_CUDA_ARCHITECTURES` (for example, `89`) if `nvidia-smi` is unavailable.

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

Rewards are customizable via the exposed `StepContext`.

Reward functions receive a `StepContext` and return `[n_sim, n_cars]`.

```python
import carl

def speed_reward(context):
    return context.current.car_velocity.norm(dim=-1) / carl.CAR_MAX_SPEED

env = CARLTorchVectorEnv(1024, 1, 1, reward_funcs=[speed_reward])
```

The context contains previous state, current transition state, actions, score events, termination flags, and truncation flags. Reward functions are summed and multiplied by `reward_scale`. Custom rewards replace the default score reward.

## Events

```python
import torch
from carl.events import ContactSurface
from carl.gymnasium import CARLTorchVectorEnv

env = CARLTorchVectorEnv(
    n_sim=1024, n_blue=1, n_orange=0, frameskip=8,
    capture_events=True, capture_ball_touches=True,
)
env.reset()
actions = torch.zeros(env.action_space.shape, device=env.device)
actions[:, -1] = 1
env.step(actions)

events = env.events                 # Also context.events in rewards
jumped = events.ego.jumped           # bool [simulation], over the frameskip window
touched = events.ego.ball_touched    # bool [simulation]
scored = events.ball.scored          # bool [simulation]
touches = events.ball_touches        # Per-car surface masks over the window
roof_touch = touches.on_surface(ContactSurface.ROOF)
saved = events.snapshot()            # Retain GPU data beyond the next step/reset

env.close()
```

Touch surfaces are accumulated per car over the whole frameskip window. Multiple
surface bits can be set; individual contacts, timestamps, and contact geometry are
not retained. The views are reused on the next step or reset; use `snapshot()` to
retain them.

For car surface events, `SIDE_WALL` refers to flat vertical side-wall contact,
not the curved floor transition. Side-dominant upward ramps count as `FLOOR`,
downward ramps as `CEILING`, and non-axis-aligned vertical side-dominant corners
as `OTHER`. Ball and back-wall surface classification are unchanged.
Car body contacts against downward-facing ceiling surfaces are also recorded in
`Event.CEILING_BODY_CONTACT`; the training-condition evaluator combines these
with wheel-support events for `SurfaceContact(surface=ArenaSurface.CEILING)`.

For cars, `LeaveSurface()` without a surface matches loss of all wheel support;
surface-qualified leave conditions also match transitions between surfaces.
Stage deadlines normally start on stage entry. Set `from_exercise_start=True`
with `within_ticks` to anchor a deadline to the exercise reset instead. Events
are aggregated per frameskip window, so windows crossing a deadline are
conservatively rejected and consecutive stages cannot advance in one window.

`BallTouch(min_speed_gain=500, max_speed_gain=1500)` optionally restricts the
signed change in ball speed in game units per second, with inclusive bounds. Either bound
may be omitted and a `surface=ContactSurface.FRONT` filter may be combined with
them. Touch capture retains surface and speeds together per physics tick, so a
different touch in the same frameskip window cannot satisfy the bounds. Speed
change is measured from immediately before the car-ball solve to after the tick's
collision impulses and speed clamp; simultaneous collisions in that tick may
also contribute. Negative values mean the ball slowed down. This is a change in
speed, not acceleration in units per second squared or a change in velocity
direction. Ball-touch speed-change conditions require
`capture_ball_touches=True`.

A stage can specify `repeat_on=condition` to match an optional recurrent event
without advancing the stage. A matched repeat renews that stage's deadline;
failure takes precedence, and stage completion takes precedence over a repeat
in the same frameskip window. `TrainingStep.repeated` reports the repeat mask;
the caller decides how to reward it. Repeat conditions are exposed in
`TrainingTarget.repeats`.
Set `max_repeats` to a nonnegative limit to fail on an additional repeat; the
count and limit are included in `TrainingTarget`. For speed-qualified touches,
multiple qualifying ticks within a window are counted individually. A stage
completion takes precedence over repeats in the same window.
`max_total_matches` instead tracks qualifying touch ticks across successive
stages that specify it, failing when their exercise-wide count exceeds the limit.

## Exercise Resets

An exercise can provide randomized 1v0 starting states on reset:

```python
from carl.events import Jump
from carl.gymnasium import CARLTorchVectorEnv
from carl.training import (
    BallState, CarState, Exercise, ExerciseResetProvider, InitialState,
    OnSurface, Stage, Uniform,
)

exercise = Exercise(
    name = "jump",
    initial = InitialState(
        ego  = CarState(placement=OnSurface(position=(0, -1000)), boost=Uniform(50, 100)),
        ball = BallState(position=(0, 0, 500)),
    ),
    stages = [Stage(Jump(), within_ticks=120)],
)
provider = ExerciseResetProvider(exercise, n_sim=1024, seed=7)
env = CARLTorchVectorEnv(n_sim=1024, n_blue=1, n_orange=0, reset_state_provider=provider)
observation = env.reset()
env.close()
```

The provider generates a fixed-size batch and applies it only to requested
simulations. Unselected entries use a `-1` index and are ignored by CARL. The
random stream advances for the full batch on each call, including empty masks.
It supplies starting states; it does not yet advance exercise stages or terminate
episodes when they finish. Reset providers must supply valid state and indices;
the environment no longer checks their contents.

Surface placement uses the flat Soccar floor, side-wall, back-wall, and ceiling
planes. It does not project positions onto curved ramps, corners, or goal openings;
choose positions on the flat portion of the intended surface.

Native state setters enqueue work asynchronously. When using the native API
directly, call `env.events.acquire(torch.cuda.current_stream().cuda_stream)`
before consuming native output tensors on a PyTorch CUDA stream. Setter inputs
must not be modified until the queued work has completed.

## Training Packs

Training packs select an exercise independently for each simulation. Their
compiled stages and runtime progress remain on the GPU:

```python
from carl.training import (
    CompiledTrainingPack, TrainingPack, TrainingPackResetProvider,
    TrainingPackRuntime, TrainingPackTerminationManager,
)

pack = TrainingPack(name="practice", exercises=[exercise])
provider = TrainingPackResetProvider(pack, n_sim=1024, seed=7)
compiled = CompiledTrainingPack.compile(pack)
runtime = TrainingPackRuntime(compiled, n_sim=1024)
manager = TrainingPackTerminationManager(runtime, provider)

env = CARLTorchVectorEnv(
    n_sim                = 1024,
    n_blue               = 1,
    n_orange             = 0,
    capture_events       = True,
    reset_state_provider = provider,
    termination_manager  = manager,
)
observation = env.reset()
```

Exercise success or failure terminates an episode; stage deadlines truncate it.
The step `info` contains per-simulation GPU tensors named `exercise_success`,
`exercise_failure`, and `exercise_timeout`. Native goals and timeouts keep their
normal termination behavior. The provider samples every exercise for the full
batch before selecting each simulation's start, so cost grows with pack size.
A model-specific sequence encoder belongs to the caller. `runtime.target_sequence()`
returns the full padded sequence for each simulation as GPU tensors, without
removing completed stages:

```python
import torch

target = runtime.target_sequence()
rows = torch.arange(runtime.n_sim, device=runtime.stage_index.device)
next_kind = target.stages.kind[rows, target.stage_index]
remaining = target.ticks_remaining
```

`target.stages` has shape `[n_sim, max_stages]` and `target.failures` has shape
`[n_sim, max_stages, max_failures]`. Each contains `valid`, `kind`, `entity`,
`event_mask`, `arena_surface_mask`, `touch_surface_mask`, `state_metric`,
`minimum`, `maximum`, and `goal_side`. `TargetKind`, `TargetEntity`, and
`StateMetric` define the categorical IDs. Event masks use `Event`; arena masks
use `ArenaSurface`; touch-surface masks use `1 << int(ContactSurface)`.
`has_state_metric`, `has_minimum`, and `has_maximum` identify optional fields;
absent numeric fields are zero. Goal side is +1 for positive Y and -1 for negative Y.

`failure_exercise_wide` marks failures originating from an exercise rather than
its stage; exercise-wide failures appear alongside each stage's failures.
`deadline_ticks` and `has_deadline` describe each stage. `stage_index`,
`ticks_remaining`, `has_remaining`, `exercise_index`, and `active` describe
current per-simulation progress. The returned tensors are GPU-resident snapshots;
request a new target after stepping or resetting to observe updated progress.

`StateCondition` can be used as a stage target or a stage-wide or exercise-wide
failure condition:

```python
from carl.training import StateCondition

low_ball = StateCondition("ball_height", maximum=300)
fast_car = StateCondition("car_speed", minimum=1500, maximum=2300)
```

Available metrics are `ball_x`, `ball_y`, `ball_height`, `ball_speed`, `car_x`,
`car_y`, `car_height`, `car_speed`, and `car_boost`. Bounds are inclusive, in
world-space physics units (boost is 0–100). Conditions inspect the transition-end
state rather than every tick in the frameskip window, so brief threshold crossings
can be missed. Goal detection itself resets the ball before the transition state
is packed; use the goal event, not ball position, to detect a goal.
