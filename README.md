# CARL

CARL is a GPU-based Rocket League simulator for reinforcement learning, based on
[RocketSim](https://github.com/ZealanL/RocketSim).

## Install

You need Linux, Python 3.10+, an NVIDIA GPU, the CUDA 12.8 toolkit, and a
compatible C++ compiler such as GCC 12. From the repository root:

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

## Training policy features

The training runtime exposes a deterministic, parameter-free feature view:

```python
from carl.training import policy_features

features = policy_features(runtime, K=3)
# features.window:   float32 [n_sim, K, 50], current stage first
# features.failures: float32 [n_sim, F, 50], current stage's failures
```

`POLICY_FEATURE_SLICES` names each vector component. Kinds and entities are
one-hot; arena surfaces and car parts are multi-hot. Physical bounds use CARL's
native observation, speed, and boost scales. Deadline/minimum ticks use
`log1p(ticks) / log1p(REGULATION_TICKS)` plus a presence bit. Progress ratios
appear only in window slot 0; unlimited quantities are zero. Goal sides are
relative to `runtime.team_sign` (blue `+1`, the default; orange `-1`).

The last component is a valid bit, also available as `features.window_valid`
and `features.failure_valid`. Padding is entirely zero, exercise-wide failures
appear once, and exercise identity is never encoded.

Compiled conditions use `EVENT`, `GOAL`, `CONTACT`, `LEAVE`, `TOUCH`, and `STATE`
with separate entities, masks, and state/gain bounds. Absent bounds and limits
are zero with `has_* = False`. Wildcard touches and car departures still match
their native flags without touch capture. The unsupported `BallTouch.min_height`
argument has been removed; height constraints use `StateCondition`.

## Mirrored aerial-pack PPO

Training definitions support left/right augmentation directly:

```python
mirrored_exercise = exercise.mirrored()
paired_pack = pack.with_mirrors()
provider = TrainingPackResetProvider(paired_pack, n_sim=1024, balanced=True)
```

`with_mirrors()` makes independent original/mirrored pairs. Reflection across
`x=0` transforms positions, velocities, axial spin, orientations, wall placement,
X-condition bounds, and LEFT/RIGHT touch parts. The target goal does not change.
Balanced resets cycle each simulation through all variants, including partial
resets; the default reset selection remains random.

The example separates the six aerial skill definitions in
`examples/training_packs/pack.py` from the trainer in
`examples/training_packs/ppo.py`. It covers aerial touches, fast aerials, aerial
shots, wall launches, ceiling drops, and air dribbles, on both sides:

```bash
uv run --extra training python -m examples.training_packs.ppo \
  --n-sim 16384 --rollout-steps 32 --minibatch-size 16384
```

The actor and critic share an attention-encoded skill embedding built from the
stage window and current failures, with padding masks and ordered stage tokens.
Rollouts retain raw features so PPO trains the encoder as well as the policy.
Stage completions earn `base * multiplier**stage_index`: by default **1, 2, 4,
8, 16, ...**, once per completed stage. Repeats, failed stages, and timeouts earn no
completion bonus. Skill deadlines are terminal; environment time limits
bootstrap the pre-reset observation and skill, never the next episode's skill.

Each PPO update in this command collects 524,288 transitions, with four epochs
and 128 optimizer steps before KL early stopping. A 65,536 minibatch reduces
that to 32 optimizer steps. bf16, compilation, and fused Adam are enabled by
default; use `--no-bf16 --no-compile` for eager FP32. The first update includes
compilation overhead. Resets sample all selected exercise distributions together,
not separately for every exercise. Checkpoints, configuration, JSONL metrics,
and TensorBoard events go under `runs/training-pack-ppo/`. Metrics
include each variant's success rate and the worse side's rate for each skill;
mastery of one side cannot mask failure on the other. Resume weights and
optimizer state with `--resume /path/to/latest.pt`; physical episodes restart.
Use `--total-timesteps` for the training budget and `--help` for other options.

New runs use the v3 curriculum, with ball starts another 250 units above v2,
timed takeoffs, explicit contact heights and car parts, bounded dribble touches,
and shot follow-through. Initial contacts accept **nose or roof**; later dribble
pushes require the nose. `BallTouch((ContactSurface.FRONT, ContactSurface.ROOF))`
represents either face. See the [curriculum specification](examples/training_packs/README.md).
Older checkpoints retain their original pack: use `--pack-version 1` or
`--pack-version 2` as appropriate when resuming, or start a new run to train v3.

Rich progress reports the current update rate. TensorBoard records each variant,
the weaker side of each skill, PPO epochs/optimizer steps, and phase timings,
including checkpoint saves. Serve events with:

```bash
uv run --extra training tensorboard --logdir runs/training-pack-ppo --port 6006
```

If TensorBoard already serves Goddard's runs on port 6006, reuse that server
instead; this environment exposes CARL runs under `carl-training-pack-ppo/`.
To watch any supported checkpoint version from the sibling Goddard checkout:

```bash
PYTHONPATH=../CARL:../CARL/python uv run --no-sync python -m vis /path/to/latest.pt
```

The viewer loads the complete attention policy and raw skill features, not a
legacy MLP approximation. Both `PYTHONPATH` entries are needed unless the current
API and example adapter are already installed in that environment.

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
