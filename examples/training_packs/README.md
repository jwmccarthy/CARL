# Mirrored aerial curriculum

`pack.py` defines six skills with independent left/right variants. New runs use
**v3**, with ball starts another 250 units above v2. v1 and v2 remain available
for checkpoint playback and resume with the matching `--pack-version`.

## Starts

Positions are in CARL/Rocket League units; velocities are units/second. v3 raises
every v2 ball start by another 250 units, without changing car states, launch
velocities, or skill sequences. Upward speeds remain 150–200 above v1.

| Skill | Ball height | Initial upward speed |
| --- | ---: | ---: |
| Aerial touch | 1,000–1,200 | 250–350 |
| Fast aerial | 1,350–1,550 | 250–350 |
| Aerial shot | 1,200 | 200 |
| Wall aerial | 1,250–1,400 | 300 |
| Ceiling aerial | 1,400–1,600 | 100 |
| Air dribble | 1,100 | 200 |

The dribble starts horizontally closer to the car for a soft catch; the shot starts
with a faster approach. The ceiling ball now rises initially instead of falling.

## Required sequences

Initial contact in **every skill accepts FRONT (nose) or ROOF**, including
nose/roof overlap in one decision window. These are alternatives, not two
required consecutive touches. Later dribble pushes specifically require FRONT.
Rear, side, and underside contacts fail the touch stages; their combined mask is
one failure condition. The surface-qualified speed change must come from the
**same physics tick**, not another contact in that decision window.

| Skill | Ordered requirements |
| --- | --- |
| Aerial touch | Leave floor → car height ≥350 by 1.2s → car speed ≥800 by 1.6s → nose/roof contact by 2.4s, speed gain ≥60, car >350 and ball >600. |
| Fast aerial | Jump by 0.65s → neutral double jump after at least 0.08s, within 0.7s of the first jump → car height ≥950 by 1.6s → nose/roof contact by 2.6s, gain ≥100, car >750 and ball >900. Dodges fail. |
| Aerial shot | Airborne nose/roof contact within 1.8s, gain ≥250, car >450 and ball >650 → ball speed ≥1,400 within 0.25s → ball crosses Y=2,000 by 3.5s → opponent goal by 5.5s. |
| Wall aerial | Leave side wall within 0.9s → clear inward past X=−3,500 within 0.85s of departure → car height ≥700 by 1.5s → nose/roof contact by 2.8s, gain ≥150, car >500 and ball >650 → goal by 6.5s. X clearance mirrors correctly on the opposite wall. |
| Ceiling aerial | Leave ceiling within 0.8s without jumping → descend below 1,600 by 1.3s → nose/roof contact by 2.8s, gain ≥100, car >500 and ball >650 → goal by 6.5s. Jump, double jump, and dodge fail before the first touch, preserving the flip for follow-through. |
| Air dribble | Soft nose/roof catch within 1.8s, speed change −300…+200 → FRONT push with gain 0…250 → another FRONT push with gain −50…+250 → ball crosses Y=1,500 by 4.5s → goal by 7s. Pushes are at least 0.2s apart and each has a 1.2s deadline. Dodges fail. |

Unless explicitly described as stage-relative, the times above are measured
from the exercise start. Airborne approach stages reject arena contact and low
car/ball heights. Shot, wall, and ceiling follow-through allow the car to land
after contact, but not a ball bounce. Dribbles must remain airborne throughout.

Only one stage can advance in a decision window. The curriculum does **not**
require `Jump → LeaveSurface` in separate stages, since a real takeoff normally
produces both events together. Persistent height/speed checkpoints are safe to
use after an event. Underside contact alone is not treated as a flip reset.

## Training and viewing

From CARL:

```bash
uv run --extra training python -m examples.training_packs.ppo \
  --n-sim 16384 --rollout-steps 32 --minibatch-size 16384
```

Use `--minibatch-size 65536` for fewer, larger optimizer steps if GPU memory
allows. PPO still uses four epochs. Rich progress, JSONL, and TensorBoard include
rollout/update/checkpoint timings. bf16, fused Adam, compiled policy/features,
and GAE are enabled by default; `--no-bf16 --no-compile` selects eager FP32.
Compilation makes the first update substantially slower. Reset distributions
are sampled in one batched pass, rather than once per exercise per step.

Start a fresh v3 run to use the higher starts. Resume older checkpoints with
`--pack-version 1` or `--pack-version 2` as appropriate, plus
`--resume /path/to/latest.pt`, retaining their `--n-sim` setting. Playback detects
all three versions automatically and restores the complete attention encoder,
shared body, and heads.

From Goddard, using the local CARL checkout and its built extension:

```bash
PYTHONPATH=../CARL:../CARL/python uv run --no-sync python -m vis /path/to/latest.pt
```

Both paths matter: the root supplies the example policy adapter, and `python/`
supplies the current CARL API. Plain `uv run` otherwise restores Goddard's pinned
CARL dependency, which may not have `carl.training`. `P` resamples; `B`/`N` cycle
through original/mirrored exercises. Add `--sample` for stochastic actions.

To preview the v3 definitions without a policy:

```bash
PYTHONPATH=../CARL:../CARL/python uv run --no-sync python -m vis \
  --training-pack ../CARL/examples/training_packs/pack.py
```
