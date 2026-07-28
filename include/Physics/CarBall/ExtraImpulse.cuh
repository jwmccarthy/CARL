#pragma once

#include "RLConstants.cuh"
#include "State/GameState.cuh"

CARL_D CARL_FI float ballCarExtraImpulseCurve(float speed)
{
    if (speed <= 500.f) return 0.65f;
    if (speed <= 2300.f)
        return 0.65f + (0.55f - 0.65f) * ((speed - 500.f) / 1800.f);
    if (speed <= 4600.f)
        return 0.55f + (0.30f - 0.55f) * ((speed - 2300.f) / 2300.f);
    return 0.30f;
}

CARL_D CARL_FI Vec3 ballCarExtraVelocity(
    const Vec3& ballPos,
    const Vec3& carPos,
    const Vec3& carVel,
    const Quat& carRot,
    const Vec3& preSolveBallVel)
{
    const Vec3 relVel = preSolveBallVel - carVel;
    const float rawSpeed = sqrtf(relVel.lenSq());
    const float speed = fminf(
        rawSpeed, BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL_UU);
    if (speed <= 0.f) return Vec3::zero();

    const Vec3 relPos = ballPos - carPos;
    Vec3 hitDir = {
        relPos.x,
        relPos.y,
        relPos.z * BALL_CAR_EXTRA_IMPULSE_Z_SCALE
    };
    const float dirLenSq = hitDir.lenSq();
    if (dirLenSq <= 1e-12f) return Vec3::zero();

    hitDir = hitDir * rsqrtf(dirLenSq);
    const Vec3 forward = carRot.toWorld(WORLD_X);
    const float forwardScale =
        hitDir.dot(forward) * (1.f - BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE);
    hitDir = hitDir - forward * forwardScale;

    const float adjustedLenSq = hitDir.lenSq();
    if (adjustedLenSq > 1e-12f)
        hitDir = hitDir * rsqrtf(adjustedLenSq);

    return hitDir * (speed * ballCarExtraImpulseCurve(speed));
}

CARL_D CARL_FI void applyBallCarExtraImpulse(
    GameState* state,
    int ballIdx,
    int carIdx,
    const Vec3& ballPos,
    const Vec3& carPos,
    const Vec3& carVel,
    const Quat& carRot,
    const Vec3& preSolveBallVel)
{
    const int tick = state->tickCount;
    state->cars.ballContactTick[carIdx] = tick;
    state->lastBallTouchTicks[ballIdx] = tick;

    const int lastTick = __ldg(&state->cars.ballHitTick[carIdx]);
    if (tick <= lastTick + 1 && lastTick <= tick) return;

    state->cars.ballHitTick[carIdx] = tick;
    const Vec3 addedVel = ballCarExtraVelocity(
        ballPos, carPos, carVel, carRot, preSolveBallVel);
    state->ball.imp[ballIdx] =
        Vec3::ldg(state->ball.imp[ballIdx]) + addedVel;
}
