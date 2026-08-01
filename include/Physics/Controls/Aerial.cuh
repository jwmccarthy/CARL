#pragma once

#include "Physics/Controls/Context.cuh"
#include "RLConstants.cuh"

enum AirAxis
{
    AIR_PITCH,
    AIR_YAW,
    AIR_ROLL,
    AIR_AXIS_COUNT
};

CARL_D CARL_FI bool jumpGraceActive(const CarInternalState& s)
{
    return s.hasJumped && s.jumpTime < JUMP_MIN_TIME + JUMP_RESET_TIME_PAD;
}

CARL_D CARL_FI bool shouldContinueJump(
    const CarInternalState& s,
    const CarControls& c)
{
    return s.jumpTime < JUMP_MIN_TIME
        || (c.jump && s.jumpTime < JUMP_MAX_TIME);
}

CARL_D CARL_FI void startJump(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();

    s.isJumping = true;
    s.jumpTime = 0.f;

    addDeferredVel(ctx, ctx.up * JUMP_IMMEDIATE_FORCE);
}

CARL_D CARL_FI void applyJumpAccel(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();
    s.hasJumped = true;

    float accel = JUMP_ACCEL;
    if (s.jumpTime < JUMP_MIN_TIME) accel *= 0.62f;

    addDeferredVel(ctx, ctx.up * (accel * PHYS_DT));
}

CARL_D CARL_FI void updateJump(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();
    const CarControls& c = ctx.input();

    // Keep a short grace period so the initial jump can leave the ground
    if (ctx.onGround && !s.isJumping && !jumpGraceActive(s))
    {
        s.hasJumped = false;
        s.jumpTime = 0.f;
    }

    // Holding jump extends acceleration through the configured jump window
    if (s.isJumping)
    {
        s.isJumping = shouldContinueJump(s, c);
    }
    else if (ctx.onGround && ctx.jumpPressed)
    {
        startJump(ctx);
    }

    if (s.isJumping) applyJumpAccel(ctx);

    if (s.isJumping || s.hasJumped) s.jumpTime += PHYS_DT;
}

CARL_D CARL_FI bool isBackwardDodge(
    const ControlCtx& ctx,
    const Vec3& dodge)
{
    if (ctx.absFwdSpeed < 100.f) return dodge.x < 0.f;
    return (dodge.x >= 0.f) != (ctx.fwdSpeed >= 0.f);
}

CARL_D CARL_FI Vec3 scaledDodgeVel(
    const ControlCtx& ctx,
    Vec3 dodge)
{
    if (fabsf(dodge.x) < 0.1f) dodge.x = 0.f;
    if (fabsf(dodge.y) < 0.1f) dodge.y = 0.f;

    const float speedScale = clampf(ctx.absFwdSpeed / CAR_MAX_SPEED, 0.f, 1.f);
    const bool isBackward = isBackwardDodge(ctx, dodge);
    const float maxFwdScale = isBackward
        ? FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE
        : FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE;

    Vec3 vel = dodge * FLIP_INITIAL_VEL_SCALE;
    vel.x *= (maxFwdScale - 1.f) * speedScale + 1.f;
    vel.y *= (FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE - 1.f) * speedScale + 1.f;

    if (isBackward) vel.x *= FLIP_BACKWARD_IMPULSE_SCALE_X;

    return vel;
}

CARL_D CARL_FI Vec3 worldDodgeVel(
    const ControlCtx& ctx,
    const Vec3& localVel)
{
    // Convert local dodge velocity into the car's horizontal world basis
    const Vec3 flatFwd = { ctx.fwd.x, ctx.fwd.y, 0.f };
    const float denom = flatFwd.lenSq();
    const Vec3 flatFwdN = denom > 1e-12f
        ? flatFwd * rsqrtf(denom)
        : WORLD_X;
    const Vec3 right = { -flatFwdN.y, flatFwdN.x, 0.f };

    return flatFwdN * localVel.x + right * localVel.y;
}

CARL_D CARL_FI void startFlip(ControlCtx& ctx, Vec3 dodge)
{
    CarInternalState& s = ctx.internal();

    s.flipTime = 0.f;
    s.hasFlipped = true;
    s.isFlipping = true;

    dodge = dodge * rsqrtf(dodge.lenSq());
    s.flipRelTorque = { -dodge.y, dodge.x, 0.f };

    const Vec3 localVel = scaledDodgeVel(ctx, dodge);
    addCarVel(ctx, worldDodgeVel(ctx, localVel));
}

CARL_D CARL_FI void resetDodgeState(CarInternalState& s)
{
    s.hasDoubleJumped = false;
    s.hasFlipped = false;
    s.isFlipping = false;
    s.airTime = 0.f;
    s.airTimeSinceJump = 0.f;
    s.flipTime = 0.f;
}

CARL_D CARL_FI void updateDodgeWindow(CarInternalState& s)
{
    s.airTime += PHYS_DT;

    if (s.hasJumped && !s.isJumping)
    {
        s.airTimeSinceJump += PHYS_DT;
    }
    else
    {
        s.airTimeSinceJump = 0.f;
    }
}

CARL_D CARL_FI bool hasFlipOrJump(const CarInternalState& s)
{
    return !s.hasFlipped && !s.hasDoubleJumped
        && s.airTimeSinceJump < DOUBLEJUMP_MAX_DELAY;
}

CARL_D CARL_FI void tryStartDodge(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();
    const CarControls& c = ctx.input();

    // A flip reset (landing on ball while airborne, then leaving) leaves
    // hasJumped false with a fresh dodge window. A normal jump sets it true
    // Either way the car can dodge if it still has a flip available
    const bool canDodge = hasFlipOrJump(s);
    if (!ctx.jumpPressed || !canDodge) return;

    // Large directional input starts a flip, otherwise double jump
    const Vec3 dodge = { -c.pitch, c.yaw + c.roll, 0.f };
    if (dodge.lenSq() >= DODGE_DEADZONE * DODGE_DEADZONE)
    {
        startFlip(ctx, dodge);
        return;
    }

    addCarVel(ctx, ctx.up * JUMP_IMMEDIATE_FORCE);
    s.hasDoubleJumped = true;
}

CARL_D CARL_FI void updateFlipMotion(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();

    if (!s.isFlipping)
    {
        if (s.hasFlipped) s.flipTime += PHYS_DT;
        return;
    }

    s.flipTime += PHYS_DT;
    if (s.flipTime > FLIP_TORQUE_TIME) return;

    // Damp vertical velocity only during the active flip landing window
    Vec3 vel = Vec3::ldg(ctx.state->cars.vel[ctx.carIdx]);
    const bool dampZ = s.flipTime >= FLIP_Z_DAMP_START
        && (vel.z < 0.f || s.flipTime < FLIP_Z_DAMP_END);

    if (!dampZ) return;

    vel.z *= 1.f - FLIP_Z_DAMP_120;
    ctx.state->cars.vel[ctx.carIdx] = vel;
}

CARL_D CARL_FI void updateDodge(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();

    if (ctx.onGround)
    {
        resetDodgeState(s);
    }
    else
    {
        updateDodgeWindow(s);
        tryStartDodge(ctx);
    }

    updateFlipMotion(ctx);
}

CARL_D CARL_FI Vec3 airAxis(const ControlCtx& ctx, AirAxis axis)
{
    if (axis == AIR_PITCH) return ctx.rot.toWorld(WORLD_Y).neg();
    if (axis == AIR_YAW) return ctx.up;
    return ctx.fwd.neg();
}

CARL_D CARL_FI bool applyFlipTorque(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();
    const CarControls& c = ctx.input();

    if (s.isFlipping)
    {
        s.isFlipping = s.hasFlipped
            && s.flipTime < FLIP_TORQUE_TIME;
    }

    if (!s.isFlipping) return true;

    // Active flip torque temporarily replaces normal aerial input
    const Vec3 rel = s.flipRelTorque;
    if (rel.lenSq() <= 1e-12f) return true;

    float pitchScale = 1.f;
    bool useAir = false;

    const bool hasPitchInput = rel.y != 0.f && c.pitch != 0.f;
    const bool pitchMatchesFlip = (rel.y > 0.f) == (c.pitch > 0.f);

    if (hasPitchInput && pitchMatchesFlip)
    {
        pitchScale = 1.f - fminf(fabsf(c.pitch), 1.f);
        useAir = true;
    }

    const Vec3 relTorque = {
        rel.x * FLIP_TORQUE_X,
        rel.y * FLIP_TORQUE_Y * pitchScale,
        0.f
    };
    const Vec3 worldTorque = ctx.rot.toWorld(relTorque);

    ctx.state->cars.ang[ctx.carIdx] = Vec3::ldg(ctx.state->cars.ang[ctx.carIdx])
        + worldTorque * PHYS_DT;

    return useAir;
}

CARL_D CARL_FI float airPitchScale(const CarInternalState& s)
{
    const bool pitchLocked = s.isFlipping
        || (s.hasFlipped && s.flipTime < FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME);

    return pitchLocked ? 0.f : 1.f;
}

CARL_D CARL_FI void applyAirTorque(ControlCtx& ctx)
{
    const CarInternalState& s = ctx.internal();
    const CarControls& c = ctx.input();
    const float pitchScale = airPitchScale(s);
    const Vec3 input = { c.pitch, c.yaw, c.roll };
    const Vec3 controlScale = { pitchScale, 1.f, 1.f };
    const Vec3 dampInput = { c.pitch * pitchScale, c.yaw, 0.f };
    const Vec3 ang = Vec3::ldg(ctx.state->cars.ang[ctx.carIdx]);

    Vec3 torque = Vec3::zero();
    Vec3 damping = Vec3::zero();
    const bool hasInput = input.lenSq() > 0.f;

    // Resolve torque and damping over the pitch, yaw, and roll axes uniformly    #pragma unroll
    for (int i = 0; i < AIR_AXIS_COUNT; i++)
    {
        const AirAxis axis = (AirAxis)i;
        const Vec3 dir = airAxis(ctx, axis);
        const float control = input[i] * controlScale[i];
        const float proj = dir.dot(ang);
        const float damp = proj * CAR_AIR_CONTROL_DAMPING[i] * (1.f - fabsf(dampInput[i]));

        if (hasInput)
        {
            torque = torque + dir * (control * CAR_AIR_CONTROL_TORQUE[i]);
        }
        damping = damping + dir * damp;
    }

    ctx.state->cars.ang[ctx.carIdx] =
        ang + (torque - damping) * (CAR_TORQUE_SCALE * PHYS_DT);
}

CARL_D CARL_FI void updateAirControl(ControlCtx& ctx, bool allowAir)
{
    const CarControls& c = ctx.input();
    const bool useAir = applyFlipTorque(ctx);

    if (useAir && allowAir)
    {
        applyAirTorque(ctx);
    }

    if (c.throttle != 0.f)
    {
        addCarVel(ctx, ctx.fwd * (c.throttle * THROTTLE_AIR_ACCEL * PHYS_DT));
    }
}
