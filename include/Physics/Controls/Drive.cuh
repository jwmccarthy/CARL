#pragma once

#include "RLConstants.cuh"
#include "Physics/Controls/Context.cuh"

CARL_D CARL_FI float effectiveThrottle(const ControlCtx& ctx)
{
    const CarControls& c = ctx.input();
    const CarInternalState& s = ctx.state->cars.internal[ctx.carIdx];

    return c.boost && s.boost > 0.f ? 1.f : c.throttle;
}

CARL_D CARL_FI float brakeFactor(
    const ControlCtx& ctx,
    float throttle,
    bool opposing)
{
    if (ctx.wheelConCount == 0 || ctx.input().slide) return 0.f;
    if (fabsf(throttle) >= THROTTLE_DEADZONE) return opposing ? 1.f : 0.f;

    return ctx.absFwdSpeed < CAR_STOPPING_VEL
        ? 1.f
        : COASTING_BRAKE_FACTOR;
}

CARL_D CARL_FI float engineDrive(
    const ControlCtx& ctx,
    float throttle,
    bool opposing)
{
    if (ctx.wheelConCount == 0) return 0.f;
    if (fabsf(throttle) < THROTTLE_DEADZONE) return 0.f;
    if (!ctx.input().slide && opposing) return 0.f;

    float scale = driveSpeedTorqueFactor(ctx.absFwdSpeed);
    if (ctx.wheelConCount < 3) scale *= 0.25f;

    return throttle * THROTTLE_GROUND_ACCEL * scale;
}

CARL_D CARL_FI void updateGroundDrive(ControlCtx& ctx)
{
    const CarControls& c = ctx.input();
    CarSuspension& susp = ctx.susp();

    // Boost promotes effective throttle without changing raw-throttle autoroll state
    const float throttle = effectiveThrottle(ctx);
    const bool opposing = opposingThrottle(throttle, ctx.fwdSpeed, ctx.absFwdSpeed);
    const float brake = brakeFactor(ctx, throttle, opposing);
    const float drive = engineDrive(ctx, throttle, opposing);

    // Brake and drive are staged one tick behind to match wheel friction order
    stageLagged(
        susp.brakeFactor[ctx.carIdx],
        susp.brakeFactorPrev[ctx.carIdx],
        brake);

    susp.throttleVal[ctx.carIdx] = throttle;
    susp.rawThrottle[ctx.carIdx] = c.throttle;

    stageLagged(
        susp.engineDrive[ctx.carIdx],
        susp.engineDrivePrev[ctx.carIdx],
        drive);
}

CARL_D CARL_FI bool shouldBoost(
    const CarInternalState& s,
    const CarControls& c)
{
    if (s.boost <= 0.f) return false;
    if (!s.isBoosting) return c.boost;

    return c.boost || s.boostingTime < BOOST_MIN_TIME;
}

CARL_D CARL_FI void updateBoost(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();
    const CarControls& c = ctx.input();

    // Once activated, boost remains on for its minimum duration
    s.isBoosting = shouldBoost(s, c);

    if (s.isBoosting)
    {
        s.boostingTime += PHYS_DT;
        s.boost = fmaxf(s.boost - BOOST_USED_PER_SECOND * PHYS_DT, 0.f);

        const float accel = ctx.onGround ? BOOST_ACCEL_GROUND : BOOST_ACCEL_AIR;
        addDeferredVel(ctx, ctx.fwd * (accel * PHYS_DT));
        s.timeSinceBoosted = 0.f;
    }
    else
    {
        s.boostingTime = 0.f;
        s.timeSinceBoosted += PHYS_DT;
    }

    s.boost = fminf(s.boost, BOOST_MAX);
}

CARL_D CARL_FI void updateSteer(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();
    const CarControls& c = ctx.input();
    CarSuspension& susp = ctx.susp();

    if (c.slide)
    {
        s.handbrakeVal += POWERSLIDE_RISE_RATE * PHYS_DT;
    }
    else
    {
        s.handbrakeVal -= POWERSLIDE_FALL_RATE * PHYS_DT;
    }

    s.handbrakeVal = clampf(s.handbrakeVal, 0.f, 1.f);
    susp.handbrakeVal[ctx.carIdx] = s.handbrakeVal;

    // Powerslide blends toward a separate, more permissive steering curve
    float angle = steerAngleFromSpeed(ctx.absFwdSpeed);
    if (s.handbrakeVal > 0.f)
    {
        const float slideAngle = powerslideSteerAngleFromSpeed(ctx.absFwdSpeed);
        
        angle += (slideAngle - angle) * s.handbrakeVal;
    }

    susp.steerAngle[ctx.carIdx] = angle * c.steer;
}

CARL_D CARL_FI void tryStartAutoFlip(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();

    if (!ctx.jumpPressed) return;
    if (__ldg(&ctx.space->ctMan.count[ctx.carIdx]) == 0) return;

    // Autoflip is allowed only against an upward-facing chassis contact
    const int manIdx = ctx.carIdx * MAX_CAR_MANIFOLD_POINTS;
    const Vec3 normal = Vec3::ldg(ctx.space->ctMan.worldNormal[manIdx]);
    if (normal.z <= CAR_AUTOFLIP_NORMZ_THRESH) return;

    const float roll = atan2f(ctx.up.x, ctx.up.z);
    const float absRoll = fabsf(roll);
    if (absRoll <= CAR_AUTOFLIP_ROLL_THRESH) return;

    s.autoFlipTimer = CAR_AUTOFLIP_TIME * (absRoll / PI);
    s.autoFlipTorqueScale = roll > 0.f ? 1.f : -1.f;
    s.isAutoFlipping = true;

    addCarVel(ctx, ctx.up * -CAR_AUTOFLIP_IMPULSE);
}

CARL_D CARL_FI void updateAutoFlip(ControlCtx& ctx)
{
    CarInternalState& s = ctx.internal();

    tryStartAutoFlip(ctx);
    if (!s.isAutoFlipping) return;

    if (s.autoFlipTimer <= 0.f)
    {
        s.isAutoFlipping = false;
        s.autoFlipTimer = 0.f;
        return;
    }

    const float torque =
        CAR_AUTOFLIP_TORQUE * s.autoFlipTorqueScale * PHYS_DT;

    ctx.state->cars.ang[ctx.carIdx] =
        Vec3::ldg(ctx.state->cars.ang[ctx.carIdx]) + ctx.fwd * torque;
    s.autoFlipTimer -= PHYS_DT;
}
