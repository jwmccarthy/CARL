#pragma once

#include "State/GameState.cuh"
#include "State/Workspace.cuh"
#include "RLConstants.cuh"
#include "Physics/SuspensionUtils.cuh"

struct ControlCtx
{
    GameState* state;
    Workspace* space;
    int carIdx;

    Quat rot;
    Vec3 fwd;
    Vec3 up;
    float fwdSpeed;
    float absFwdSpeed;
    int wheelConCount;
    bool onGround;
    bool jumpPressed;

    CARL_D CARL_FI CarInternalState& internal()
    {
        return state->cars.internal[carIdx];
    }

    CARL_D CARL_FI const CarControls& input() const
    {
        return state->cars.controls[carIdx];
    }

    CARL_D CARL_FI CarSuspension& susp()
    {
        return space->susp;
    }
};

CARL_D CARL_FI int countWheelContacts(const ControlCtx& ctx)
{
    const int base = NUM_WHEELS * ctx.carIdx;
    const CarSuspension& susp = ctx.space->susp;
    int conCount = 0;

    #pragma unroll
    for (int wheel = 0; wheel < NUM_WHEELS; wheel++)
    {
        const float dist = __ldg(&susp.rayDist[base + wheel]);
        if (dist <= suspRayLength(wheel)) conCount++;
    }

    return conCount;
}

CARL_D CARL_FI ControlCtx makeControlCtx(
    GameState* state,
    Workspace* space,
    int carIdx)
{
    ControlCtx ctx{};
    ctx.state = state;
    ctx.space = space;
    ctx.carIdx = carIdx;

    ctx.rot = Quat::ldg(state->cars.rot[carIdx]);
    ctx.fwd = ctx.rot.toWorld(WORLD_X);
    ctx.up = ctx.rot.toWorld(WORLD_Z);
    ctx.fwdSpeed = Vec3::ldg(state->cars.vel[carIdx]).dot(ctx.fwd);
    ctx.absFwdSpeed = fabsf(ctx.fwdSpeed);

    // RocketSim treats three suspension contacts as fully grounded
    ctx.wheelConCount = countWheelContacts(ctx);
    ctx.onGround = ctx.wheelConCount >= 3;

    CarInternalState& s = ctx.internal();
    const CarControls& c = ctx.input();
    ctx.jumpPressed = c.jump && !s.lastJump;
    s.lastJump = c.jump;
    s.isOnGround = ctx.onGround;

    return ctx;
}

CARL_D CARL_FI void addCarVel(ControlCtx& ctx, const Vec3& vel)
{
    ctx.state->cars.vel[ctx.carIdx] =
        Vec3::ldg(ctx.state->cars.vel[ctx.carIdx]) + vel;
}

CARL_D CARL_FI void addDeferredVel(ControlCtx& ctx, const Vec3& vel)
{
    // Deferred velocity is applied with gravity during the arena solve
    CarSuspension& susp = ctx.susp();
    susp.jumpImpulse[ctx.carIdx] =
        Vec3::ldg(susp.jumpImpulse[ctx.carIdx]) + vel;
}

CARL_D CARL_FI void stageLagged(float& current, float& previous, float next)
{
    // Suspension consumes the previous tick while controls stage the next
    previous = current;
    current = next;
}

CARL_D CARL_FI bool opposingThrottle(
    float throttle,
    float fwdSpeed,
    float absFwdSpeed)
{
    return absFwdSpeed > CAR_STOPPING_VEL
        && ((throttle > 0.f) != (fwdSpeed > 0.f));
}
