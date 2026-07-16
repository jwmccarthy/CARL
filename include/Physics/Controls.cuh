#pragma once

#include "Physics/Controls/Context.cuh"
#include "Physics/Controls/Aerial.cuh"
#include "Physics/Controls/Drive.cuh"

CARL_D CARL_FI void processCarControls(
    GameState* state,
    Workspace* space,
    int carIdx)
{
    if (__ldg(&state->cars.isDemoed[carIdx]))
    {
        space->susp.jumpImpulse[carIdx] = Vec3::zero();
        return;
    }

    ControlCtx ctx = makeControlCtx(state, space, carIdx);

    // Stage grounded drive state before suspension consumes it in the solve
    updateGroundDrive(ctx);
    updateBoost(ctx);
    updateSteer(ctx);
    updateAutoFlip(ctx);

    // Aerial control depends on how many wheels currently reach the surface
    if (ctx.wheelConCount < 3)
    {
        updateAirControl(ctx, ctx.wheelConCount == 0);
    }

    updateJump(ctx);
    updateDodge(ctx);
}
