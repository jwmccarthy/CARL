#pragma once

#include "Physics/Controls/Context.cuh"
#include "Physics/Controls/Aerial.cuh"
#include "Physics/Controls/Drive.cuh"

CARL_D CARL_FI void processCarControls(
    GameState* state,
    Workspace* space,
    int carIdx,
    EventBuffer events = {})
{
    if (__ldg(&state->cars.isDemoed[carIdx]))
    {
        space->susp.jumpImpulse[carIdx] = Vec3::zero();
        return;
    }

    if (events.enabled())
    {
        int surfaces = 0;
        int arenaWheels = 0;

        for (int wheel = 0; wheel < NUM_WHEELS; wheel++)
        {
            const int index = carIdx * NUM_WHEELS + wheel;
            const int source = state->cars.wheelSource[index];
            const float length = space->susp.rayDist[index];

            if (source != WHEEL_SOURCE_ARENA) continue;
            if (length > suspRayLength(wheel)) continue;

            surfaces |= carArenaSurface(space->susp.rayNormal[index]);
            arenaWheels++;
        }

        const int supportingSurfaces = arenaWheels >= 3 ? surfaces : 0;
        events.carSurface(carIdx, supportingSurfaces);
    }

    ControlCtx ctx = makeControlCtx(state, space, carIdx, events);

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
