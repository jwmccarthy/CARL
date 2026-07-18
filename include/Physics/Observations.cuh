#pragma once

#include "../State/GameState.cuh"

// --- Observation packing ---

// Serialize state into obs buffer (layout: OBS_BALL + nCars * OBS_PER_CAR)
CARL_D CARL_FI void packObservations(
    GameState* __restrict__ state,
    int simIdx, int nCars, 
    float* __restrict__ obs)
{
    const int carBase = simIdx * nCars;

    int o = 0;

    // Ball
    const Vec3 ballPos = Vec3::ldg(state->ball.pos[simIdx]);
    const Vec3 ballVel = Vec3::ldg(state->ball.vel[simIdx]);
    const Vec3 ballAng = Vec3::ldg(state->ball.ang[simIdx]);

    obs[o++] = ballPos.x;
    obs[o++] = ballPos.y;
    obs[o++] = ballPos.z;

    obs[o++] = ballVel.x;
    obs[o++] = ballVel.y;
    obs[o++] = ballVel.z;
    
    obs[o++] = ballAng.x;
    obs[o++] = ballAng.y;
    obs[o++] = ballAng.z;

    // Cars
    for (int c = 0; c < nCars; c++)
    {
        const int carIdx = carBase + c;
        const Vec3 pos = Vec3::ldg(state->cars.pos[carIdx]);
        const Vec3 vel = Vec3::ldg(state->cars.vel[carIdx]);
        const Vec3 ang = Vec3::ldg(state->cars.ang[carIdx]);
        const Quat rot = Quat::ldg(state->cars.rot[carIdx]);
        const Vec3 forward = rot.toWorld(WORLD_X);
        const Vec3 up = rot.toWorld(WORLD_Z);
        const CarInternalState internal = state->cars.internal[carIdx];

        obs[o++] = pos.x; obs[o++] = pos.y; obs[o++] = pos.z;
        obs[o++] = vel.x; obs[o++] = vel.y; obs[o++] = vel.z;
        obs[o++] = ang.x; obs[o++] = ang.y; obs[o++] = ang.z;
        obs[o++] = forward.x; obs[o++] = forward.y; obs[o++] = forward.z;
        obs[o++] = up.x; obs[o++] = up.y; obs[o++] = up.z;
        obs[o++] = internal.boost;
        obs[o++] = (float)internal.isOnGround;
        obs[o++] = (float)(state->cars.isDemoed[carIdx] != 0);
        obs[o++] = (float)internal.hasFlipped;
        obs[o++] = (float)internal.hasDoubleJumped;
        obs[o++] = (float)internal.isBoosting;
    }
}

// --- Reward and done ---

CARL_D CARL_FI void packRewards(
    GameState* __restrict__ state,
    int simIdx, float* __restrict__ rewards)
{
    const GoalState& goal = state->goals[simIdx];

    // Reward: +1 for blue scoring, -1 for orange
    rewards[simIdx] = (float)(goal.blueScore - goal.orangeScore);
}

CARL_D CARL_FI void packDones(
    GameState* __restrict__ state,
    int simIdx, int maxTicks, bool* __restrict__ dones)
{
    const GoalState& goal = state->goals[simIdx];
    dones[simIdx] = state->episodeTicks[simIdx] >= maxTicks
                 || goal.blueScore != 0
                 || goal.orangeScore != 0;
}
