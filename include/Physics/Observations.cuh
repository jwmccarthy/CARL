#pragma once

#include "../State/GameState.cuh"

// --- Observation packing ---

CARL_D CARL_FI Vec3 observationVector(Vec3 value, bool invert)
{
    return invert ? Vec3{ -value.x, -value.y, value.z } : value;
}

CARL_D CARL_FI void packObservedCar(
    GameState* __restrict__ state,
    int carIdx,
    bool invert,
    float* __restrict__ obs,
    int& o)
{
    const Vec3 pos = observationVector(
        Vec3::ldg(state->cars.pos[carIdx]), invert);

    const Vec3 vel = observationVector(
        Vec3::ldg(state->cars.vel[carIdx]), invert);

    const Vec3 ang = observationVector(
        Vec3::ldg(state->cars.ang[carIdx]), invert);

    const Quat rot     = Quat::ldg(state->cars.rot[carIdx]);
    const Vec3 forward = observationVector(rot.toWorld(WORLD_X), invert);
    const Vec3 up      = observationVector(rot.toWorld(WORLD_Z), invert);

    const CarInternalState internal = state->cars.internal[carIdx];

    obs[o++] = pos.x;
    obs[o++] = pos.y;
    obs[o++] = pos.z;

    obs[o++] = vel.x;
    obs[o++] = vel.y;
    obs[o++] = vel.z;

    obs[o++] = ang.x;
    obs[o++] = ang.y;
    obs[o++] = ang.z;

    obs[o++] = forward.x;
    obs[o++] = forward.y;
    obs[o++] = forward.z;

    obs[o++] = up.x;
    obs[o++] = up.y;
    obs[o++] = up.z;

    obs[o++] = internal.boost;
    obs[o++] = (float)internal.isOnGround;
    obs[o++] = (float)(state->cars.isDemoed[carIdx] != 0);
    obs[o++] = (float)internal.hasFlipped;
    obs[o++] = (float)internal.hasDoubleJumped;
    obs[o++] = (float)internal.isBoosting;
}

// Serialize an observer-specific view: ball, self, teammates, then opponents.
CARL_D CARL_FI void packObservations(
    GameState* __restrict__ state,
    int simIdx,
    int observerIdx,
    int nCars,
    bool invertOrange,
    float* __restrict__ obs)
{
    const int  carBase          = simIdx * nCars;
    const bool observerIsOrange = observerIdx >= state->nBlue;
    const bool invert          = invertOrange && observerIsOrange;

    int o = 0;

    // Ball
    const Vec3 ballPos = observationVector(
        Vec3::ldg(state->ball.pos[simIdx]), invert);

    const Vec3 ballVel = observationVector(
        Vec3::ldg(state->ball.vel[simIdx]), invert);

    const Vec3 ballAng = observationVector(
        Vec3::ldg(state->ball.ang[simIdx]), invert);

    obs[o++] = ballPos.x;
    obs[o++] = ballPos.y;
    obs[o++] = ballPos.z;

    obs[o++] = ballVel.x;
    obs[o++] = ballVel.y;
    obs[o++] = ballVel.z;

    obs[o++] = ballAng.x;
    obs[o++] = ballAng.y;
    obs[o++] = ballAng.z;

    packObservedCar(state, carBase + observerIdx, invert, obs, o);

    const int teamStart = observerIsOrange ? state->nBlue : 0;
    const int teamEnd   = observerIsOrange ? nCars : state->nBlue;

    for (int c = teamStart; c < teamEnd; c++)
    {
        if (c != observerIdx)
        {
            packObservedCar(state, carBase + c, invert, obs, o);
        }
    }

    const int opponentStart = observerIsOrange ? 0 : state->nBlue;
    const int opponentEnd   = observerIsOrange ? state->nBlue : nCars;

    for (int c = opponentStart; c < opponentEnd; c++)
    {
        packObservedCar(state, carBase + c, invert, obs, o);
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
