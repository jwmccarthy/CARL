#pragma once

#include "../State/GameState.cuh"
#include "BoostPads.cuh"

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

    const Quat rot = Quat::ldg(state->cars.rot[carIdx]);
    const Vec3 forward = observationVector(rot.toWorld(WORLD_X), invert);
    const Vec3 up = observationVector(rot.toWorld(WORLD_Z), invert);

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

CARL_D CARL_FI void packRelativeCar(
    GameState* __restrict__ state,
    int carIdx,
    Vec3 egoPos,
    Vec3 egoVel,
    bool invert,
    float* __restrict__ obs,
    int& o)
{
    const Vec3 relativePos = observationVector(
        Vec3::ldg(state->cars.pos[carIdx]) - egoPos, invert);
    const Vec3 relativeVel = observationVector(
        Vec3::ldg(state->cars.vel[carIdx]) - egoVel, invert);

    obs[o++] = relativePos.x;
    obs[o++] = relativePos.y;
    obs[o++] = relativePos.z;
    obs[o++] = relativeVel.x;
    obs[o++] = relativeVel.y;
    obs[o++] = relativeVel.z;
}

constexpr CARL_D int INVERTED_BOOST_PAD_INDICES[NUM_BOOST_PADS] = {
     1,  0,  5,  4,  3,  2, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
    22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6
};

CARL_D CARL_FI int observedBoostPadIndex(int padIdx, bool invert)
{
    return invert ? INVERTED_BOOST_PAD_INDICES[padIdx] : padIdx;
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
    const Vec3 rawBallPos = Vec3::ldg(state->ball.pos[simIdx]);
    const Vec3 rawBallVel = Vec3::ldg(state->ball.vel[simIdx]);
    const Vec3 egoPos = Vec3::ldg(state->cars.pos[carBase + observerIdx]);
    const Vec3 egoVel = Vec3::ldg(state->cars.vel[carBase + observerIdx]);

    const Vec3 ballPos = observationVector(rawBallPos, invert);

    const Vec3 ballVel = observationVector(rawBallVel, invert);

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

    for (int p = 0; p < NUM_BOOST_PADS; p++)
    {
        const int padIdx = observedBoostPadIndex(p, invert);
        obs[o++] = state->boostPadCooldowns[
            simIdx * NUM_BOOST_PADS + padIdx] <= 0.f;
    }

    for (int p = 0; p < NUM_BOOST_PADS; p++)
    {
        const int padIdx = observedBoostPadIndex(p, invert);
        obs[o++] = (egoPos - BOOST_PADS[padIdx].pos).len();
    }

    const Vec3 egoToBallPos = observationVector(rawBallPos - egoPos, invert);
    const Vec3 egoToBallVel = observationVector(rawBallVel - egoVel, invert);
    obs[o++] = egoToBallPos.x;
    obs[o++] = egoToBallPos.y;
    obs[o++] = egoToBallPos.z;
    obs[o++] = egoToBallVel.x;
    obs[o++] = egoToBallVel.y;
    obs[o++] = egoToBallVel.z;

    for (int c = teamStart; c < teamEnd; c++)
    {
        if (c != observerIdx)
        {
            packRelativeCar(
                state, carBase + c, egoPos, egoVel, invert, obs, o);
        }
    }

    for (int c = opponentStart; c < opponentEnd; c++)
    {
        packRelativeCar(
            state, carBase + c, egoPos, egoVel, invert, obs, o);
    }

    const float ownGoalY = observerIsOrange
        ? SOCCAR_GOAL_CENTER_Y : -SOCCAR_GOAL_CENTER_Y;
    const Vec3 ownGoal = { 0.f, ownGoalY, SOCCAR_GOAL_CENTER_Z };
    const Vec3 opponentGoal = { 0.f, -ownGoalY, SOCCAR_GOAL_CENTER_Z };
    const Vec3 ballToOwnGoal = observationVector(ownGoal - rawBallPos, invert);
    const Vec3 ballToOpponentGoal = observationVector(
        opponentGoal - rawBallPos, invert);

    obs[o++] = ballToOwnGoal.x;
    obs[o++] = ballToOwnGoal.y;
    obs[o++] = ballToOwnGoal.z;
    obs[o++] = ballToOpponentGoal.x;
    obs[o++] = ballToOpponentGoal.y;
    obs[o++] = ballToOpponentGoal.z;
}

CARL_D CARL_FI void normalizeObservations(float* obs, int nCars)
{
    constexpr float positionScale[3] = { 4108.f, 6000.f, 2076.f };
    constexpr float arenaDiagonal = 14692.54f;

    for (int axis = 0; axis < 3; axis++)
    {
        obs[axis] /= positionScale[axis];
    }

    for (int axis = 3; axis < 6; axis++)
    {
        obs[axis] /= BALL_MAX_SPEED;
    }

    for (int axis = 6; axis < 9; axis++)
    {
        obs[axis] /= BALL_MAX_ANG_SPEED;
    }

    for (int car = 0; car < nCars; car++)
    {
        const int offset = OBS_BALL + car * OBS_PER_CAR;

        for (int axis = 0; axis < 3; axis++)
        {
            obs[offset + axis] /= positionScale[axis];
        }

        for (int axis = 3; axis < 6; axis++)
        {
            obs[offset + axis] /= CAR_MAX_SPEED;
        }

        for (int axis = 6; axis < 9; axis++)
        {
            obs[offset + axis] /= CAR_MAX_ANG_SPEED;
        }

        obs[offset + 15] /= BOOST_MAX;
    }

    const int distanceOffset = OBS_BALL + nCars * OBS_PER_CAR + NUM_BOOST_PADS;

    for (int pad = 0; pad < NUM_BOOST_PADS; pad++)
    {
        obs[distanceOffset + pad] /= arenaDiagonal;
    }

    int relativeOffset = distanceOffset + NUM_BOOST_PADS;
    for (int axis = 0; axis < 3; axis++)
    {
        obs[relativeOffset + axis] /= 2.f * positionScale[axis];
        obs[relativeOffset + 3 + axis] /= BALL_MAX_SPEED + CAR_MAX_SPEED;
    }
    relativeOffset += OBS_RELATIVE_EGO_BALL;

    for (int car = 1; car < nCars; car++)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            obs[relativeOffset + axis] /= 2.f * positionScale[axis];
            obs[relativeOffset + 3 + axis] /= 2.f * CAR_MAX_SPEED;
        }
        relativeOffset += OBS_RELATIVE_PER_OTHER_CAR;
    }

    for (int goal = 0; goal < 2; goal++)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            obs[relativeOffset + goal * 3 + axis]
                /= 2.f * positionScale[axis];
        }
    }
}

// Serialize canonical world state once per simulation for reward computation.
CARL_D CARL_FI void packState(
    GameState* __restrict__ state,
    int simIdx,
    int nCars,
    int touchWindow,
    float* __restrict__ output)
{
    int o = 0;

    const Vec3 ballPos = Vec3::ldg(state->ball.pos[simIdx]);
    const Vec3 ballVel = Vec3::ldg(state->ball.vel[simIdx]);
    const Vec3 ballAng = Vec3::ldg(state->ball.ang[simIdx]);

    output[o++] = ballPos.x;
    output[o++] = ballPos.y;
    output[o++] = ballPos.z;
    output[o++] = ballVel.x;
    output[o++] = ballVel.y;
    output[o++] = ballVel.z;
    output[o++] = ballAng.x;
    output[o++] = ballAng.y;
    output[o++] = ballAng.z;

    const int carBase = simIdx * nCars;
    for (int c = 0; c < nCars; c++)
    {
        const int carIdx = carBase + c;
        packObservedCar(state, carIdx, false, output, o);
        output[o++] = state->cars.ballContactTick[carIdx]
                    > state->tickCount - touchWindow;
    }

    for (int p = 0; p < NUM_BOOST_PADS; p++)
    {
        output[o++] = state->boostPadCooldowns[simIdx * NUM_BOOST_PADS + p] <= 0.f;
    }
}

// --- Reward and done ---

CARL_D CARL_FI void packRewards(
    GameState* __restrict__ state,
    int simIdx, float* __restrict__ rewards)
{
    const GoalState& goal = state->goals[simIdx];

    // Transition reward: +1 for a blue goal, -1 for an orange goal.
    rewards[simIdx] = (float)goal.lastScorer;
}

CARL_D CARL_FI void packDones(
    GameState* __restrict__ state,
    int simIdx,
    int maxTicks,
    int overtimeTimeoutTicks,
    int noTouchTimeoutTicks,
    bool* __restrict__ dones)
{
    const GoalState& goal = state->goals[simIdx];
    const bool regulationExpired = state->episodeTicks[simIdx] >= maxTicks
        && goal.blueScore != goal.orangeScore;
    const bool overtimeExpired = goal.overtime
        && state->episodeTicks[simIdx] >= maxTicks + overtimeTimeoutTicks;
    const bool scored = goal.lastScorer != 0;
    dones[simIdx] = (!goal.overtime && regulationExpired)
                 || overtimeExpired
                 || (noTouchTimeoutTicks > 0
                       && state->tickCount - state->lastBallTouchTicks[simIdx]
                          >= noTouchTimeoutTicks)
                 || scored;
}

CARL_D CARL_FI void packOvertime(
    GameState* __restrict__ state, int simIdx, bool* __restrict__ overtime)
{
    overtime[simIdx] = state->goals[simIdx].overtime;
}
