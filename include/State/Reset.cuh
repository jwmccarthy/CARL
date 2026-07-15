#pragma once

#include "../Cuda/Random.cuh"
#include "GameState.cuh"
#include "../RLConstants.cuh"

CARL_D CARL_FI void resetBall(BallState* __restrict__ ball, int simIdx)
{
    ball->pos[simIdx] = { 0.f, 0.f, BALL_REST_Z };
    ball->vel[simIdx] = Vec3::zero();
    ball->ang[simIdx] = Vec3::zero();
}

CARL_D CARL_FI void resetCar(CarState* __restrict__ cars, int carIdx, int locIdx, bool invert)
{
    CarSpawn loc = KICKOFF_LOCATIONS[locIdx % 5];

    // Orange mirrors both horizontal coordinates and rotates by half a turn.
    float x   = invert ? -loc.x   : loc.x;
    float y   = invert ? -loc.y   : loc.y;
    float yaw = invert ? loc.yaw + PI : loc.yaw;

    cars->pos[carIdx] = { x, y, loc.z };
    cars->vel[carIdx] = Vec3::zero();
    cars->ang[carIdx] = Vec3::zero();
    cars->rot[carIdx] = Quat::angle(yaw);
    cars->cen[carIdx] = cars->getHitboxCenter(carIdx);
    cars->internal[carIdx].isOnGround = true;
}

CARL_D CARL_FI void resetToKickoff(GameState* __restrict__ state, int simIdx)
{
    const int nBlue = state->nBlue;
    const int nOrange = state->nOrange;
    const int nCars = state->nCars;

    // Both teams use corresponding entries from one shuffled location order.
    const uint32_t seed = hash32((uint32_t)state->seed) ^ (uint32_t)simIdx;
    const int permIdx = randomIndex(seed, 120);
    const int* carLocs = KICKOFF_PERMUTATIONS[permIdx];

    resetBall(&state->ball, simIdx);

    #pragma unroll 2
    for (int team = 0; team < 2; team++)
    {
        const bool invert = team;
        const int teamSize = team ? nOrange : nBlue;

        for (int i = 0; i < teamSize; i++)
        {
            const int locIdx = carLocs[i];
            const int carIdx = simIdx * nCars + (team * nBlue + i);

            resetCar(&state->cars, carIdx, locIdx, invert);
        }
    }
}
