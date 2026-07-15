#pragma once

#include "Cuda/Common.cuh"
#include "Cuda/Math.cuh"
#include "State/GameState.cuh"

struct CarPose
{
    Vec3 cen;
    Quat rot;
};

CARL_D CARL_FI CarPose loadCarPose(
    GameState* __restrict__ state,
    const int carIdx)
{
    return {
        Vec3::ldg(state->cars.cen[carIdx]),
        Quat::ldg(state->cars.rot[carIdx])
    };
}

