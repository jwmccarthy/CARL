#pragma once

#include "../../Cuda/Common.cuh"
#include "../../State/GameState.cuh"
#include "../../State/Workspace.cuh"
#include "../../Arena/ArenaMesh.cuh"

#include "../Integration.cuh"

CARL_D CARL_FI void carArenaBroadPhase(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena,
    const int carIdx)
{
    const CarPose pose = loadCarPose(state, carIdx);
    const CarPose pred = predictCarPose(state, carIdx, pose.rot);

    const Vec3 sweptMin = carAABBMin(pose).min(carAABBMin(pred));
    const Int3 cellMin = arena->aabbToCell3D(sweptMin);
    const int cellIdx = arena->cell3DToFlatIdx(cellMin);
    const int pairBase = carIdx * MAX_CAR_TRI_PAIRS;

    space->bp.cellIdx[carIdx] = cellIdx;
    space->bp.numTris[carIdx] = arena->numTrisInCell(cellIdx);
    space->ctHit.carHitStart[carIdx] = pairBase;
    space->ctHit.carHitCount[carIdx] = 0;
}
