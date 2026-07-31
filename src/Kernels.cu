#include "Kernels.cuh"
#include "State/Reset.cuh"
#include "Cuda/PrefixSum.cuh"
#include "Physics/CarArena/BroadPhase.cuh"
#include "Physics/CarArena/SAT.cuh"
#include "Physics/CarArena/Clip.cuh"
#include "Physics/CarArena/Solve.cuh"
#include "Physics/BallArena/Solve.cuh"
#include "Physics/BoostPads.cuh"
#include "Physics/CarCar/Solve.cuh"
#include "Physics/Controls.cuh"
#include "Physics/FinishTick.cuh"
#include "Physics/SuspensionRaycast.cuh"

__global__ void resetKernel(GameState* __restrict__ state)
{
    int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->nSim) return;

    if (simIdx == 0) state->tickCount = 0;

    resetToKickoff(state, simIdx);
}

__global__ void resetDonesKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    int maxTicks,
    int overtimeTimeoutTicks,
    int noTouchTimeoutTicks)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->nSim) return;

    resetAfterDone(
        state, space, simIdx, maxTicks, overtimeTimeoutTicks,
        noTouchTimeoutTicks);
}

__global__ void beginStepKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space)
{
    const int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (carIdx >= state->nTotalCars) return;

    if (carIdx == 0) state->tickCount++;
    if (carIdx % state->nCars == 0)
    {
        state->episodeTicks[carIdx / state->nCars]++;
    }
}

__global__ void carArenaBroadPhaseKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena)
{
    int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (carIdx >= state->nTotalCars) return;

    carArenaBroadPhase(state, space, arena, carIdx);
}

__global__ void carControlsKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    const DiscreteControls* __restrict__ actions)
{
    const int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (carIdx >= state->nTotalCars) return;

    state->cars.controls[carIdx] = actions[carIdx].decode();
    processCarControls(state, space, carIdx);
}

__global__ void carArenaSATKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena)
{
    const int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int numPairs = __ldg(&space->bp.triPrefix[state->nTotalCars]);

    for (int i = startIdx; i < numPairs; i += stride)
    {
        const int2 found =
            binarySearchWithValue(space->bp.triPrefix, state->nTotalCars, i);
        const int carIdx = found.x;
        const int offset = i - found.y;

        writeCarTriPairResult(state, space, arena, carIdx, offset);
    }
}

__global__ void carArenaClipKernel(
    ArenaMesh* __restrict__ arena,
    Workspace* __restrict__ space)
{
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int hitIdx = start; hitIdx < space->ctNrw.maxCarTriPairs; hitIdx += stride)
    {
        carArenaClip(arena, space, hitIdx);
    }
}

__global__ void carSuspensionRaycastKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena)
{
    const int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (carIdx >= state->nTotalCars) return;

    raycastCarSuspension(state, space, arena, carIdx);
}

__global__ void carArenaSolveKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena)
{
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int carIdx = start; carIdx < state->nTotalCars; carIdx += stride)
    {
        solveCarArena(state, space, arena, carIdx);
    }
}

__global__ void carCarBallSolveKernel(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->nSim) return;

    solveCarCarForSim(state, space, simIdx);
    solveBallForSim(state, arena, simIdx);
}

__global__ void integrateCarsKernel(GameState* __restrict__ state)
{
    const int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (carIdx >= state->nTotalCars) return;

    integrateCar(state, carIdx);
}

__global__ void applyImpulseCacheKernel(GameState* __restrict__ state)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < state->nSim) finishBallTick(state, idx);
    if (idx >= state->nTotalCars) return;

    finishCarTick(state, idx);
}

__global__ void boostPadKernel(GameState* __restrict__ state)
{
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int slotCount = state->nSim * NUM_BOOST_PADS;

    for (int slotIdx = start; slotIdx < slotCount; slotIdx += stride)
    {
        processBoostPad(state, slotIdx);
    }
}
