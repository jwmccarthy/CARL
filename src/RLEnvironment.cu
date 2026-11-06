#include "RLEnvironment.cuh"

#include <cstdlib>

#include <cuda_runtime_api.h>

#include "Kernels.cuh"
#include "Cuda/Profiler.cuh"
#include "Cuda/DeviceArray.cuh"


RLEnvironment::RLEnvironment(
    const int nSim, const int nBlue,
    const int nOrange, const int seed)
    : h_state(nSim, nBlue, nOrange, seed)
    , h_space(nSim, h_state.nCars, h_state.nTotalCars)
    , carTriCandPrefix(
        h_space.bp.numTris,
        h_space.bp.triPrefix,
        h_state.nTotalCars + 1)
{
    const int blockDim = 256;

    cudaMallocCopy(d_state, h_state);
    cudaMallocCopy(d_space, h_space);
    cudaMallocCopy(d_arena, h_arena);

    CUDA_CHECK(cudaStreamCreate(&stream));

    threadPerSimKernelConfig
        .setBlockDim(32)
        .setGridFromThreads(nSim)
        .setStream(stream);

    threadPerCarKernelConfig
        .setBlockDim(blockDim)
        .setGridFromThreads(h_state.nTotalCars)
        .setStream(stream);

    carTriPairKernelConfig
        .setBlockDim(128)
        .setGridFromSMs(8)
        .setStream(stream);

    solveKernelConfig
        .setBlockDim(blockDim)
        .setGridFromSMs(3)
        .setStream(stream);

    boostPadKernelConfig
        .setBlockDim(128)
        .setGridFromThreads(nSim * NUM_BOOST_PADS)
        .setStream(stream);
}

RLEnvironment::~RLEnvironment()
{
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_space));
    CUDA_CHECK(cudaFree(d_arena));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

void RLEnvironment::beginStep()
{
    threadPerCarKernelConfig.launch(
        beginStepKernel, d_state, d_space);
}

void RLEnvironment::stepBroadSusp()
{
    threadPerCarKernelConfig.launch(
        carArenaBroadPhaseKernel, d_state, d_space, d_arena);

    carTriCandPrefix.scanOnStream(stream);

    threadPerCarKernelConfig.launch(
        carSuspensionRaycastKernel, d_state, d_space, d_arena);
}

void RLEnvironment::stepControls(const DiscreteControls* actions, EventBuffer events)
{
    threadPerCarKernelConfig.launch(
        carControlsKernel, d_state, d_space, actions, events);
}

void RLEnvironment::stepControls(const ContinuousControls* actions, EventBuffer events)
{
    threadPerCarKernelConfig.launch(
        carContinuousControlsKernel, d_state, d_space, actions, events);
}

void RLEnvironment::stepNarrow()
{
    carTriPairKernelConfig.launch(
        carArenaSATKernel, d_state, d_space, d_arena);

    carTriPairKernelConfig.launch(
        carArenaClipKernel, d_arena, d_space);
}

void RLEnvironment::stepCarManifoldSolve(EventBuffer events)
{
    solveKernelConfig.launch(
        carArenaSolveKernel, d_state, d_space, d_arena, events);
}

void RLEnvironment::stepCarCarSolve(EventBuffer events)
{
    threadPerSimKernelConfig.launch(
        carCarBallSolveKernel, d_state, d_space, d_arena, events);
}

void RLEnvironment::integrateCars()
{
    threadPerCarKernelConfig.launch(
        integrateCarsKernel, d_state);
}

void RLEnvironment::applyImpulseCache(EventBuffer events)
{
    threadPerCarKernelConfig.launch(
        applyImpulseCacheKernel, d_state, events);
}

void RLEnvironment::stepBoostPad()
{
    boostPadKernelConfig.launch(
        boostPadKernel, d_state);
}

void RLEnvironment::step(const DiscreteControls* actions, EventBuffer events)
{
    PROFILE("init step",      beginStep());
    PROFILE("broad+susp",     stepBroadSusp());
    PROFILE("controls",       stepControls(actions, events));
    PROFILE("narrow",         stepNarrow());
    PROFILE("manifold+solve", stepCarManifoldSolve(events));
    PROFILE("car-car+ball",   stepCarCarSolve(events));
    PROFILE("integrate-cars", integrateCars());
    PROFILE("impulse-cache",  applyImpulseCache(events));
    PROFILE("boost-pads",     stepBoostPad());
}

void RLEnvironment::step(
    const ContinuousControls* actions, EventBuffer events)
{
    PROFILE("init step",      beginStep());
    PROFILE("broad+susp",     stepBroadSusp());
    PROFILE("controls",       stepControls(actions, events));
    PROFILE("narrow",         stepNarrow());
    PROFILE("manifold+solve", stepCarManifoldSolve(events));
    PROFILE("car-car+ball",   stepCarCarSolve(events));
    PROFILE("integrate-cars", integrateCars());
    PROFILE("impulse-cache",  applyImpulseCache(events));
    PROFILE("boost-pads",     stepBoostPad());
}

void RLEnvironment::reset()
{
    h_state.reset();
    h_space.reset();

    PROFILE("reset", threadPerSimKernelConfig.launch(resetKernel, d_state));
}

void RLEnvironment::resetDones(
    int maxTicks, int overtimeTimeoutTicks, int noTouchTimeoutTicks,
    EventBuffer events)
{
    PROFILE("reset dones", threadPerSimKernelConfig.launch(
        resetDonesKernel, d_state, d_space,
        maxTicks, overtimeTimeoutTicks, noTouchTimeoutTicks, events));
}

void RLEnvironment::resetSelected(const bool* mask, EventBuffer events)
{
    PROFILE("reset selected", threadPerSimKernelConfig.launch(
        resetSelectedKernel, d_state, d_space, mask, events));
}
