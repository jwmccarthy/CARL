#pragma once

#include "Cuda/PrefixSum.cuh"
#include "Cuda/KernelConfig.cuh"

#include "State/Events.cuh"
#include "State/GameState.cuh"
#include "State/Workspace.cuh"
#include "Arena/ArenaMesh.cuh"

class RLEnvironment
{
private:
    GameState h_state;
    Workspace h_space;
    ArenaMesh h_arena{};

    GameState* d_state = nullptr;
    Workspace* d_space = nullptr;
    ArenaMesh* d_arena = nullptr;

    cudaStream_t stream;

    PrefixSum carTriCandPrefix;

    KernelConfig threadPerSimKernelConfig;
    KernelConfig threadPerCarKernelConfig;
    KernelConfig carTriPairKernelConfig;
    KernelConfig solveKernelConfig;
    KernelConfig boostPadKernelConfig;

    void beginStep();
    void stepBroadSusp();
    void stepControls(const DiscreteControls* actions, EventBuffer events);
    void stepControls(const ContinuousControls* actions, EventBuffer events);
    void stepNarrow();
    void stepCarManifoldSolve(EventBuffer events);
    void stepCarCarSolve(EventBuffer events);
    void integrateCars();
    void applyImpulseCache(EventBuffer events);
    void stepBoostPad();

public:
    RLEnvironment(
        const int nSim, const int nBlue,
        const int nOrange, const int seed);
    ~RLEnvironment();

    void step(const DiscreteControls* actions, EventBuffer events = {});
    void step(const ContinuousControls* actions, EventBuffer events = {});

    void reset();
    void resetSelected(const bool* mask, EventBuffer events = {});
    void resetDones(
        int maxTicks, int overtimeTimeoutTicks, int noTouchTimeoutTicks,
        EventBuffer events = {});

    GameState* getDeviceState() { return d_state; }
    Workspace* getDeviceWorkspace() { return d_space; }
    cudaStream_t getStream() const { return stream; }

    int getNSim()  const { return h_state.nSim; }
    int getNCars() const { return h_state.nCars; }
};
