#pragma once

#include "Cuda/PrefixSum.cuh"
#include "Cuda/KernelConfig.cuh"

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
    void stepControls(const DiscreteControls* actions);
    void stepNarrow();
    void stepCarManifoldSolve();
    void stepCarCarSolve();
    void integrateCars();
    void applyImpulseCache();
    void stepBoostPad();

public:
    RLEnvironment(
        const int nSim, const int nBlue,
        const int nOrange, const int seed);
    ~RLEnvironment();

    void step(const DiscreteControls* actions);
    void reset();

    GameState* getDeviceState() { return d_state; }
    cudaStream_t getStream() { return stream; }
    int getNSim() const { return h_state.nSim; }
    int getNCars() const { return h_state.nCars; }
};
