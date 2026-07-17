#pragma once

#include "Cuda/DLPack.h"
#include "Cuda/KernelConfig.cuh"
#include "State/GameState.cuh"

// --- DLPack capsule helpers ---

DLManagedTensor* makeFloatTensor(
    void* data, 
    int64_t* shape, 
    int ndim, 
    int deviceId = 0);

DLManagedTensor* makeIntTensor(
    void* data,
    int64_t* shape,
    int ndim,
    int deviceId = 0);

// --- EnvIO ---

// Owns device I/O buffers - DLPack accessors return non-owning views
class EnvIO
{
private:
    float* d_obs = nullptr;
    DiscreteControls* d_actions = nullptr;
    float* d_rewards = nullptr;
    float* d_touches = nullptr;
    bool*  d_dones = nullptr;

    int64_t obsShape[2];
    int64_t actShape[3];
    int64_t rewardShape[1];
    int64_t touchShape[2];
    int64_t doneShape[1];

    int nSim;
    int nCars;
    int obsDim;
    int actDim;
    int maxTicks = 30000;

    cudaStream_t stream;
    KernelConfig perSimConfig;

public:
    EnvIO(int nSim, int nCars, cudaStream_t stream);
    ~EnvIO();

    // Pack state into observation buffer
    void packObs(GameState* d_state);

    // Pack rewards and dones
    void packRewardsDones(GameState* d_state);

    // DLPack accessors (caller owns the capsule)
    DLManagedTensor* getObsTensor();
    DLManagedTensor* getActionsTensor();
    DLManagedTensor* getRewardsTensor();
    DLManagedTensor* getTouchesTensor();
    DLManagedTensor* getDonesTensor();

    // Copy external actions into internal buffer
    void setActions(const int32_t* src);
    const DiscreteControls* getActions() const { return d_actions; }

    // Properties
    int getObsDim() const { return obsDim; }
    int getActDim() const { return actDim; }
    void setMaxTicks(int ticks) { maxTicks = ticks; }
};
