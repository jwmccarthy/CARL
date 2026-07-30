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

DLManagedTensor* makeBoolTensor(
    void* data,
    int64_t* shape,
    int ndim,
    int deviceId = 0);

// --- EnvIO ---

// Owns device I/O buffers - DLPack accessors return non-owning views
class EnvIO
{
private:
    float* d_obs     = nullptr;
    float* d_transitionObs = nullptr;
    float* d_state   = nullptr;
    float* d_transitionState = nullptr;
    float* d_rewards = nullptr;
    bool*  d_dones   = nullptr;
    int*   d_scoreDifference = nullptr;
    int*   d_episodeTicks = nullptr;
    int*   d_transitionScoreDifference = nullptr;
    int*   d_transitionEpisodeTicks = nullptr;
    bool*  d_overtime = nullptr;
    bool*  d_transitionOvertime = nullptr;

    DiscreteControls* d_actions = nullptr;

    int64_t obsShape[3];
    int64_t stateShape[2];
    int64_t actShape[3];
    int64_t rewardShape[1];
    int64_t doneShape[1];

    int  nSim;
    int  nCars;
    int  obsDim;
    int  stateDim;
    int  actDim;
    int  maxTicks = 5 * 60 * 120;
    int  noTouchTimeoutTicks = 0;
    bool invertOrange;
    bool normalize;

    cudaStream_t stream;
    KernelConfig perSimConfig;
    KernelConfig perAgentConfig;

public:
    EnvIO(
        int nSim,
        int nCars,
        cudaStream_t stream,
        bool invertOrange,
        bool normalize);
    ~EnvIO();

    // Pack state into observation buffer
    void packObs(GameState* d_state);
    void packTransitionObs(GameState* d_state);

    // Pack canonical state before and after same-step autoreset
    void packState(GameState* d_state);
    void packTransitionState(GameState* d_state, int touchWindow);

    // Pack rewards and dones
    void packRewardsDones(GameState* d_state);
    void packRawMatchState(GameState* d_state);

    // DLPack accessors (caller owns the capsule)
    DLManagedTensor* getObsTensor();
    DLManagedTensor* getTransitionObsTensor();
    DLManagedTensor* getStateTensor();
    DLManagedTensor* getTransitionStateTensor();
    DLManagedTensor* getActionsTensor();
    DLManagedTensor* getRewardsTensor();
    DLManagedTensor* getDonesTensor();
    DLManagedTensor* getScoreDifferenceTensor();
    DLManagedTensor* getEpisodeTicksTensor();
    DLManagedTensor* getTransitionScoreDifferenceTensor();
    DLManagedTensor* getTransitionEpisodeTicksTensor();
    DLManagedTensor* getOvertimeTensor();
    DLManagedTensor* getTransitionOvertimeTensor();

    // Copy external actions into internal buffer
    void setActions(const int32_t* src);

    const DiscreteControls* getActions() const { return d_actions; }

    // Set state from contiguous device buffers
    void setBall(
        GameState* d_state,
        const float* pos,
        const float* vel,
        const float* ang,
        const int64_t* simulationIndices = nullptr,
        int nSelected = -1);

    void setCar(
        GameState* d_state,
        const float* pos,
        const float* rot,
        const float* vel,
        const float* ang,
        const void* demoed,
        bool byteDemoed,
        const float* boost = nullptr,
        const int64_t* simulationIndices = nullptr,
        int nSelected = -1);
    void setMatchState(GameState* d_state, const int32_t* blueScore,
        const int32_t* orangeScore, const int32_t* episodeTicks,
        const int64_t* simulationIndices = nullptr, int nSelected = -1);

    // Properties
    int getObsDim() const { return obsDim; }
    int getActDim() const { return actDim; }

    int getMaxTicks() const { return maxTicks; }
    void setMaxTicks(int ticks) { maxTicks = ticks; }
    int getNoTouchTimeoutTicks() const { return noTouchTimeoutTicks; }
    void setNoTouchTimeoutTicks(int ticks) { noTouchTimeoutTicks = ticks; }
};
