#include "EnvIO.cuh"
#include "Physics/Observations.cuh"
#include "RLConstants.cuh"

// --- DLPack capsule helpers ---

// Wrap device pointer as non-owning DLPack view
DLManagedTensor* makeFloatTensor(
    void* data, int64_t* shape,
    int ndim, int deviceId)
{
    auto* tensor = new DLManagedTensor{};
    tensor->dl_tensor.data = data;
    tensor->dl_tensor.device = { kDLCUDA, deviceId };
    tensor->dl_tensor.ndim = ndim;
    tensor->dl_tensor.dtype = { kDLFloat, 32, 1 };
    tensor->dl_tensor.shape = shape;
    tensor->dl_tensor.strides = nullptr;
    tensor->dl_tensor.byte_offset = 0;
    tensor->manager_ctx = nullptr;
    tensor->deleter = [](DLManagedTensor* self) { delete self; };
    return tensor;
}

DLManagedTensor* makeIntTensor(
    void* data, int64_t* shape,
    int ndim, int deviceId)
{
    auto* tensor = makeFloatTensor(data, shape, ndim, deviceId);
    tensor->dl_tensor.dtype = { kDLInt, 32, 1 };
    return tensor;
}

DLManagedTensor* makeBoolTensor(
    void* data, int64_t* shape,
    int ndim, int deviceId)
{
    auto* tensor = makeFloatTensor(data, shape, ndim, deviceId);
    tensor->dl_tensor.dtype = { kDLBool, 8, 1 };
    return tensor;
}

// --- Packing kernels ---

__global__ void packObsKernel(
    GameState* __restrict__ state,
    float* __restrict__ obs,
    const bool* __restrict__ mask,
    int nSim,
    int nCars,
    bool invertOrange,
    bool normalize)
{
    const int agentIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agentIdx >= nSim * nCars)
    {
        return;
    }

    const int simIdx = agentIdx / nCars;
    if (mask && !mask[simIdx])
    {
        return;
    }
    const int observerIdx = agentIdx % nCars;
    const int obsDim = OBS_BALL + nCars * OBS_PER_CAR + OBS_BOOST_PADS
        + OBS_RELATIVE_EGO_BALL
        + (nCars - 1) * OBS_RELATIVE_PER_OTHER_CAR
        + OBS_RELATIVE_GOALS;

    packObservations(
        state,
        simIdx,
        observerIdx,
        nCars,
        invertOrange,
        obs + agentIdx * obsDim);
        
    if (normalize)
    {
        normalizeObservations(obs + agentIdx * obsDim, nCars);
    }
}

__global__ void packStateKernel(
    GameState* __restrict__ state,
    float* __restrict__ output,
    int nSim,
    int nCars,
    int touchWindow)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= nSim) return;

    const int stateDim = OBS_BALL + nCars * STATE_PER_CAR + NUM_BOOST_PADS;
    packState(state, simIdx, nCars, touchWindow, output + simIdx * stateDim);
}

__global__ void packRewardsDonesKernel(
    GameState* __restrict__ state,
    float* __restrict__ rewards,
    bool* __restrict__ dones,
    int nSim, int maxTicks, int noTouchTimeoutTicks)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= nSim) return;

    packRewards(state, simIdx, rewards);
    packDones(state, simIdx, maxTicks, noTouchTimeoutTicks, dones);
}

__global__ void packRawMatchStateKernel(GameState* state, int* score, int* ticks)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->nSim) return;
    const GoalState goal = state->goals[simIdx];
    score[simIdx] = goal.blueScore - goal.orangeScore;
    ticks[simIdx] = state->episodeTicks[simIdx];
}

__global__ void packOvertimeKernel(GameState* state, bool* overtime)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= state->nSim) return;
    packOvertime(state, simIdx, overtime);
}

__global__ void setBallKernel(
    GameState* __restrict__ state,
    const float* __restrict__ pos,
    const float* __restrict__ vel,
    const float* __restrict__ ang,
    const int64_t* __restrict__ simulationIndices,
    int nSelected,
    int nSim)
{
    const int sourceSimIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sourceSimIdx >= nSelected) return;

    const int64_t targetSimIdx64 = simulationIndices
        ? simulationIndices[sourceSimIdx] : sourceSimIdx;
    if (targetSimIdx64 < 0 || targetSimIdx64 >= nSim) return;

    const int targetSimIdx = static_cast<int>(targetSimIdx64);
    const int offset = sourceSimIdx * 3;
    state->ball.pos[targetSimIdx] = { pos[offset], pos[offset + 1], pos[offset + 2] };
    state->ball.vel[targetSimIdx] = { vel[offset], vel[offset + 1], vel[offset + 2] };
    state->ball.ang[targetSimIdx] = { ang[offset], ang[offset + 1], ang[offset + 2] };
    state->ball.imp[targetSimIdx] = Vec3::zero();
}

__global__ void setCarKernel(
    GameState* __restrict__ state,
    const float* __restrict__ pos,
    const float* __restrict__ rot,
    const float* __restrict__ vel,
    const float* __restrict__ ang,
    const void* __restrict__ demoed,
    bool byteDemoed,
    const float* __restrict__ boost,
    const int64_t* __restrict__ simulationIndices,
    int nSelected,
    int nCars,
    int nSim)
{
    const int sourceCarIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sourceCarIdx >= nSelected * nCars) return;

    const int sourceSimIdx = sourceCarIdx / nCars;
    const int localCarIdx = sourceCarIdx % nCars;
    const int64_t targetSimIdx64 = simulationIndices
        ? simulationIndices[sourceSimIdx] : sourceSimIdx;
    if (targetSimIdx64 < 0 || targetSimIdx64 >= nSim) return;

    const int carIdx = static_cast<int>(targetSimIdx64) * nCars + localCarIdx;

    const int vecOffset = sourceCarIdx * 3;
    const int rotOffset = sourceCarIdx * 4;

    const Vec3 carPos = {
        pos[vecOffset], pos[vecOffset + 1], pos[vecOffset + 2]
    };

    const Quat carRot = {
        rot[rotOffset], rot[rotOffset + 1],
        rot[rotOffset + 2], rot[rotOffset + 3]
    };

    const int isDemoed = byteDemoed
        ? static_cast<const uint8_t*>(demoed)[sourceCarIdx]
        : static_cast<const int32_t*>(demoed)[sourceCarIdx];

    state->cars.pos[carIdx] = carPos;
    state->cars.rot[carIdx] = carRot;

    state->cars.vel[carIdx] = {
        vel[vecOffset],
        vel[vecOffset + 1],
        vel[vecOffset + 2]
    };

    state->cars.ang[carIdx] = {
        ang[vecOffset],
        ang[vecOffset + 1],
        ang[vecOffset + 2]
    };

    state->cars.cen[carIdx] = carPos + carRot.toWorld(CAR_OFFSETS);
    state->cars.imp[carIdx] = Vec3::zero();
    state->cars.isDemoed[carIdx] = isDemoed != 0;
    state->cars.demoRespawnTimer[carIdx] = isDemoed 
        ? DEMO_RESPAWN_TIME + PHYS_DT : 0.f;

    if (boost)
    {
        state->cars.internal[carIdx].boost = fminf(
            BOOST_MAX, fmaxf(0.f, boost[sourceCarIdx]));
    }
}

__global__ void setMatchStateKernel(GameState* state, const int32_t* blue,
    const int32_t* orange, const int32_t* ticks, const int64_t* indices,
    int nSelected)
{
    const int source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= nSelected) return;
    const int64_t target = indices ? indices[source] : source;
    if (target < 0 || target >= state->nSim) return;
    state->goals[target].blueScore = blue[source];
    state->goals[target].orangeScore = orange[source];
    state->goals[target].overtime = false;
    state->episodeTicks[target] = ticks[source];
}

// --- EnvIO ---

EnvIO::EnvIO(
    int nSim,
    int nCars,
    cudaStream_t stream,
    bool invertOrange,
    bool normalize)
    : nSim(nSim)
    , nCars(nCars)
    , obsDim(OBS_BALL + nCars * OBS_PER_CAR + OBS_BOOST_PADS
        + OBS_RELATIVE_EGO_BALL
        + (nCars - 1) * OBS_RELATIVE_PER_OTHER_CAR
        + OBS_RELATIVE_GOALS)
    , stateDim(OBS_BALL + nCars * STATE_PER_CAR + NUM_BOOST_PADS)
    , actDim(nCars * ACT_PER_CAR)
    , invertOrange(invertOrange)
    , normalize(normalize)
    , stream(stream)
{
    obsShape[0] = nSim;
    obsShape[1] = nCars;
    obsShape[2] = obsDim;
    stateShape[0] = nSim;
    stateShape[1] = stateDim;
    actShape[0] = nSim;
    actShape[1] = nCars;
    actShape[2] = ACT_PER_CAR;
    rewardShape[0] = nSim;
    doneShape[0] = nSim;

    CUDA_CHECK(cudaMalloc(&d_obs, nSim * nCars * obsDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_transitionObs, nSim * nCars * obsDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_state, nSim * stateDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_transitionState, nSim * stateDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_actions, nSim * nCars * sizeof(DiscreteControls)));
    CUDA_CHECK(cudaMalloc(&d_rewards, nSim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dones, nSim * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_scoreDifference, nSim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_episodeTicks, nSim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_transitionScoreDifference, nSim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_transitionEpisodeTicks, nSim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_overtime, nSim * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_transitionOvertime, nSim * sizeof(bool)));

    perSimConfig
        .setBlockDim(32)
        .setGridFromThreads(nSim)
        .setStream(stream);

    perAgentConfig
        .setBlockDim(128)
        .setGridFromThreads(nSim * nCars)
        .setStream(stream);
}

EnvIO::~EnvIO()
{
    CUDA_CHECK(cudaFree(d_obs));
    CUDA_CHECK(cudaFree(d_transitionObs));
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_transitionState));
    CUDA_CHECK(cudaFree(d_actions));
    CUDA_CHECK(cudaFree(d_rewards));
    CUDA_CHECK(cudaFree(d_dones));
    CUDA_CHECK(cudaFree(d_scoreDifference));
    CUDA_CHECK(cudaFree(d_episodeTicks));
    CUDA_CHECK(cudaFree(d_transitionScoreDifference));
    CUDA_CHECK(cudaFree(d_transitionEpisodeTicks));
    CUDA_CHECK(cudaFree(d_overtime));
    CUDA_CHECK(cudaFree(d_transitionOvertime));
}

void EnvIO::packState(GameState* d_gameState)
{
    packStateKernel<<<perSimConfig.gridDim,
        perSimConfig.blockDim, 0, stream>>>(
        d_gameState, d_state, nSim, nCars, 1);
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::packTransitionState(GameState* d_gameState, int touchWindow)
{
    packStateKernel<<<perSimConfig.gridDim,
        perSimConfig.blockDim, 0, stream>>>(
        d_gameState, d_transitionState, nSim, nCars, touchWindow);
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::packObs(GameState* d_state)
{
    packObsKernel<<<perAgentConfig.gridDim,
        perAgentConfig.blockDim, 0, stream>>>(
        d_state, d_obs, nullptr, nSim, nCars, invertOrange, normalize);
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::packTransitionObs(GameState* d_state)
{
    packObsKernel<<<perAgentConfig.gridDim,
        perAgentConfig.blockDim, 0, stream>>>(
        d_state, d_transitionObs, d_dones, nSim, nCars, invertOrange, normalize);
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::packRewardsDones(GameState* d_state)
{
    packRewardsDonesKernel<<<perSimConfig.gridDim,
        perSimConfig.blockDim, 0, stream>>>(
        d_state, d_rewards, d_dones,
        nSim, maxTicks, noTouchTimeoutTicks);
    CUDA_CHECK(cudaGetLastError());
    packRawMatchStateKernel<<<perSimConfig.gridDim, perSimConfig.blockDim, 0, stream>>>(
        d_state, d_transitionScoreDifference, d_transitionEpisodeTicks);
    CUDA_CHECK(cudaGetLastError());
    packOvertimeKernel<<<perSimConfig.gridDim, perSimConfig.blockDim, 0, stream>>>(
        d_state, d_transitionOvertime);
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::packRawMatchState(GameState* d_state)
{
    packRawMatchStateKernel<<<perSimConfig.gridDim, perSimConfig.blockDim, 0, stream>>>(
        d_state, d_scoreDifference, d_episodeTicks);
    CUDA_CHECK(cudaGetLastError());
    packOvertimeKernel<<<perSimConfig.gridDim, perSimConfig.blockDim, 0, stream>>>(
        d_state, d_overtime);
    CUDA_CHECK(cudaGetLastError());
}

DLManagedTensor* EnvIO::getObsTensor()
{
    return makeFloatTensor(d_obs, obsShape, 3);
}

DLManagedTensor* EnvIO::getTransitionObsTensor()
{
    return makeFloatTensor(d_transitionObs, obsShape, 3);
}

DLManagedTensor* EnvIO::getStateTensor()
{
    return makeFloatTensor(d_state, stateShape, 2);
}

DLManagedTensor* EnvIO::getTransitionStateTensor()
{
    return makeFloatTensor(d_transitionState, stateShape, 2);
}

DLManagedTensor* EnvIO::getActionsTensor()
{
    return makeIntTensor(d_actions, actShape, 3);
}

DLManagedTensor* EnvIO::getRewardsTensor()
{
    return makeFloatTensor(d_rewards, rewardShape, 1);
}

DLManagedTensor* EnvIO::getDonesTensor()
{
    return makeBoolTensor(d_dones, doneShape, 1);
}

DLManagedTensor* EnvIO::getScoreDifferenceTensor()
{
    return makeIntTensor(d_scoreDifference, doneShape, 1);
}

DLManagedTensor* EnvIO::getEpisodeTicksTensor()
{
    return makeIntTensor(d_episodeTicks, doneShape, 1);
}

DLManagedTensor* EnvIO::getTransitionScoreDifferenceTensor()
{
    return makeIntTensor(d_transitionScoreDifference, doneShape, 1);
}

DLManagedTensor* EnvIO::getTransitionEpisodeTicksTensor()
{
    return makeIntTensor(d_transitionEpisodeTicks, doneShape, 1);
}

DLManagedTensor* EnvIO::getOvertimeTensor()
{
    return makeBoolTensor(d_overtime, doneShape, 1);
}

DLManagedTensor* EnvIO::getTransitionOvertimeTensor()
{
    return makeBoolTensor(d_transitionOvertime, doneShape, 1);
}

// Copy external device action tensor into internal buffer (D2D, same stream)
void EnvIO::setActions(const int32_t* src)
{
    CUDA_CHECK(cudaMemcpyAsync(d_actions, src,
        nSim * actDim * sizeof(int32_t),
        cudaMemcpyDeviceToDevice, stream));
}

void EnvIO::setBall(
    GameState* d_state,
    const float* pos,
    const float* vel,
    const float* ang,
    const int64_t* simulationIndices,
    int nSelected)
{
    if (nSelected < 0) nSelected = nSim;
    if (nSelected == 0) return;

    const KernelConfig config = KernelConfig{}
        .setBlockDim(32)
        .setGridFromThreads(nSelected)
        .setStream(stream);

    setBallKernel<<<config.gridDim, config.blockDim, 0, stream>>>(
        d_state, pos, vel, ang, simulationIndices, nSelected, nSim);
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::setCar(
    GameState* d_state,
    const float* pos,
    const float* rot,
    const float* vel,
    const float* ang,
    const void* demoed,
    bool byteDemoed,
    const float* boost,
    const int64_t* simulationIndices,
    int nSelected)
{
    if (nSelected < 0) nSelected = nSim;
    if (nSelected == 0) return;

    const KernelConfig perCarConfig = KernelConfig{}
        .setBlockDim(256)
        .setGridFromThreads(nSelected * nCars)
        .setStream(stream);

    setCarKernel<<<perCarConfig.gridDim,
        perCarConfig.blockDim, 0, stream>>>(
        d_state, pos, rot, vel, ang, demoed, byteDemoed,
        boost, simulationIndices, nSelected, nCars, nSim);
        
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::setMatchState(GameState* d_state, const int32_t* blueScore,
    const int32_t* orangeScore, const int32_t* episodeTicks,
    const int64_t* simulationIndices, int nSelected)
{
    if (nSelected < 0) nSelected = nSim;
    if (nSelected == 0) return;
    const KernelConfig config = KernelConfig{}.setBlockDim(32)
        .setGridFromThreads(nSelected).setStream(stream);
    setMatchStateKernel<<<config.gridDim, config.blockDim, 0, stream>>>(
        d_state, blueScore, orangeScore, episodeTicks,
        simulationIndices, nSelected);
    CUDA_CHECK(cudaGetLastError());
}
