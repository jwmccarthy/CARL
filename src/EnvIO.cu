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
    int nSim,
    int nCars,
    bool invertOrange)
{
    const int agentIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agentIdx >= nSim * nCars)
    {
        return;
    }

    const int simIdx      = agentIdx / nCars;
    const int observerIdx = agentIdx % nCars;
    const int obsDim      = OBS_BALL + nCars * OBS_PER_CAR;

    packObservations(
        state,
        simIdx,
        observerIdx,
        nCars,
        invertOrange,
        obs + agentIdx * obsDim);
}

__global__ void packRewardsDonesKernel(
    GameState* __restrict__ state,
    float* __restrict__ rewards,
    float* __restrict__ touches,
    bool* __restrict__ dones,
    int nSim, int nCars, int maxTicks, int touchWindow)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= nSim) return;

    packRewards(state, simIdx, rewards);
    packDones(state, simIdx, maxTicks, dones);

    const int carBase = simIdx * nCars;

    for (int c = 0; c < nCars; c++)
    {
        touches[carBase + c] = state->cars.ballContactTick[carBase + c]
                             > state->tickCount - touchWindow;
    }
}

__global__ void setBallKernel(
    GameState* __restrict__ state,
    const float* __restrict__ pos,
    const float* __restrict__ vel,
    const float* __restrict__ ang,
    int nSim)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= nSim) return;

    const int offset = simIdx * 3;
    state->ball.pos[simIdx] = { pos[offset], pos[offset + 1], pos[offset + 2] };
    state->ball.vel[simIdx] = { vel[offset], vel[offset + 1], vel[offset + 2] };
    state->ball.ang[simIdx] = { ang[offset], ang[offset + 1], ang[offset + 2] };
    state->ball.imp[simIdx] = Vec3::zero();
}

__global__ void setCarKernel(
    GameState* __restrict__ state,
    const float* __restrict__ pos,
    const float* __restrict__ rot,
    const float* __restrict__ vel,
    const float* __restrict__ ang,
    const void* __restrict__ demoed,
    bool byteDemoed,
    int nTotalCars)
{
    const int carIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (carIdx >= nTotalCars) return;

    const int vecOffset = carIdx * 3;
    const int rotOffset = carIdx * 4;

    const Vec3 carPos = {
        pos[vecOffset], pos[vecOffset + 1], pos[vecOffset + 2]
    };

    const Quat carRot = {
        rot[rotOffset], rot[rotOffset + 1],
        rot[rotOffset + 2], rot[rotOffset + 3]
    };

    const int isDemoed = byteDemoed
        ? static_cast<const uint8_t*>(demoed)[carIdx]
        : static_cast<const int32_t*>(demoed)[carIdx];

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
}

// --- EnvIO ---

EnvIO::EnvIO(
    int nSim,
    int nCars,
    cudaStream_t stream,
    bool invertOrange)
    : nSim(nSim)
    , nCars(nCars)
    , obsDim(OBS_BALL + nCars * OBS_PER_CAR)
    , actDim(nCars * ACT_PER_CAR)
    , invertOrange(invertOrange)
    , stream(stream)
{
    obsShape[0] = nSim;
    obsShape[1] = nCars;
    obsShape[2] = obsDim;
    actShape[0] = nSim;
    actShape[1] = nCars;
    actShape[2] = ACT_PER_CAR;
    rewardShape[0] = nSim;
    touchShape[0] = nSim;
    touchShape[1] = nCars;
    doneShape[0] = nSim;

    CUDA_CHECK(cudaMalloc(&d_obs, nSim * nCars * obsDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_actions, nSim * nCars * sizeof(DiscreteControls)));
    CUDA_CHECK(cudaMalloc(&d_rewards, nSim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_touches, nSim * nCars * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dones, nSim * sizeof(bool)));

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
    CUDA_CHECK(cudaFree(d_actions));
    CUDA_CHECK(cudaFree(d_rewards));
    CUDA_CHECK(cudaFree(d_touches));
    CUDA_CHECK(cudaFree(d_dones));
}

void EnvIO::packObs(GameState* d_state)
{
    packObsKernel<<<perAgentConfig.gridDim,
        perAgentConfig.blockDim, 0, stream>>>(
        d_state, d_obs, nSim, nCars, invertOrange);
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::packRewardsDones(GameState* d_state, int touchWindow)
{
    packRewardsDonesKernel<<<perSimConfig.gridDim,
        perSimConfig.blockDim, 0, stream>>>(
        d_state, d_rewards, d_touches, d_dones,
        nSim, nCars, maxTicks, touchWindow);
    CUDA_CHECK(cudaGetLastError());
}

DLManagedTensor* EnvIO::getObsTensor()
{
    return makeFloatTensor(d_obs, obsShape, 3);
}

DLManagedTensor* EnvIO::getActionsTensor()
{
    return makeIntTensor(d_actions, actShape, 3);
}

DLManagedTensor* EnvIO::getRewardsTensor()
{
    return makeFloatTensor(d_rewards, rewardShape, 1);
}

DLManagedTensor* EnvIO::getTouchesTensor()
{
    return makeFloatTensor(d_touches, touchShape, 2);
}

DLManagedTensor* EnvIO::getDonesTensor()
{
    return makeBoolTensor(d_dones, doneShape, 1);
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
    const float* ang)
{
    setBallKernel<<<perSimConfig.gridDim,
        perSimConfig.blockDim, 0, stream>>>(
        d_state, pos, vel, ang, nSim);
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::setCar(
    GameState* d_state,
    const float* pos,
    const float* rot,
    const float* vel,
    const float* ang,
    const void* demoed,
    bool byteDemoed)
{
    const KernelConfig perCarConfig = KernelConfig{}
        .setBlockDim(256)
        .setGridFromThreads(nSim * nCars)
        .setStream(stream);

    setCarKernel<<<perCarConfig.gridDim,
        perCarConfig.blockDim, 0, stream>>>(
        d_state, pos, rot, vel, ang, demoed, byteDemoed, nSim * nCars);
        
    CUDA_CHECK(cudaGetLastError());
}
