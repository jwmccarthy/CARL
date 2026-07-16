#include "EnvIO.cuh"
#include "Physics/Observations.cuh"

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

// --- Packing kernels ---

__global__ void packObsKernel(
    GameState* __restrict__ state,
    float* __restrict__ obs,
    int nSim, int nCars)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= nSim) return;

    const int offset = simIdx * (OBS_BALL + nCars * OBS_PER_CAR);
    packObservations(state, simIdx, nCars, obs + offset);
}

__global__ void packRewardsDonesKernel(
    GameState* __restrict__ state,
    float* __restrict__ rewards,
    bool* __restrict__ dones,
    int nSim, int maxTicks)
{
    const int simIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (simIdx >= nSim) return;

    packRewards(state, simIdx, rewards);
    packDones(state, simIdx, maxTicks, dones);
}

// --- EnvIO ---

EnvIO::EnvIO(int nSim, int nCars, cudaStream_t stream)
    : nSim(nSim)
    , nCars(nCars)
    , obsDim(OBS_BALL + nCars * OBS_PER_CAR)
    , actDim(nCars * ACT_PER_CAR)
    , stream(stream)
{
    obsShape[0] = nSim;
    obsShape[1] = obsDim;
    actShape[0] = nSim;
    actShape[1] = nCars;
    actShape[2] = ACT_PER_CAR;
    rewardShape[0] = nSim;
    doneShape[0] = nSim;

    CUDA_CHECK(cudaMalloc(&d_obs, nSim * obsDim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_actions,
        nSim * nCars * sizeof(DiscreteControls)));
    CUDA_CHECK(cudaMalloc(&d_rewards, nSim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dones, nSim * sizeof(bool)));

    perSimConfig
        .setBlockDim(32)
        .setGridFromThreads(nSim)
        .setStream(stream);
}

EnvIO::~EnvIO()
{
    CUDA_CHECK(cudaFree(d_obs));
    CUDA_CHECK(cudaFree(d_actions));
    CUDA_CHECK(cudaFree(d_rewards));
    CUDA_CHECK(cudaFree(d_dones));
}

void EnvIO::packObs(GameState* d_state)
{
    packObsKernel<<<perSimConfig.gridDim,
        perSimConfig.blockDim, 0, stream>>>(
        d_state, d_obs, nSim, nCars);
    CUDA_CHECK(cudaGetLastError());
}

void EnvIO::packRewardsDones(GameState* d_state)
{
    packRewardsDonesKernel<<<perSimConfig.gridDim,
        perSimConfig.blockDim, 0, stream>>>(
        d_state, d_rewards, d_dones, nSim, maxTicks);
    CUDA_CHECK(cudaGetLastError());
}

DLManagedTensor* EnvIO::getObsTensor()
{
    return makeFloatTensor(d_obs, obsShape, 2);
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
    return makeFloatTensor(d_dones, doneShape, 1);
}

// Copy external device action tensor into internal buffer (D2D, same stream)
void EnvIO::setActions(const int32_t* src)
{
    CUDA_CHECK(cudaMemcpyAsync(d_actions, src,
        nSim * actDim * sizeof(int32_t),
        cudaMemcpyDeviceToDevice, stream));
}
