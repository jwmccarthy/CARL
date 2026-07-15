#pragma once

#include "Common.cuh"
#include <cuda_runtime_api.h>
#include <driver_types.h>

struct KernelConfig
{
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedBytes = 0;
    cudaStream_t stream = 0;

    KernelConfig& setBlockDim(int d)
    {
        blockDim = dim3(d);
        return *this;
    }

    KernelConfig& setGridDim(int d)
    {
        gridDim = dim3(d);
        return *this;
    }

    KernelConfig& setGridFromThreads(int threadsNeeded)
    {
        int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        int blocksNeeded = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
        return setGridDim(blocksNeeded);
    }

    KernelConfig& setGridFromSMs(int blocksPerSM)
    {
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        return setGridDim(props.multiProcessorCount * blocksPerSM);
    }

    KernelConfig& setSharedBytes(size_t bytes)
    {
        sharedBytes = bytes;
        return *this;
    }

    KernelConfig& setStream(cudaStream_t s)
    {
        stream = s;
        return *this;
    }

    template<typename Func, typename... Args>
    void launch(Func kernel, Args&&... args)
    {
        kernel<<<gridDim, blockDim, sharedBytes, stream>>>(std::forward<Args>(args)...);
        CUDA_CHECK(cudaGetLastError());
    }
};

inline KernelConfig gridFromThreads(int d, int total)
{
    return KernelConfig{}
        .setBlockDim(d)
        .setGridFromThreads(total);
}

inline KernelConfig gridFromSMs(int d, int perSM)
{
    return KernelConfig{}
        .setBlockDim(d)
        .setGridFromSMs(perSM);
}