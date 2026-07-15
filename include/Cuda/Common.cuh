#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CARL_H  __host__
#define CARL_D  __device__
#define CARL_C  __constant__
#define CARL_HD CARL_H CARL_D
#define CARL_FI __forceinline__

#define CARL_HToD cudaMemcpyHostToDevice
#define CARL_DToH cudaMemcpyDeviceToHost

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)

inline void check(
    cudaError_t err, const char* const func,
    const char* const file, const int line)
{
    if (err == cudaSuccess) return;

    std::fprintf(
        stderr, "CUDA_CHECK failed: %s at %s:%d: %s\n",
        func, file, line, cudaGetErrorString(err));

    std::exit(EXIT_FAILURE);
}
