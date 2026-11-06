#pragma once

#include <cstdio>
#include <cstdlib>

#ifdef CARL_USE_HIP
#include <hip/hip_runtime.h>
#define cudaDeviceProp hipDeviceProp_t
#define cudaError_t hipError_t
#define cudaEvent_t hipEvent_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaFree hipFree
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#else
#include <cuda_runtime.h>
#endif

#define CARL_H  __host__
#define CARL_D  __device__
#define CARL_C  __constant__
#define CARL_HD CARL_H CARL_D
#define CARL_FI __forceinline__

#ifdef CARL_USE_HIP
#define CARL_CONSTEXPR constexpr
#else
#define CARL_CONSTEXPR constexpr CARL_HD
#endif

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
