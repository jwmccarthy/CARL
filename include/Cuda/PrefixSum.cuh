#pragma once

#include <cooperative_groups.h>
#include <cub/device/device_scan.cuh>

#include "./Common.cuh"

using scan = cub::DeviceScan;

struct PrefixSum
{
    size_t bytes  = 0;
    int    count  = 0;
    void*  buffer = nullptr;
    int*   input  = nullptr;
    int*   output = nullptr;

    PrefixSum() = default;

    PrefixSum(int* input, int* output, int count)
        : input(input), output(output), count(count)
    {
        CUDA_CHECK(scan::ExclusiveSum(nullptr, bytes, input, output, count));
        CUDA_CHECK(cudaMalloc(&buffer, bytes));
    }

    ~PrefixSum()
    {
        CUDA_CHECK(cudaFree(buffer));
    }

    PrefixSum(const PrefixSum&) = delete;
    PrefixSum& operator=(const PrefixSum&) = delete;
    PrefixSum(PrefixSum&&) = delete;
    PrefixSum& operator=(PrefixSum&&) = delete;

    void scan()
    {
        CUDA_CHECK(scan::ExclusiveSum(buffer, bytes, input, output, count));
    }

    void scanOnStream(cudaStream_t stream)
    {
        CUDA_CHECK(scan::ExclusiveSum(buffer, bytes, input, output, count, stream));
    }
};

CARL_D CARL_FI int2 binarySearchWithValue(const int* a, int n, int t)
{
    int lo = 0;
    int hi = n;
    int loVal = __ldg(&a[0]);

    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        const int midVal = __ldg(&a[mid + 1]);

        if (midVal <= t)
        {
            lo = mid + 1;
            loVal = midVal;
        }
        else
        {
            hi = mid;
        }
    }

    return { lo, loVal };
}
