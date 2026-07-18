#pragma once

#include <vector>

#include "./Common.cuh"

template<typename T>
T* cudaMallocArray(int n)
{
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    return ptr;
}

template<typename T>
T* cudaMallocCopy(T*& dst, const std::vector<T>& src)
{
    if (src.empty())
    {
        dst = nullptr;
        return dst;
    }

    dst = cudaMallocArray<T>(src.size());
    CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), CARL_HToD));
    return dst;
}

template<typename T>
void cudaMallocCopy(T*& dst, const T& src)
{
    dst = cudaMallocArray<T>(1);
    CUDA_CHECK(cudaMemcpy(dst, &src, sizeof(T), CARL_HToD));
}

template<typename T>
struct DeviceArray
{
    T*  ptr   = nullptr;
    int count = 0;

    DeviceArray() = default;

    DeviceArray(int count) 
        : ptr(cudaMallocArray<T>(count)), count(count)
    {}

    DeviceArray(const std::vector<T>& src)
        : count((int)src.size())
    {
        cudaMallocCopy(ptr, src);
    }

    ~DeviceArray() { cudaFree(ptr); }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray& operator=(DeviceArray&& other) noexcept
    {
        if (this != &other)
        {
            cudaFree(ptr);

            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }

        return *this;
    }

    void fill(int value = 0)
    {
        CUDA_CHECK(cudaMemset(ptr, value, (size_t)count * sizeof(T)));
    }

    CARL_HD CARL_FI operator T*() const { return ptr; }
};
