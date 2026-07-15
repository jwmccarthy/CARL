#include <cstdlib>
#include <chrono>
#include <cstdio>

#include "Cuda/Common.cuh"
#include "Cuda/Profiler.cuh"
#include "RLEnvironment.cuh"

int main(int argc, char** argv)
{
    const int ticks = argc > 1 ? std::atoi(argv[1]) : 10000;
    const int envs = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int blueCars = argc > 3 ? std::atoi(argv[3]) : 4;
    const int orangeCars = argc > 4 ? std::atoi(argv[4]) : 4;
    const int seed = argc > 5 ? std::atoi(argv[5]) : 123;

    RLEnvironment env{envs, blueCars, orangeCars, seed};
    
    env.reset();
    
    CUDA_CHECK(cudaDeviceSynchronize());

    ScopedCudaProfiler& profiler = ScopedCudaProfiler::get();
    profiler.configure(ticks, envs);

    profiler.start();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ticks; i++)
    {
        env.step();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    profiler.stop();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::printf("%.1f us/tick (%.1f FPS)\n", us / ticks, 1e6 * ticks / us);

    return 0;
}
