#include <cmath>
#include <cstdio>
#include "RLEnvironment.cuh"

__global__ void readControls(GameState* state, CarControls* controls, Vec3* positions)
{
    const int car = threadIdx.x;
    controls[car] = state->cars.controls[car];
    positions[car] = state->cars.pos[car];
}

int main()
{
    int input[] = {3, 1, 4, 2};
    int expected[] = {0, 3, 4, 8};
    int *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, sizeof(input)));
    CUDA_CHECK(cudaMalloc(&deviceOutput, sizeof(input)));
    CUDA_CHECK(cudaMemcpy(deviceInput, input, sizeof(input), cudaMemcpyHostToDevice));
    PrefixSum scan(deviceInput, deviceOutput, 4);
    scan.scan();
    int output[4];
    CUDA_CHECK(cudaMemcpy(output, deviceOutput, sizeof(output), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; ++i)
        if (output[i] != expected[i]) return 1;

    RLEnvironment env(1, 1, 1, 42);
    env.reset();
    CarControls inputControls[] = {
        {1.f, .5f, -.75f, .25f, -.5f, 0, 1, 0},
        {-1.f, -.25f, .5f, -.5f, .75f, 1, 0, 1}
    };
    CarControls *actions, *actual;
    Vec3* positions;
    CUDA_CHECK(cudaMalloc(&actions, sizeof(inputControls)));
    CUDA_CHECK(cudaMalloc(&actual, sizeof(inputControls)));
    CUDA_CHECK(cudaMalloc(&positions, 2 * sizeof(Vec3)));
    CUDA_CHECK(cudaMemcpyAsync(actions, inputControls, sizeof(inputControls),
                               cudaMemcpyHostToDevice, env.getStream()));
    for (int tick = 0; tick < 8; ++tick) env.step(actions);
    readControls<<<1, 2, 0, env.getStream()>>>(env.getDeviceState(), actual, positions);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(env.getStream()));
    CarControls controls[2];
    Vec3 pos[2];
    CUDA_CHECK(cudaMemcpy(controls, actual, sizeof(controls), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pos, positions, sizeof(pos), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2; ++i)
    {
        const auto& a = inputControls[i];
        const auto& b = controls[i];
        if (a.throttle != b.throttle || a.steer != b.steer || a.yaw != b.yaw ||
            a.pitch != b.pitch || a.roll != b.roll || a.jump != b.jump ||
            a.boost != b.boost || a.slide != b.slide) return 1;
        if (!std::isfinite(pos[i].x) || !std::isfinite(pos[i].y) ||
            !std::isfinite(pos[i].z)) return 1;
    }
    CUDA_CHECK(cudaFree(actions));
    CUDA_CHECK(cudaFree(actual));
    CUDA_CHECK(cudaFree(positions));
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    std::puts("External CUDA scan and independent car controls passed");
}
