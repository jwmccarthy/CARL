#include <cstdio>

#include "Physics/Controls/Drive.cuh"

__global__ void recover(GameState* state, Workspace* space, Quat rotation, bool jump)
{
    ControlCtx ctx{};
    ctx.state = state;
    ctx.space = space;
    ctx.rot = rotation;
    ctx.fwd = rotation.toWorld(WORLD_X);
    ctx.up = rotation.toWorld(WORLD_Z);
    ctx.jumpPressed = jump;

    updateAutoFlip(ctx);
}

int main()
{
    GameState state(1, 1, 1, 42);
    Workspace space(1, 2, 2);
    GameState* deviceState;
    Workspace* deviceSpace;

    cudaMallocCopy(deviceState, state);
    cudaMallocCopy(deviceSpace, space);

    int failures = 0;
    int cases = 0;

    auto checkRecovery = [&](float yaw, float roll, bool expected, bool jump = true,
                             int contacts = 1, Vec3 normal = WORLD_Z)
    {
        state.cars.internal.fill();
        state.cars.vel.fill();
        state.cars.ang.fill();
        CUDA_CHECK(cudaMemcpy(space.ctMan.count, &contacts, sizeof(contacts), CARL_HToD));
        CUDA_CHECK(cudaMemcpy(space.ctMan.worldNormal, &normal, sizeof(normal), CARL_HToD));

        const Quat tilt = { sinf(roll / 2), 0.f, 0.f, cosf(roll / 2) };
        const Quat rotation = Quat::angle(yaw).comp(tilt);

        recover<<<1, 1>>>(deviceState, deviceSpace, rotation, jump);
        CUDA_CHECK(cudaGetLastError());

        CarInternalState result;
        Vec3 angularVelocity;
        CUDA_CHECK(cudaMemcpy(&result, state.cars.internal, sizeof(result), CARL_DToH));
        CUDA_CHECK(cudaMemcpy(&angularVelocity, state.cars.ang, sizeof(angularVelocity), CARL_DToH));

        const float torque = angularVelocity.dot(rotation.toWorld(WORLD_X));
        const bool passed = expected
            ? result.isAutoFlipping && torque * roll < 0.f && result.autoFlipTimer > 0.f
            : !result.isAutoFlipping && angularVelocity.lenSq() == 0.f;

        cases++;

        if (!passed)
        {
            std::fprintf(stderr,
                "yaw=%.3f roll=%.3f jump=%d contacts=%d normal.z=%.1f: "
                "expected recovery=%d, got recovery=%d torque=%.3f\n",
                yaw, roll, jump, contacts, normal.z, expected, result.isAutoFlipping, torque);
            failures++;
        }
    };

    for (float yaw : { 0.f, .7f, PI / 2, PI, -PI / 2, -.7f })
    {
        checkRecovery(yaw, -3.f, true);
        checkRecovery(yaw, 3.f, true);
        checkRecovery(yaw, -2.7f, false);
        checkRecovery(yaw, 2.7f, false);
    }

    checkRecovery(0.f, 3.f, false, false);
    checkRecovery(0.f, 3.f, false, true, 0);
    checkRecovery(0.f, 3.f, false, true, 1, WORLD_X);

    CUDA_CHECK(cudaFree(deviceState));
    CUDA_CHECK(cudaFree(deviceSpace));

    std::printf("%d/%d recovery cases passed\n", cases - failures, cases);
    return failures ? 1 : 0;
}
