#include <cstdio>
#include "Physics/Controls/Aerial.cuh"
#include "Physics/Controls/Drive.cuh"
#include "Physics/Goals.cuh"
#include "State/Reset.cuh"

__global__ void checkPhysics(GameState* state, Workspace* space, int* result)
{
    int cases = 0, failures = 0;
    auto check = [&](bool passed) { ++cases; failures += !passed; };
    const float inputs[][3] = {
        {0.f, 1.f, -1.f}, {0.f, -1.f, 1.f},
        {.3f, .3f, 0.f}, {-.3f, -.3f, 0.f},
        {0.f, .25f, -.25f}, {0.f, .24f, -.24f},
        {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}
    };
    const bool flips[] = {true, true, true, true, true, false, false, true};
    for (int i = 0; i < 8; ++i)
    {
        state->cars.internal[0] = {};
        state->cars.vel[0] = Vec3::zero();
        CarControls controls{};
        controls.pitch = inputs[i][0];
        controls.yaw = inputs[i][1];
        controls.roll = inputs[i][2];
        state->cars.controls[0] = controls;
        ControlCtx ctx{};
        ctx.state = state;
        ctx.space = space;
        ctx.rot = Quat::angle(0.f);
        ctx.fwd = WORLD_X;
        ctx.up = WORLD_Z;
        ctx.jumpPressed = true;
        tryStartDodge(ctx);
        const auto internal = state->cars.internal[0];
        const Vec3 velocity = state->cars.vel[0];
        check(bool(internal.hasFlipped) == flips[i]);
        check(bool(internal.hasDoubleJumped) == !flips[i]);
        check(isfinite(velocity.x) && isfinite(velocity.y) && isfinite(velocity.z));
        if (i < 2) check(velocity.lenSq() == 0.f);
    }
    for (int sign = -1; sign <= 1; sign += 2)
    {
        state->goals[0] = {};
        const Vec3 position = {0.f, sign * 5500.f, 200.f};
        const Vec3 velocity = {10.f, sign * 1000.f, -50.f};
        state->ball.pos[0] = position;
        state->ball.vel[0] = velocity;
        detectGoal(state, 0, position);
        const int tick = state->goals[0].tick;
        state->tickCount++;
        detectGoal(state, 0, position);
        const auto goal = state->goals[0];
        check(goal.blueScore + goal.orangeScore == 1);
        check(goal.lastScorer == sign && goal.tick == tick);
        check((state->ball.pos[0] - position).lenSq() == 0.f);
        check((state->ball.vel[0] - velocity).lenSq() == 0.f);
    }
    state->ball.integrationValid[0] = 1;
    resetBall(&state->ball, 0);
    check(state->ball.integrationValid[0] == 0);
    space->susp.steerAngle[0] = .25f;
    state->cars.controls[0] = {};
    ControlCtx ctx{};
    ctx.state = state;
    ctx.space = space;
    updateSteer(ctx);
    check(space->susp.steerAnglePrev[0] == .25f);
    check(space->susp.steerAngle[0] == 0.f);
    state->tickCount = 77;
    for (int sim = 0; sim < 2; ++sim)
    {
        state->goals[sim] = {};
        state->episodeTicks[sim] = 10;
        state->lastBallTouchTicks[sim] = 77;
        state->ballArenaContactTick[sim] = 77;
    }
    for (int car = 0; car < 4; ++car)
    {
        state->cars.arenaContactTick[car] = 77;
        for (int wheel = 0; wheel < NUM_WHEELS; ++wheel)
            state->cars.wheelSource[car * NUM_WHEELS + wheel] = WHEEL_SOURCE_BALL;
    }
    state->goals[0].lastScorer = 1;
    resetAfterDone(state, space, 0, 1200, 1200, 0);
    check(state->goals[0].lastScorer == 0 && state->episodeTicks[0] == 0);
    check(state->episodeTicks[1] == 10);
    check(state->ballArenaContactTick[0] == -1);
    check(state->ballArenaContactTick[1] == 77);
    for (int car = 0; car < 4; ++car)
    {
        check(state->cars.arenaContactTick[car] == (car < 2 ? -1 : 77));
        for (int wheel = 0; wheel < NUM_WHEELS; ++wheel)
            check(state->cars.wheelSource[car * NUM_WHEELS + wheel] ==
                  (car < 2 ? WHEEL_SOURCE_NONE : WHEEL_SOURCE_BALL));
    }
    result[0] = cases;
    result[1] = failures;
}

int main()
{
    GameState state(2, 1, 1, 42);
    Workspace space(2, 2, 4);
    GameState* deviceState;
    Workspace* deviceSpace;
    int* result;
    cudaMallocCopy(deviceState, state);
    cudaMallocCopy(deviceSpace, space);
    CUDA_CHECK(cudaMalloc(&result, 2 * sizeof(int)));
    checkPhysics<<<1, 1>>>(deviceState, deviceSpace, result);
    CUDA_CHECK(cudaGetLastError());
    int counts[2];
    CUDA_CHECK(cudaMemcpy(counts, result, sizeof(counts), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(deviceState));
    CUDA_CHECK(cudaFree(deviceSpace));
    std::printf("%d/%d physics regression checks passed\n", counts[0] - counts[1], counts[0]);
    return counts[1] ? 1 : 0;
}
