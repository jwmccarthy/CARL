#pragma once

#include "State/GameState.cuh"

struct BoostPad
{
    Vec3 pos;
    float radius;
    float amount;
    float cooldown;
};

constexpr float SMALL_PAD_RADIUS = 144.f;
constexpr float BIG_PAD_RADIUS = 208.f;
constexpr float BOOST_PAD_HEIGHT = 95.f;
constexpr float SMALL_PAD_AMOUNT = 12.f;
constexpr float BIG_PAD_AMOUNT = 100.f;
constexpr float SMALL_PAD_COOLDOWN = 4.f;
constexpr float BIG_PAD_COOLDOWN = 10.f;

constexpr CARL_D BoostPad BOOST_PADS[NUM_BOOST_PADS] = {
    {{ -3584.f,     0.f, 73.f }, BIG_PAD_RADIUS, BIG_PAD_AMOUNT, BIG_PAD_COOLDOWN},
    {{  3584.f,     0.f, 73.f }, BIG_PAD_RADIUS, BIG_PAD_AMOUNT, BIG_PAD_COOLDOWN},
    {{ -3072.f,  4096.f, 73.f }, BIG_PAD_RADIUS, BIG_PAD_AMOUNT, BIG_PAD_COOLDOWN},
    {{  3072.f,  4096.f, 73.f }, BIG_PAD_RADIUS, BIG_PAD_AMOUNT, BIG_PAD_COOLDOWN},
    {{ -3072.f, -4096.f, 73.f }, BIG_PAD_RADIUS, BIG_PAD_AMOUNT, BIG_PAD_COOLDOWN},
    {{  3072.f, -4096.f, 73.f }, BIG_PAD_RADIUS, BIG_PAD_AMOUNT, BIG_PAD_COOLDOWN},
    {{     0.f, -4240.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{ -1792.f, -4184.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  1792.f, -4184.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  -940.f, -3308.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{   940.f, -3308.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{     0.f, -2816.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{ -3584.f, -2484.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  3584.f, -2484.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{ -1788.f, -2300.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  1788.f, -2300.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{ -2048.f, -1036.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{     0.f, -1024.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  2048.f, -1036.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{ -1024.f,     0.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  1024.f,     0.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{ -2048.f,  1036.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{     0.f,  1024.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  2048.f,  1036.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{ -1788.f,  2300.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  1788.f,  2300.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{ -3584.f,  2484.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  3584.f,  2484.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{     0.f,  2816.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  -940.f,  3308.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{   940.f,  3308.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{ -1792.f,  4184.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{  1792.f,  4184.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN},
    {{     0.f,  4240.f, 70.f }, SMALL_PAD_RADIUS, SMALL_PAD_AMOUNT, SMALL_PAD_COOLDOWN}
};

CARL_D CARL_FI bool insideBoostPad(
    const Vec3& carPos,
    const BoostPad& pad)
{
    const float dx = carPos.x - pad.pos.x;
    const float dy = carPos.y - pad.pos.y;
    const float distSq = dx * dx + dy * dy;

    return distSq < pad.radius * pad.radius
        && fabsf(carPos.z - pad.pos.z) < BOOST_PAD_HEIGHT;
}

CARL_D CARL_FI void processBoostPad(
    GameState* __restrict__ state,
    int slotIdx)
{
    const int simIdx = slotIdx / NUM_BOOST_PADS;
    const int padIdx = slotIdx - simIdx * NUM_BOOST_PADS;

    float cooldown = state->boostPadCooldowns[slotIdx];
    if (cooldown > 0.f)
    {
        cooldown = fmaxf(cooldown - PHYS_DT, 0.f);
        state->boostPadCooldowns[slotIdx] = cooldown;
    }
    if (cooldown > 0.f) return;

    const BoostPad pad = BOOST_PADS[padIdx];
    const int carBase = simIdx * state->nCars;
    int lockedCar = -1;

    // The last colliding car owns the pad, matching RocketSim's lock update
    for (int carOffset = 0; carOffset < state->nCars; carOffset++)
    {
        const int carIdx = carBase + carOffset;
        if (__ldg(&state->cars.isDemoed[carIdx])) continue;

        const Vec3 carPos = Vec3::ldg(state->cars.pos[carIdx]);
        if (insideBoostPad(carPos, pad)) lockedCar = carIdx;
    }

    if (lockedCar < 0) return;

    CarInternalState& car = state->cars.internal[lockedCar];
    if (car.boost >= BOOST_MAX) return;

    car.boost = fminf(car.boost + pad.amount, BOOST_MAX);
    state->boostPadCooldowns[slotIdx] = pad.cooldown;
}
