#pragma once

#include "State/GameState.cuh"

CARL_D CARL_FI void respawnCar(
    GameState* __restrict__ state,
    int carIdx)
{
    const int localIdx = carIdx % state->nCars;
    const int spawnIdx = state->cars.internal[carIdx].respawnIdx;
    const CarSpawn spawn = RESPAWN_LOCATIONS[spawnIdx];
    
    const bool orange = localIdx >= state->nBlue;
    const float y = orange ? -spawn.y : spawn.y;
    const float yaw = orange ? spawn.yaw + PI : spawn.yaw;
    const Quat rot = Quat::angle(yaw);

    state->cars.pos[carIdx] = { spawn.x, y, CAR_RESPAWN_Z };
    state->cars.cen[carIdx] = state->cars.pos[carIdx] 
                            + rot.toWorld(CAR_OFFSETS);
    state->cars.rot[carIdx] = rot;
    state->cars.vel[carIdx] = Vec3::zero();
    state->cars.ang[carIdx] = Vec3::zero();
    state->cars.imp[carIdx] = Vec3::zero();

    state->cars.isDemoed[carIdx] = 0;
    state->cars.demoRespawnTimer[carIdx] = 0.f;
    state->cars.carContactIdx[carIdx] = -1;
    state->cars.carContactCooldown[carIdx] = 0.f;
    state->cars.ballHitTick[carIdx] = -1;
    state->cars.ballContactTick[carIdx] = -1;

    CarInternalState internal{};
    internal.isOnGround = 1;
    internal.boost = BOOST_SPAWN_AMOUNT;
    state->cars.internal[carIdx] = internal;
}

CARL_D CARL_FI int respawnInputDir(const CarControls& controls)
{
    if (controls.steer < -DEMO_RESPAWN_INPUT_DEADZONE) return -1;
    if (controls.steer > DEMO_RESPAWN_INPUT_DEADZONE) return 1;
    return 0;
}

CARL_D CARL_FI void updateRespawnSelection(
    GameState* __restrict__ state,
    int carIdx)
{
    CarInternalState& internal = state->cars.internal[carIdx];
    const CarControls& controls = state->cars.controls[carIdx];
    const int dir = respawnInputDir(controls);

    if (dir != 0 && dir != internal.lastRespawnDir)
    {
        if (dir < 0 && internal.respawnIdx > 0)
        {
            internal.respawnIdx--;
        }
        else if (dir > 0 && internal.respawnIdx < 3)
        {
            internal.respawnIdx++;
        }
    }

    internal.lastRespawnDir = dir;
}

CARL_D CARL_FI void finishBallTick(
    GameState* __restrict__ state,
    int ballIdx)
{
    const Vec3 impulse = Vec3::ldg(state->ball.imp[ballIdx]);
    Vec3 vel = Vec3::ldg(state->ball.vel[ballIdx]) + impulse;
    Vec3 ang = Vec3::ldg(state->ball.ang[ballIdx]);
    float speedSq = vel.lenSq();

    if (speedSq > BALL_MAX_SPEED * BALL_MAX_SPEED)
    {
        vel = vel * (BALL_MAX_SPEED * rsqrtf(speedSq));
    }

    speedSq = ang.lenSq();
    if (speedSq > BALL_MAX_ANG_SPEED * BALL_MAX_ANG_SPEED)
    {
        ang = ang * (BALL_MAX_ANG_SPEED * rsqrtf(speedSq));
    }

    state->ball.vel[ballIdx] = vel;
    state->ball.ang[ballIdx] = ang;
    state->ball.imp[ballIdx] = Vec3::zero();
}

CARL_D CARL_FI void finishCarTick(
    GameState* __restrict__ state,
    int carIdx)
{
    if (__ldg(&state->cars.isDemoed[carIdx]))
    {
        float timer = __ldg(&state->cars.demoRespawnTimer[carIdx]);

        // Selection starts on the tick after demolition, not the impact tick.
        if (timer <= DEMO_RESPAWN_TIME)
        {
            updateRespawnSelection(state, carIdx);
        }

        timer = fmaxf(timer - PHYS_DT, 0.f);
        state->cars.demoRespawnTimer[carIdx] = timer;

        if (timer <= 0.f) respawnCar(state, carIdx);
        return;
    }

    const Vec3 impulse = Vec3::ldg(state->cars.imp[carIdx]);
    if (impulse.lenSq() > 1e-12f)
    {
        Vec3 vel = Vec3::ldg(state->cars.vel[carIdx]) + impulse;
        const float speedSq = vel.lenSq();

        if (speedSq > CAR_MAX_SPEED * CAR_MAX_SPEED)
        {
            vel = vel * (CAR_MAX_SPEED * rsqrtf(speedSq));
        }

        state->cars.vel[carIdx] = vel;
        state->cars.imp[carIdx] = Vec3::zero();
    }

    float cooldown = __ldg(&state->cars.carContactCooldown[carIdx]);
    if (cooldown <= 0.f) return;

    cooldown = fmaxf(cooldown - PHYS_DT, 0.f);
    state->cars.carContactCooldown[carIdx] = cooldown;
}
