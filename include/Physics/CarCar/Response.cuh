#pragma once

#include "Physics/CarArena/SolverBody.cuh"
#include "Physics/Integration.cuh"
#include "Cuda/Random.cuh"
#include "Types.cuh"

CARL_D CARL_FI float bumpGround(float speed)
{
    if (speed <= 0.f)
    {
        return 5.f / 6.f;
    }
    if (speed <= 1400.f)
    {
        return 5.f / 6.f + (1100.f - 5.f / 6.f) * (speed / 1400.f);
    }
    if (speed <= 2200.f)
    {
        return 1100.f + (1530.f - 1100.f) * ((speed - 1400.f) / 800.f);
    }
    return 1530.f;
}

CARL_D CARL_FI float bumpAir(float speed)
{
    if (speed <= 0.f)
    {
        return 5.f / 6.f;
    }
    if (speed <= 1400.f)
    {
        return 5.f / 6.f + (1390.f - 5.f / 6.f) * (speed / 1400.f);
    }
    if (speed <= 2200.f)
    {
        return 1390.f + (1945.f - 1390.f) * ((speed - 1400.f) / 800.f);
    }
    return 1945.f;
}

CARL_D CARL_FI float bumpUp(float speed)
{
    if (speed <= 0.f)
    {
        return 2.f / 6.f;
    }
    if (speed <= 1400.f)
    {
        return 2.f / 6.f + (278.f - 2.f / 6.f) * (speed / 1400.f);
    }
    if (speed <= 2200.f)
    {
        return 278.f + (417.f - 278.f) * ((speed - 1400.f) / 800.f);
    }
    return 417.f;
}

CARL_D CARL_FI bool frontHit(
    const CarPairCollision& pair,
    bool aIsBumper,
    const Vec3& delta,
    const Quat& bumperRot)
{
    if (bumperRot.toLocal(delta).x > BUMP_MIN_FORWARD_DIST) return true;

    for (int i = 0; i < pair.count; i++)
    {
        const Vec3 point =
            aIsBumper ? pair.contacts[i].carPosA : pair.contacts[i].carPosB;
        if (point.x + CAR_OFFSETS.x > BUMP_MIN_FORWARD_DIST) return true;
    }

    return false;
}

CARL_D CARL_FI DemoHit findDemo(
    GameState* __restrict__ state,
    const CarPairCollision& pair,
    float penetration)
{
    const int teamA = (pair.carA % state->nCars) < state->nBlue ? 0 : 1;
    const int teamB = (pair.carB % state->nCars) < state->nBlue ? 0 : 1;
    if (teamA == teamB || penetration < 5.f) return {};

    const Vec3 velA = Vec3::ldg(state->cars.vel[pair.carA]);
    const Vec3 velB = Vec3::ldg(state->cars.vel[pair.carB]);
    const Vec3 posA = Vec3::ldg(state->cars.pos[pair.carA]);
    const Vec3 posB = Vec3::ldg(state->cars.pos[pair.carB]);

    for (int side = 0; side < 2; side++)
    {
        const bool aIsBumper = side == 0;
        const int bumper = aIsBumper ? pair.carA : pair.carB;
        const int victim = aIsBumper ? pair.carB : pair.carA;
        const Vec3 bumperVel = aIsBumper ? velA : velB;
        const Vec3 victimVel = aIsBumper ? velB : velA;
        const Vec3 bumperPos = aIsBumper ? posA : posB;
        const Vec3 victimPos = aIsBumper ? posB : posA;
        const Quat bumperRot = aIsBumper ? pair.boxA.rot : pair.boxB.rot;

        const float speed = bumperVel.len();
        if (speed < SUPERSONIC_START_SPEED) continue;

        const Vec3 delta = victimPos - bumperPos;
        if (bumperVel.dot(delta) <= 0.f) continue;

        const Vec3 velDir = bumperVel / speed;
        const Vec3 hitDir = delta.norm();
        const float closingSpeed = bumperVel.dot(hitDir);
        if (closingSpeed <= victimVel.dot(velDir)) continue;
        if (!frontHit(pair, aIsBumper, delta, bumperRot)) continue;

        return { bumper, victim };
    }

    return {};
}

CARL_D CARL_FI void applyDemo(
    GameState* __restrict__ state,
    const DemoHit& demo)
{
    Vec3 pos = Vec3::ldg(state->cars.pos[demo.victim]);
    Vec3 cen;
    Quat rot = Quat::ldg(state->cars.rot[demo.victim]);

    integrateCarState(
        pos, cen, rot,
        Vec3::ldg(state->cars.vel[demo.victim]),
        Vec3::ldg(state->cars.ang[demo.victim]));

    state->cars.pos[demo.victim] = pos;
    state->cars.cen[demo.victim] = cen;
    state->cars.rot[demo.victim] = rot;
    state->cars.isDemoed[demo.victim] = 1;
    state->cars.demoRespawnTimer[demo.victim] =
        DEMO_RESPAWN_TIME + PHYS_DT;
    state->cars.carContactIdx[demo.bumper] = demo.victim;
    state->cars.carContactCooldown[demo.bumper] = BUMP_COOLDOWN_TIME;

    CarInternalState& victim = state->cars.internal[demo.victim];
    const uint32_t seed = (uint32_t)state->seed
        ^ (uint32_t)demo.victim * 0x9e3779b9u
        ^ (uint32_t)state->tickCount;
    victim.respawnIdx = randomIndex(seed, 4);
    victim.lastRespawnDir = 0;
}

CARL_D CARL_FI void tryApplyBump(
    GameState* __restrict__ state,
    const CarPairCollision& pair,
    const SolverBody& bodyA,
    const SolverBody& bodyB,
    const Vec3& preVelA,
    const Vec3& preVelB,
    bool aIsBumper)
{
    const int bumper = aIsBumper ? pair.carA : pair.carB;
    const int victim = aIsBumper ? pair.carB : pair.carA;
    const Vec3 bumperVel = aIsBumper ? preVelA : preVelB;
    const Vec3 victimVel = aIsBumper ? preVelB : preVelA;
    const Vec3 bumperPos = aIsBumper ? bodyA.pos : bodyB.pos;
    const Vec3 victimPos = aIsBumper ? bodyB.pos : bodyA.pos;
    const Quat bumperRot = aIsBumper ? bodyA.rot : bodyB.rot;
    const Quat victimRot = aIsBumper ? bodyB.rot : bodyA.rot;

    const int lastVictim = __ldg(&state->cars.carContactIdx[bumper]);
    const float cooldown = __ldg(&state->cars.carContactCooldown[bumper]);
    if (lastVictim == victim && cooldown > 0.f) return;

    const Vec3 delta = victimPos - bumperPos;
    if (bumperVel.dot(delta) <= 0.f) return;

    const float speed = bumperVel.len();
    if (speed < 1e-6f) return;

    const Vec3 velDir = bumperVel / speed;
    const Vec3 hitDir = delta.norm();
    const float closingSpeed = bumperVel.dot(hitDir);
    if (closingSpeed <= victimVel.dot(velDir)) return;
    if (!frontHit(pair, aIsBumper, delta, bumperRot)) return;

    const bool grounded = state->cars.internal[victim].isOnGround != 0;
    const float forwardScale =
        grounded ? bumpGround(closingSpeed) : bumpAir(closingSpeed);

    const Vec3 up = grounded ? victimRot.toWorld(WORLD_Z) : WORLD_Z;
    const Vec3 impulse = velDir * forwardScale + up * bumpUp(closingSpeed);

    state->cars.imp[victim] = Vec3::ldg(state->cars.imp[victim]) + impulse;
    state->cars.carContactIdx[bumper] = victim;
    state->cars.carContactCooldown[bumper] = BUMP_COOLDOWN_TIME;
}
