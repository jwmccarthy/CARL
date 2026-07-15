#pragma once

#include "../../RLConstants.cuh"
#include "../../State/GameState.cuh"

struct SolverBody
{
    Vec3 pos;
    Vec3 vel;
    Vec3 ang;
    Vec3 cen;
    Quat rot;

    Vec3 extVel;
    Vec3 extAng;
    Vec3 deltaVel;
    Vec3 deltaAng;
    Vec3 pushVel;
    Vec3 turnVel;
};

CARL_D CARL_FI SolverBody loadSolverBody(
    GameState* __restrict__ state,
    int carIdx)
{
    SolverBody body{};

    body.pos = Vec3::ldg(state->cars.pos[carIdx]);
    body.vel = Vec3::ldg(state->cars.vel[carIdx]);
    body.ang = Vec3::ldg(state->cars.ang[carIdx]);
    body.cen = Vec3::ldg(state->cars.cen[carIdx]);
    body.rot = Quat::ldg(state->cars.rot[carIdx]);
    body.extVel = WORLD_GRAVITY * PHYS_DT;

    return body;
}

CARL_D CARL_FI void writeSolverBody(
    GameState* __restrict__ state,
    const SolverBody& body,
    int carIdx)
{
    state->cars.pos[carIdx] = body.pos;
    state->cars.vel[carIdx] = body.vel;
    state->cars.ang[carIdx] = body.ang;
    state->cars.cen[carIdx] = body.cen;
    state->cars.rot[carIdx] = body.rot;
}

CARL_D CARL_FI Vec3 applyInvInertiaWorld(
    const SolverBody& body,
    const Vec3& value)
{
    const Vec3 local = body.rot.toLocal(value);
    return body.rot.toWorld(local * CAR_INV_INERTIA);
}

CARL_D CARL_FI Vec3 pointVelocity(
    const SolverBody& body,
    const Vec3& relPos)
{
    return body.vel + body.ang.cross(relPos);
}

CARL_D CARL_FI void clampSolverVelocity(SolverBody& body)
{
    const float velLenSq = body.vel.lenSq();
    if (velLenSq > CAR_MAX_SPEED * CAR_MAX_SPEED)
    {
        body.vel = body.vel * (CAR_MAX_SPEED * rsqrtf(velLenSq));
    }

    const float angLenSq = body.ang.lenSq();
    if (angLenSq > CAR_MAX_ANG_SPEED * CAR_MAX_ANG_SPEED)
    {
        body.ang = body.ang * (CAR_MAX_ANG_SPEED * rsqrtf(angLenSq));
    }
}
