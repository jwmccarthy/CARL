#pragma once

#include "Arena/ArenaMesh.cuh"
#include "Physics/BallArena/Contact.cuh"
#include "Physics/BallArena/Solver.cuh"
#include "State/GameState.cuh"

CARL_D __noinline__ Vec3 solveBallArena(
    GameState* state,
    ArenaMesh* arena,
    int ballIdx,
    const Vec3& preSolveVel)
{
    BallArenaSolverBody body = {
        Vec3::ldg(state->ball.pos[ballIdx]),
        preSolveVel,
        Vec3::ldg(state->ball.ang[ballIdx]),
        Vec3::zero(),
        Vec3::zero(),
        Vec3::zero(),
        Vec3::zero()
    };
    BallArenaContact contacts[4];
    const int count = gatherBallArenaContacts(contacts, body.pos, arena);

    solveBallArenaContacts(body, contacts, count);

    state->ball.vel[ballIdx] = body.vel;
    state->ball.ang[ballIdx] = body.ang;
    return body.pushVel;
}
