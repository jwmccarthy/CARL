#pragma once

#include "Arena/ArenaMesh.cuh"
#include "Physics/BallArena/Contact.cuh"
#include "Physics/BallArena/Solver.cuh"
#include "Physics/CarBall/Solve.cuh"
#include "Physics/Goals.cuh"
#include "State/GameState.cuh"

CARL_D __noinline__ Vec3 solveBallArena(
    GameState* state,
    ArenaMesh* arena,
    int ballIdx,
    const Vec3& preSolveVel,
    const Vec3& velNoExt)
{
    BallArenaSolverBody body = {
        Vec3::ldg(state->ball.pos[ballIdx]),
        preSolveVel,
        velNoExt,
        Vec3::ldg(state->ball.ang[ballIdx]),
        Vec3::zero(),
        Vec3::zero(),
        Vec3::zero(),
        Vec3::zero()
    };
    BallArenaContact contacts[4];
    const int count = gatherBallArenaContacts(contacts, body.pos, arena);

    bool ballWasHit = false;
    const int carBase = ballIdx * state->nCars;
    for (int carOffset = 0; carOffset < state->nCars; carOffset++)
    {
        ballWasHit |= __ldg(&state->cars.ballHitTick[carBase + carOffset]) >= 0;
    }
    solveBallArenaContacts(body, contacts, count, ballWasHit);

    state->ball.vel[ballIdx] = body.vel;
    state->ball.ang[ballIdx] = body.ang;

    return body.pushVel;
}

// Solve ball-arena, car-ball, integrate position, detect goals for one sim
CARL_D CARL_FI void solveBallForSim(
    GameState* __restrict__ state,
    ArenaMesh* __restrict__ arena,
    int simIdx)
{
    const int ballIdx = simIdx;
    const Vec3 ballVel = Vec3::ldg(state->ball.vel[ballIdx]);
    const Vec3 ballAng = Vec3::ldg(state->ball.ang[ballIdx]);
    const bool sleeping = ballVel.lenSq() == 0.f && ballAng.lenSq() == 0.f;

    // Damped velocity: exponential drag then gravity, matching Bullet
    const Vec3 velNoExt = sleeping
        ? ballVel
        : ballVel * powf(1.f - BALL_DRAG, PHYS_DT);
    const Vec3 preSolveVel = sleeping
        ? ballVel
        : velNoExt + WORLD_GRAVITY * PHYS_DT;

    Vec3 pushVel = Vec3::zero();
    
    if (!sleeping)
    {
        pushVel = solveBallArena(
            state, arena, ballIdx, preSolveVel, velNoExt);
    }

    solveBallCarForSim(state, simIdx, preSolveVel);

    // Integrate position after all collision solves
    Vec3 ballPos = Vec3::ldg(state->ball.pos[ballIdx]);
    ballPos = ballPos + pushVel * PHYS_DT;
    ballPos = ballPos + Vec3::ldg(state->ball.vel[ballIdx]) * PHYS_DT;
    state->ball.pos[ballIdx] = ballPos;

    detectGoal(state, simIdx, ballPos);
}
