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
    BallArenaContact contacts[MAX_BALL_ARENA_CONTACTS];
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

    auto &ball = state->ball;
    const Vec3 exportedPos = Vec3::ldg(ball.pos[ballIdx]);
    const bool valid = ball.integrationValid[ballIdx] != 0;
    Vec3 posBT = valid && (exportedPos - Vec3::ldg(ball.exportedPos[ballIdx])).lenSq() == 0.f
                     ? Vec3::ldg(ball.integrationPos[ballIdx])
                     : exportedPos * .02f;
    Vec3 velBT = valid && (ballVel - Vec3::ldg(ball.exportedVel[ballIdx])).lenSq() == 0.f
                     ? Vec3::ldg(ball.integrationVel[ballIdx])
                     : ballVel * .02f;

    // Damped velocity: exponential drag then gravity, matching Bullet
    const Vec3 dampedBT = sleeping ? velBT : velBT * powf(1.f - BALL_DRAG, PHYS_DT);
    const Vec3 preSolveBT = sleeping ? velBT : dampedBT + (WORLD_GRAVITY * .02f) * PHYS_DT;
    const Vec3 velNoExt = dampedBT * 50.f, preSolveVel = preSolveBT * 50.f;

    Vec3 pushVel = Vec3::zero();
    
    if (!sleeping)
    {
        pushVel = solveBallArena(
            state, arena, ballIdx, preSolveVel, velNoExt);
    }

    solveBallCarForSim(state, simIdx, preSolveVel);

    // Integrate position after all collision solves
    const Vec3 correctedPos = ball.pos[ballIdx];
    if ((correctedPos - exportedPos).lenSq() != 0.f)
        posBT = correctedPos * .02f;
    const Vec3 finalVel = ball.vel[ballIdx];
    velBT = (finalVel - preSolveVel).lenSq() == 0.f ? preSolveBT : finalVel * .02f;
    posBT = posBT + (pushVel * .02f) * PHYS_DT;
    posBT = posBT + velBT * PHYS_DT;
    const Vec3 ballPos = posBT * 50.f;
    state->ball.pos[ballIdx] = ballPos;
    ball.integrationPos[ballIdx] = posBT;
    ball.integrationVel[ballIdx] = velBT;
    ball.exportedPos[ballIdx] = ballPos;
    ball.exportedVel[ballIdx] = finalVel;
    ball.integrationValid[ballIdx] = 1;

    detectGoal(state, simIdx, ballPos);
}
