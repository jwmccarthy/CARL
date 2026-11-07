#pragma once

#include "Arena/ArenaMesh.cuh"
#include "Physics/BallArena/Contact.cuh"
#include "Physics/BallArena/Solver.cuh"
#include "Physics/CarBall/Solve.cuh"
#include "Physics/Goals.cuh"
#include "State/Events.cuh"
#include "State/GameState.cuh"

CARL_D __noinline__ Vec3 solveBallArena(
    GameState* state,
    ArenaMesh* arena,
    int ballIdx,
    const Vec3& preSolveVel,
    const Vec3& velNoExt,
    EventBuffer events = {})
{
    BallArenaSolverBody body = {
        state->ball.pos[ballIdx],
        preSolveVel,
        velNoExt,
        state->ball.ang[ballIdx],
        Vec3::zero(),
        Vec3::zero(),
        Vec3::zero(),
        Vec3::zero()
    };

    BallArenaContact contacts[MAX_BALL_ARENA_CONTACTS];
    const int count = gatherBallArenaContacts(contacts, body.pos, arena);

    if (events.enabled())
    {
        int surfaces = 0;

        for (int i = 0; i < count; i++)
        {
            const Vec3 normal = contacts[i].normal.norm();
            surfaces |= arenaSurface(normal);
        }

        events.ballSurface(ballIdx, surfaces);
    }

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

// Solve ball collisions and integrate the result for one sim
CARL_D CARL_FI void solveBallForSim(
    GameState* __restrict__ state,
    ArenaMesh* __restrict__ arena,
    int simIdx,
    BallTouchBuffer touches = {},
    EventBuffer events = {})
{
    const int ballIdx = simIdx;
    const Vec3 ballVel = state->ball.vel[ballIdx];
    const Vec3 ballAng = state->ball.ang[ballIdx];
    const bool sleeping = ballVel.lenSq() == 0.f && ballAng.lenSq() == 0.f;

    if (events.enabled())
    {
        events.ballSpeedBefore[simIdx] = ballVel.len();
    }

    auto& ball = state->ball;
    const Vec3 exportedPos = ball.pos[ballIdx];
    const bool valid = ball.integrationValid[ballIdx] != 0;

    // Reuse native-unit state unless an external caller changed the exported state
    Vec3 posBT = valid && (exportedPos - ball.exportedPos[ballIdx]).lenSq() == 0.f
        ? ball.integrationPos[ballIdx]
        : exportedPos * .02f;
    Vec3 velBT = valid && (ballVel - ball.exportedVel[ballIdx]).lenSq() == 0.f
        ? ball.integrationVel[ballIdx]
        : ballVel * .02f;

    // Apply exponential drag followed by gravity
    const Vec3 dampedBT = sleeping ? velBT : velBT * powf(1.f - BALL_DRAG, PHYS_DT);
    const Vec3 preSolveBT = sleeping ? velBT : dampedBT + (WORLD_GRAVITY * .02f) * PHYS_DT;
    const Vec3 velNoExt = dampedBT * 50.f;
    const Vec3 preSolveVel = preSolveBT * 50.f;

    Vec3 pushVel = Vec3::zero();

    if (!sleeping)
    {
        pushVel = solveBallArena(
            state, arena, ballIdx, preSolveVel, velNoExt, events);
    }
    else if (events.enabled())
    {
        BallArenaContact contacts[MAX_BALL_ARENA_CONTACTS];
        const Vec3 position = ball.pos[ballIdx];
        const int count = gatherBallArenaContacts(contacts, position, arena);

        int surfaces = 0;

        for (int i = 0; i < count; i++)
        {
            const Vec3 normal = contacts[i].normal.norm();
            surfaces |= arenaSurface(normal);
        }

        events.ballSurface(ballIdx, surfaces);
    }

    solveBallCarForSim(state, simIdx, preSolveVel, touches, events);

    // Integrate position after all collision solves
    const Vec3 correctedPos = ball.pos[ballIdx];

    if ((correctedPos - exportedPos).lenSq() != 0.f)
    {
        posBT = correctedPos * .02f;
    }

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

    detectGoal(state, simIdx, ballPos, events);
}
