#pragma once

#include "Clip.cuh"
#include "Response.cuh"
#include "Solver.cuh"

CARL_D CARL_FI void clearCarCarManifold(
    CarCarManifold& manifold,
    int pairIdx)
{
    manifold.count[pairIdx] = 0;
}

CARL_D CARL_FI void writeCarCarManifold(
    CarCarManifold& manifold,
    const CarPairCollision& pair,
    const PairImpulse* impulses)
{
    const int base = pair.pairIdx * MAX_CAR_MANIFOLD_POINTS;
    manifold.count[pair.pairIdx] = pair.count;

    for (int i = 0; i < pair.count; i++)
    {
        const int idx = base + i;
        manifold.carPosA[idx] = pair.contacts[i].carPosA;
        manifold.carPosB[idx] = pair.contacts[i].carPosB;
        manifold.normal[idx] = pair.contacts[i].normal;
        manifold.dist[idx] = pair.contacts[i].depth;
        manifold.impulse[idx] = impulses[i].normal;
        manifold.impulseTan[idx] = impulses[i].friction;
        manifold.lifetime[idx] = 1;
    }
}

CARL_D CARL_FI void applyPairResult(
    SolverBody& bodyA,
    SolverBody& bodyB)
{
    bodyA.vel = bodyA.vel + bodyA.deltaVel;
    bodyA.ang = bodyA.ang + bodyA.deltaAng;
    bodyB.vel = bodyB.vel + bodyB.deltaVel;
    bodyB.ang = bodyB.ang + bodyB.deltaAng;

    // Split impulse changes position without leaking into reported velocity
    bodyA.pos = bodyA.pos + bodyA.pushVel * PHYS_DT;
    bodyB.pos = bodyB.pos + bodyB.pushVel * PHYS_DT;
    bodyA.cen = bodyA.pos + bodyA.rot.toWorld(CAR_OFFSETS);
    bodyB.cen = bodyB.pos + bodyB.rot.toWorld(CAR_OFFSETS);

    clampSolverVelocity(bodyA);
    clampSolverVelocity(bodyB);
}

CARL_D CARL_FI void solveCarCarPair(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    int pairIdx,
    int carA,
    int carB);

CARL_D CARL_FI void solveCarCarForSim(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    int simIdx)
{
    const int carBase = simIdx * state->nCars;
    int pairIdx = simIdx * space->ccMan.maxPairsPerSim;

    for (int a = 0; a < state->nCars - 1; a++)
    {
        for (int b = a + 1; b < state->nCars; b++)
        {
            solveCarCarPair(state, space, pairIdx, carBase + a, carBase + b);
            pairIdx++;
        }
    }
}

CARL_D CARL_FI void solveCarCarPair(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    int pairIdx,
    int carA,
    int carB)
{
    CarCarManifold& manifold = space->ccMan;
    if (__ldg(&state->cars.isDemoed[carA])
        || __ldg(&state->cars.isDemoed[carB]))
    {
        clearCarCarManifold(manifold, pairIdx);
        return;
    }

    CarPairCollision pair{};
    pair.pairIdx = pairIdx;
    pair.carA = carA;
    pair.carB = carB;
    pair.boxA.cen = Vec3::ldg(state->cars.cen[carA]);
    pair.boxB.cen = Vec3::ldg(state->cars.cen[carB]);

    const Vec3 centerDelta = pair.boxB.cen - pair.boxA.cen;
    const float maxDist = 2.f * CAR_HALF_EX.len();
    if (centerDelta.lenSq() > maxDist * maxDist)
    {
        clearCarCarManifold(manifold, pairIdx);
        return;
    }

    pair.boxA.rot = Quat::ldg(state->cars.rot[carA]);
    pair.boxB.rot = Quat::ldg(state->cars.rot[carB]);

    const SATResult sat = carCarSAT(pair.boxA, pair.boxB);
    if (!sat.overlap)
    {
        clearCarCarManifold(manifold, pairIdx);
        return;
    }

    pair.count = carCarContacts(sat, pair.boxA, pair.boxB, pair.contacts);
    if (pair.count == 0)
    {
        clearCarCarManifold(manifold, pairIdx);
        return;
    }

    const DemoHit demo = findDemo(state, pair, sat.minPen);
    SolverBody bodyA = loadSolverBody(state, carA);
    SolverBody bodyB = loadSolverBody(state, carB);
    const Vec3 preVelA = bodyA.vel;
    const Vec3 preVelB = bodyB.vel;
    PairImpulse impulses[MAX_CAR_MANIFOLD_POINTS]{};

    solvePairContacts(bodyA, bodyB, pair.contacts, pair.count, impulses);
    applyPairResult(bodyA, bodyB);

    writeSolverBody(state, bodyA, carA);
    writeSolverBody(state, bodyB, carB);
    writeCarCarManifold(manifold, pair, impulses);

    if (demo.valid())
    {
        applyDemo(state, demo);
        clearCarCarManifold(manifold, pairIdx);
        return;
    }

    tryApplyBump(state, pair, bodyA, bodyB, preVelA, preVelB, true);
    tryApplyBump(state, pair, bodyA, bodyB, preVelA, preVelB, false);
}
