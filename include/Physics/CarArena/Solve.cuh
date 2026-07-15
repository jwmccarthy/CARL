#pragma once

#include "../../Arena/ArenaMesh.cuh"
#include "../../RLConstants.cuh"
#include "../../State/GameState.cuh"
#include "../../State/Workspace.cuh"
#include "../CarSuspension.cuh"
#include "../Collision/Solver.cuh"
#include "../Integration.cuh"
#include "Manifold.cuh"
#include "SolverBody.cuh"

struct SolverRow
{
    Vec3 axis;
    Vec3 relCross;
    Vec3 angular;

    float jacInv;
    float rhs;
    float rhsPen;
    float cfm;
    float lower;
    float upper;
    float applied;
    float appliedPush;
};

struct SolverContact
{
    SolverRow normal;
    SolverRow friction;
    float frictionCoeff;
};

struct SolverMotion
{
    Vec3 centerOffset;
    Vec3 totalVel;
    Vec3 totalAng;
};

CARL_D CARL_FI SolverRow makeAxisRow(
    const SolverBody& body,
    const Vec3& axis,
    const Vec3& relPos)
{
    SolverRow row{};

    row.axis = axis;
    row.relCross = relPos.cross(axis);
    row.angular = applyInvInertiaWorld(body, row.relCross);

    const Vec3 linearResponse = row.angular.cross(relPos);
    const float denom =
        CAR_INV_MASS + axis.dot(linearResponse);

    row.jacInv = 1.f / fmaxf(denom, 1e-8f);
    row.upper = 1e10f;

    return row;
}

CARL_D CARL_FI float rowVelocity(
    const SolverRow& row,
    const Vec3& linear,
    const Vec3& angular)
{
    return row.axis.dot(linear) + row.relCross.dot(angular);
}

CARL_D CARL_FI SolverRow makeNormalRow(
    const SolverBody& body,
    const ManifoldPoint& point,
    const Vec3& relPos,
    const SolverMotion& motion)
{
    SolverRow row = makeAxisRow(body, point.worldNormal, relPos);

    const float relVelNoExt = rowVelocity(row, body.vel, body.ang);
    const float relVel = rowVelocity(row, motion.totalVel, motion.totalAng);
    const float restitution = restitutionVelocity(relVelNoExt, point.restitution);
    const float penetration = point.dist + CAR_SOLVER_LINEAR_SLOP;
    const float invDt = 1.f / PHYS_DT;

    float posError = 0.f;
    float velError = restitution - relVel;

    if (penetration <= 0.f)
    {
        posError = -penetration * CAR_CONTACT_ERP * invDt;
    }
    else
    {
        velError -= penetration * invDt;
    }

    const float posImpulse = posError * row.jacInv;
    const float velImpulse = velError * row.jacInv;

    if (penetration > CAR_SPLIT_PENETRATION_THRESH)
    {
        row.rhs = posImpulse + velImpulse;
    }
    else
    {
        row.rhs = velImpulse;
        row.rhsPen = posImpulse;
    }

    row.applied = point.impulse * CAR_SOLVER_WARMSTART;

    return row;
}

CARL_D CARL_FI SolverRow makeFrictionRow(
    const SolverBody& body,
    const ManifoldPoint& point,
    const Vec3& relPos,
    const SolverMotion& motion)
{
    const Vec3 contactVel = motion.totalVel + body.ang.cross(relPos);
    const float normalVel = point.worldNormal.dot(contactVel);
    Vec3 tangent = contactVel - point.worldNormal * normalVel;
    const float tangentLenSq = tangent.lenSq();

    tangent = tangentLenSq > 1e-8f
        ? tangent * rsqrtf(tangentLenSq)
        : fallbackTangent(point.worldNormal);

    SolverRow row = makeAxisRow(body, tangent, relPos);
    const float tangentVel = rowVelocity(row, body.vel, body.ang);

    row.rhs = -tangentVel * row.jacInv;
    row.upper = 0.f;
    row.applied = 0.f;

    return row;
}

CARL_D CARL_FI SolverContact makeSolverContact(
    const SolverBody& body,
    const ManifoldPoint& point,
    const SolverMotion& motion)
{
    const Vec3 relPos =
        motion.centerOffset + body.rot.toWorld(point.carPos);

    return {
        makeNormalRow(body, point, relPos, motion),
        makeFrictionRow(body, point, relPos, motion),
        point.friction
    };
}

CARL_D CARL_FI void applyRowImpulse(
    SolverBody& body,
    const SolverRow& row,
    float impulse)
{
    body.deltaVel = body.deltaVel + row.axis * (CAR_INV_MASS * impulse);
    body.deltaAng = body.deltaAng + row.angular * impulse;
}

CARL_D CARL_FI void warmStartRow(
    SolverBody& body,
    const SolverRow& row)
{
    applyRowImpulse(body, row, row.applied);
}

CARL_D CARL_FI void resolveRow(
    SolverBody& body,
    SolverRow& row)
{
    const float relVel = rowVelocity(row, body.deltaVel, body.deltaAng);
    const float impulse = solveImpulse(
        row.rhs,
        row.cfm,
        row.jacInv,
        relVel,
        row.lower,
        row.upper,
        row.applied);

    applyRowImpulse(body, row, impulse);
}

CARL_D CARL_FI void resolveLowerRow(
    SolverBody& body,
    SolverRow& row)
{
    const float relVel = rowVelocity(row, body.deltaVel, body.deltaAng);
    const float impulse = solveLowerImpulse(
        row.rhs,
        row.cfm,
        row.jacInv,
        relVel,
        row.lower,
        row.applied);

    applyRowImpulse(body, row, impulse);
}

CARL_D CARL_FI void resolveSplitRow(
    SolverBody& body,
    SolverRow& row)
{
    if (row.rhsPen == 0.f) return;

    const float relVel = rowVelocity(row, body.pushVel, body.turnVel);
    const float impulse = solveLowerImpulse(
        row.rhsPen,
        row.cfm,
        row.jacInv,
        relVel,
        row.lower,
        row.appliedPush);

    body.pushVel = body.pushVel + row.axis * (CAR_INV_MASS * impulse);
    body.turnVel = body.turnVel + row.angular * impulse;
}

CARL_D __noinline__ void solveCarManifold(
    SolverBody& body,
    ManifoldPoint* points,
    int count)
{
    SolverContact contacts[MAX_CAR_MANIFOLD_POINTS]{};
    const SolverMotion motion = {
        body.cen - body.pos,
        body.vel + body.extVel,
        body.ang + body.extAng
    };

    #pragma unroll
    for (int i = 0; i < count; i++)
    {
        contacts[i] = makeSolverContact(body, points[i], motion);
        warmStartRow(body, contacts[i].normal);
    }

    // Split impulses correct penetration without changing reported velocity.
    for (int iter = 0; iter < CAR_SOLVER_ITERS; iter++)
    {
        #pragma unroll
        for (int i = 0; i < count; i++)
        {
            resolveSplitRow(body, contacts[i].normal);
        }
    }

    for (int iter = 0; iter < CAR_SOLVER_ITERS; iter++)
    {
        #pragma unroll
        for (int i = 0; i < count; i++)
        {
            resolveLowerRow(body, contacts[i].normal);
        }

        #pragma unroll
        for (int i = 0; i < count; i++)
        {
            SolverContact& contact = contacts[i];
            const float maxImpulse =
                contact.frictionCoeff * contact.normal.applied;
            if (maxImpulse <= 0.f) continue;

            contact.friction.lower = -maxImpulse;
            contact.friction.upper = maxImpulse;
            resolveRow(body, contact.friction);
        }
    }

    #pragma unroll
    for (int i = 0; i < count; i++)
    {
        points[i].impulse = contacts[i].normal.applied;
        points[i].impulseTan = contacts[i].friction.applied;
    }
}

CARL_D CARL_FI void integrateAndClampCar(SolverBody& body)
{
    integrateCarState(body.pos, body.cen, body.rot, body.vel, body.ang);
    clampSolverVelocity(body);
}

CARL_D CARL_FI void integrateSplitCorrection(SolverBody& body)
{
    const Vec3 splitAng = body.turnVel * CAR_SPLIT_IMPULSE_TURN_ERP;

    integrateCarState(
        body.pos,
        body.cen,
        body.rot,
        body.pushVel,
        splitAng);
}

CARL_D CARL_FI void solveCarArena(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena,
    int carIdx)
{
    if (__ldg(&state->cars.isDemoed[carIdx])) return;

    SolverBody body = loadSolverBody(state, carIdx);
    const CarPose pose = { body.cen, body.rot };

    applyCarSuspension(body, space, carIdx, state->tickCount);

    const int pairCount = carHitPairCount(space, carIdx);
    const int oldCount = __ldg(&space->ctMan.count[carIdx]);
    if (oldCount == 0 && pairCount == 0)
    {
        body.vel = body.vel + body.extVel + body.deltaVel;
        body.ang = body.ang + body.extAng + body.deltaAng;

        writeSolverBody(state, body, carIdx);
        return;
    }

    ManifoldPoint points[MAX_CAR_MANIFOLD_POINTS]{};
    const int count = buildLocalManifold(points, space, arena, carIdx, pose);

    if (count > 0)
    {
        solveCarManifold(body, points, count);
        integrateSplitCorrection(body);
    }

    writeLocalManifold(space->ctMan, carIdx, points, count);

    body.vel = body.vel + body.extVel + body.deltaVel;
    body.ang = body.ang + body.extAng + body.deltaAng;

    writeSolverBody(state, body, carIdx);
}

CARL_D CARL_FI void integrateCar(
    GameState* __restrict__ state,
    int carIdx)
{
    if (__ldg(&state->cars.isDemoed[carIdx])) return;

    SolverBody body = loadSolverBody(state, carIdx);

    integrateAndClampCar(body);

    writeSolverBody(state, body, carIdx);
}
