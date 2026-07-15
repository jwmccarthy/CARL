#pragma once

#include "Physics/CarArena/SolverBody.cuh"
#include "Physics/Collision/Solver.cuh"
#include "Types.cuh"

struct PairSolverRow
{
    Vec3 axis;
    Vec3 relCrossA;
    Vec3 relCrossB;
    Vec3 angularA;
    Vec3 angularB;

    float jacInv;
    float rhs;
    float rhsPen;
    float applied;
    float appliedPush;
};

struct PairSolverContact
{
    PairSolverRow normal;
    PairSolverRow friction;
};

struct PairImpulse
{
    float normal;
    float friction;
};

CARL_D CARL_FI PairSolverRow makePairAxisRow(
    const SolverBody& bodyA,
    const SolverBody& bodyB,
    const Vec3& axis,
    const Vec3& relA,
    const Vec3& relB)
{
    PairSolverRow row{};

    row.axis = axis;
    row.relCrossA = relA.cross(axis);
    row.relCrossB = relB.cross(axis);
    row.angularA = applyInvInertiaWorld(bodyA, row.relCrossA);
    row.angularB = applyInvInertiaWorld(bodyB, row.relCrossB);

    const float denom = 2.f * CAR_INV_MASS
        + axis.dot(row.angularA.cross(relA))
        + axis.dot(row.angularB.cross(relB));
    row.jacInv = 1.f / fmaxf(denom, 1e-8f);

    return row;
}

CARL_D CARL_FI float pairRowVelocity(
    const PairSolverRow& row,
    const Vec3& linearA,
    const Vec3& angularA,
    const Vec3& linearB,
    const Vec3& angularB)
{
    return row.axis.dot(linearA) + row.relCrossA.dot(angularA)
        - row.axis.dot(linearB) - row.relCrossB.dot(angularB);
}

CARL_D CARL_FI PairSolverRow makePairNormalRow(
    const SolverBody& bodyA,
    const SolverBody& bodyB,
    const CarCarContact& contact,
    const Vec3& relA,
    const Vec3& relB)
{
    PairSolverRow row =
        makePairAxisRow(bodyA, bodyB, contact.normal, relA, relB);
    const float relVel = pairRowVelocity(
        row, bodyA.vel, bodyA.ang, bodyB.vel, bodyB.ang);
    const float bounce = relVel < -CAR_RESTITUTION_VEL_THRESH
        ? CARCAR_COLLISION_RESTITUTION * -relVel
        : 0.f;
    const float penetration = fmaxf(
        contact.depth - CAR_SOLVER_LINEAR_SLOP, 0.f);

    row.rhs = (bounce - relVel) * row.jacInv;
    row.rhsPen = penetration * CAR_CONTACT_ERP / PHYS_DT * row.jacInv;

    return row;
}

CARL_D CARL_FI PairSolverRow makePairFrictionRow(
    const SolverBody& bodyA,
    const SolverBody& bodyB,
    const CarCarContact& contact,
    const Vec3& relA,
    const Vec3& relB)
{
    const Vec3 velA = pointVelocity(bodyA, relA);
    const Vec3 velB = pointVelocity(bodyB, relB);
    const Vec3 rel = velA - velB;
    const float normalVel = contact.normal.dot(rel);
    Vec3 tangent = rel - contact.normal * normalVel;
    const float tangentLenSq = tangent.lenSq();
    const float threshold =
        fmaxf(1.192092896e-07f, normalVel * normalVel * 1e-8f);

    tangent = tangentLenSq > threshold
        ? tangent * rsqrtf(tangentLenSq)
        : fallbackTangent(contact.normal);

    PairSolverRow row = makePairAxisRow(bodyA, bodyB, tangent, relA, relB);
    const float tangentVel = pairRowVelocity(
        row, bodyA.vel, bodyA.ang, bodyB.vel, bodyB.ang);
    row.rhs = -tangentVel * row.jacInv;

    return row;
}

CARL_D CARL_FI void applyPairImpulse(
    SolverBody& bodyA,
    SolverBody& bodyB,
    const PairSolverRow& row,
    float impulse,
    bool split)
{
    Vec3& linearA = split ? bodyA.pushVel : bodyA.deltaVel;
    Vec3& angularA = split ? bodyA.turnVel : bodyA.deltaAng;
    Vec3& linearB = split ? bodyB.pushVel : bodyB.deltaVel;
    Vec3& angularB = split ? bodyB.turnVel : bodyB.deltaAng;

    linearA = linearA + row.axis * (CAR_INV_MASS * impulse);
    angularA = angularA + row.angularA * impulse;
    linearB = linearB - row.axis * (CAR_INV_MASS * impulse);
    angularB = angularB - row.angularB * impulse;
}

CARL_D CARL_FI void resolvePairRow(
    SolverBody& bodyA,
    SolverBody& bodyB,
    PairSolverRow& row,
    float lower,
    float upper)
{
    const float relVel = pairRowVelocity(
        row,
        bodyA.deltaVel,
        bodyA.deltaAng,
        bodyB.deltaVel,
        bodyB.deltaAng);
    const float impulse = solveImpulse(
        row.rhs, 0.f, row.jacInv, relVel, lower, upper, row.applied);

    applyPairImpulse(bodyA, bodyB, row, impulse, false);
}

CARL_D CARL_FI void resolvePairLower(
    SolverBody& bodyA,
    SolverBody& bodyB,
    PairSolverRow& row)
{
    const float relVel = pairRowVelocity(
        row,
        bodyA.deltaVel,
        bodyA.deltaAng,
        bodyB.deltaVel,
        bodyB.deltaAng);
    const float impulse = solveLowerImpulse(
        row.rhs, 0.f, row.jacInv, relVel, 0.f, row.applied);

    applyPairImpulse(bodyA, bodyB, row, impulse, false);
}

CARL_D CARL_FI void resolvePairSplit(
    SolverBody& bodyA,
    SolverBody& bodyB,
    PairSolverRow& row)
{
    const float relVel = pairRowVelocity(
        row,
        bodyA.pushVel,
        bodyA.turnVel,
        bodyB.pushVel,
        bodyB.turnVel);
    const float impulse = solveLowerImpulse(
        row.rhsPen,
        0.f,
        row.jacInv,
        relVel,
        0.f,
        row.appliedPush);

    applyPairImpulse(bodyA, bodyB, row, impulse, true);
}

CARL_D __noinline__ void solvePairContacts(
    SolverBody& bodyA,
    SolverBody& bodyB,
    const CarCarContact* contacts,
    int count,
    PairImpulse* impulses)
{
    PairSolverContact rows[MAX_CAR_MANIFOLD_POINTS]{};

    #pragma unroll
    for (int i = 0; i < count; i++)
    {
        const Vec3 relA =
            bodyA.cen - bodyA.pos + bodyA.rot.toWorld(contacts[i].carPosA);
        const Vec3 relB =
            bodyB.cen - bodyB.pos + bodyB.rot.toWorld(contacts[i].carPosB);

        rows[i].normal =
            makePairNormalRow(bodyA, bodyB, contacts[i], relA, relB);
        rows[i].friction =
            makePairFrictionRow(bodyA, bodyB, contacts[i], relA, relB);
    }

    for (int iter = 0; iter < CAR_SOLVER_ITERS; iter++)
    {
        #pragma unroll
        for (int i = 0; i < count; i++)
        {
            resolvePairSplit(bodyA, bodyB, rows[i].normal);
        }
    }

    for (int iter = 0; iter < CAR_SOLVER_ITERS; iter++)
    {
        #pragma unroll
        for (int i = 0; i < count; i++)
        {
            resolvePairLower(bodyA, bodyB, rows[i].normal);
        }

        #pragma unroll
        for (int i = 0; i < count; i++)
        {
            const float maxImpulse =
                CARCAR_COLLISION_FRICTION * rows[i].normal.applied;
            resolvePairRow(
                bodyA,
                bodyB,
                rows[i].friction,
                -maxImpulse,
                maxImpulse);
        }
    }

    #pragma unroll
    for (int i = 0; i < count; i++)
    {
        impulses[i] = {
            rows[i].normal.applied,
            rows[i].friction.applied
        };
    }
}
