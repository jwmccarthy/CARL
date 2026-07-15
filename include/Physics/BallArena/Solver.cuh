#pragma once

#include "Physics/BallArena/Contact.cuh"
#include "Physics/Collision/Solver.cuh"
#include "RLConstants.cuh"

struct BallArenaSolverBody
{
    Vec3 pos;
    Vec3 vel;
    Vec3 ang;
    Vec3 deltaVel;
    Vec3 deltaAng;
    Vec3 pushVel;
    Vec3 pushAng;
};

struct BallArenaSolverRow
{
    Vec3 axis;
    Vec3 relCross;
    float jacInv;
    float rhs;
    float rhsPen;
    float applied;
    float appliedPush;
    float lower;
    float upper;
};

struct BallArenaSolverContact
{
    BallArenaSolverRow normal;
    BallArenaSolverRow friction;
};

CARL_D CARL_FI float ballArenaRowVelocity(
    const BallArenaSolverRow& row,
    const Vec3& linear,
    const Vec3& angular)
{
    return row.axis.dot(linear) + row.relCross.dot(angular);
}

CARL_D CARL_FI BallArenaSolverRow makeBallArenaAxisRow(
    const Vec3& axis,
    const Vec3& relPos)
{
    BallArenaSolverRow row{};
    row.axis = axis;
    row.relCross = relPos.cross(axis);

    const float denom = BALL_INV_MASS
        + BALL_INV_INERTIA * row.relCross.lenSq();
    row.jacInv = 1.f / fmaxf(denom, 1e-8f);
    row.upper = 1e10f;
    return row;
}

CARL_D CARL_FI BallArenaSolverRow makeBallArenaNormalRow(
    const BallArenaSolverBody& body,
    const BallArenaContact& contact,
    const Vec3& relPos)
{
    BallArenaSolverRow row =
        makeBallArenaAxisRow(contact.normal, relPos);
    const float relVel = ballArenaRowVelocity(row, body.vel, body.ang);
    float bounce = fabsf(relVel) > CAR_RESTITUTION_VEL_THRESH
        ? BALL_WORLD_RESTITUTION * -relVel
        : 0.f;
    bounce = fmaxf(bounce, 0.f);
    const float posError = contact.depth * CAR_CONTACT_ERP / PHYS_DT;
    const float posImpulse = posError * row.jacInv;
    const float velImpulse = (bounce - relVel) * row.jacInv;

    if (-contact.depth > CAR_SPLIT_PENETRATION_THRESH)
    {
        row.rhs = posImpulse + velImpulse;
    }
    else
    {
        row.rhs = velImpulse;
        row.rhsPen = posImpulse;
    }

    return row;
}

CARL_D CARL_FI BallArenaSolverRow makeBallArenaFrictionRow(
    const BallArenaSolverBody& body,
    const BallArenaContact& contact,
    const Vec3& relPos)
{
    const Vec3 pointVel = body.vel + body.ang.cross(relPos);
    const float normalVel = contact.normal.dot(pointVel);
    Vec3 tangent = pointVel - contact.normal * normalVel;
    const float tangentLenSq = tangent.lenSq();

    tangent = tangentLenSq > 1e-8f
        ? tangent * rsqrtf(tangentLenSq)
        : fallbackTangent(contact.normal);

    BallArenaSolverRow row = makeBallArenaAxisRow(tangent, relPos);
    const float tangentVel = ballArenaRowVelocity(row, body.vel, body.ang);
    row.rhs = -tangentVel * row.jacInv;
    row.upper = 0.f;
    return row;
}

CARL_D CARL_FI void applyBallArenaImpulse(
    BallArenaSolverBody& body,
    const BallArenaSolverRow& row,
    float impulse,
    bool split)
{
    Vec3& linear = split ? body.pushVel : body.deltaVel;
    Vec3& angular = split ? body.pushAng : body.deltaAng;

    linear = linear + row.axis * (BALL_INV_MASS * impulse);
    angular = angular + row.relCross * (BALL_INV_INERTIA * impulse);
}

CARL_D CARL_FI void resolveBallArenaSplit(
    BallArenaSolverBody& body,
    BallArenaSolverRow& row)
{
    if (row.rhsPen == 0.f) return;

    const float relVel =
        ballArenaRowVelocity(row, body.pushVel, body.pushAng);
    const float impulse = solveLowerImpulse(
        row.rhsPen,
        0.f,
        row.jacInv,
        relVel,
        row.lower,
        row.appliedPush);
    applyBallArenaImpulse(body, row, impulse, true);
}

CARL_D CARL_FI void resolveBallArenaNormal(
    BallArenaSolverBody& body,
    BallArenaSolverRow& row)
{
    const float relVel =
        ballArenaRowVelocity(row, body.deltaVel, body.deltaAng);
    const float impulse = solveLowerImpulse(
        row.rhs, 0.f, row.jacInv, relVel, row.lower, row.applied);
    applyBallArenaImpulse(body, row, impulse, false);
}

CARL_D CARL_FI void resolveBallArenaFriction(
    BallArenaSolverBody& body,
    BallArenaSolverRow& row,
    float maxImpulse)
{
    const float relVel =
        ballArenaRowVelocity(row, body.deltaVel, body.deltaAng);
    const float impulse = solveImpulse(
        row.rhs,
        0.f,
        row.jacInv,
        relVel,
        -maxImpulse,
        maxImpulse,
        row.applied);
    applyBallArenaImpulse(body, row, impulse, false);
}

CARL_D CARL_FI void clampBallArenaVelocity(BallArenaSolverBody& body)
{
    const float speedSq = body.vel.lenSq();
    if (speedSq > BALL_MAX_SPEED * BALL_MAX_SPEED)
    {
        body.vel = body.vel * (BALL_MAX_SPEED * rsqrtf(speedSq));
    }

    const float angSpeedSq = body.ang.lenSq();
    if (angSpeedSq > BALL_MAX_ANG_SPEED * BALL_MAX_ANG_SPEED)
    {
        body.ang = body.ang
            * (BALL_MAX_ANG_SPEED * rsqrtf(angSpeedSq));
    }
}

CARL_D CARL_FI void solveBallArenaContacts(
    BallArenaSolverBody& body,
    const BallArenaContact* contacts,
    int count)
{
    BallArenaSolverContact rows[4]{};

    #pragma unroll
    for (int i = 0; i < count; i++)
    {
        const Vec3 relPos =
            contacts[i].normal * (-BALL_COLLISION_RADIUS);
        rows[i].normal =
            makeBallArenaNormalRow(body, contacts[i], relPos);
        rows[i].friction =
            makeBallArenaFrictionRow(body, contacts[i], relPos);
    }

    for (int iter = 0; iter < CAR_SOLVER_ITERS; iter++)
    {
        #pragma unroll
        for (int i = 0; i < count; i++)
        {
            resolveBallArenaSplit(body, rows[i].normal);
        }
    }

    for (int iter = 0; iter < CAR_SOLVER_ITERS; iter++)
    {
        #pragma unroll
        for (int i = 0; i < count; i++)
        {
            resolveBallArenaNormal(body, rows[i].normal);
        }

        #pragma unroll
        for (int i = 0; i < count; i++)
        {
            const float maxImpulse =
                BALL_WORLD_FRICTION * rows[i].normal.applied;
            if (maxImpulse <= 0.f) continue;

            resolveBallArenaFriction(body, rows[i].friction, maxImpulse);
        }
    }

    body.vel = body.vel + body.deltaVel;
    body.ang = body.ang + body.deltaAng;
    clampBallArenaVelocity(body);
}
