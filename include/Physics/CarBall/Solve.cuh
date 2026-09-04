#pragma once

#include "Physics/Collision/Solver.cuh"
#include "Physics/Integration.cuh"
#include "RLConstants.cuh"
#include "State/GameState.cuh"
#include "Contact.cuh"
#include "ExtraImpulse.cuh"

struct BallCarBody
{
    Vec3 pos;
    Vec3 cen;
    Vec3 vel;
    Vec3 ang;
    Quat rot;
};

struct BallCarRow
{
    Vec3 axis;
    Vec3 ballTorque;
    Vec3 carAngular;
    float jacInv;
    float applied;
};

struct BallCarConstraint
{
    BallCarRow normal;
    BallCarRow friction;
    Vec3 ballOffset;
    Vec3 carOffset;
    float restitution;
};

CARL_D CARL_FI BallCarBody loadBallCarBody(
    GameState* state,
    int carIdx)
{
    return {
        Vec3::ldg(state->cars.pos[carIdx]),
        Vec3::ldg(state->cars.cen[carIdx]),
        Vec3::ldg(state->cars.vel[carIdx]),
        Vec3::ldg(state->cars.ang[carIdx]),
        Quat::ldg(state->cars.rot[carIdx])
    };
}

CARL_D CARL_FI BallCarRow makeBallCarRow(
    const BallCarBody& car,
    const Vec3& axis,
    const Vec3& ballOffset,
    const Vec3& carOffset)
{
    BallCarRow row{};
    row.axis = axis;
    row.ballTorque = ballOffset.cross(axis);

    const Vec3 carTorque = carOffset.cross(axis);

    row.carAngular = car.rot.toWorld(
        car.rot.toLocal(carTorque) * CAR_INV_INERTIA);

    const float denom = BALL_INV_MASS + CAR_INV_MASS
        + BALL_INV_INERTIA * row.ballTorque.lenSq()
        + row.carAngular.dot(carTorque);

    row.jacInv = 1.f / fmaxf(denom, 1e-8f);

    return row;
}

CARL_D CARL_FI float ballCarRowVelocity(
    const BallCarRow& row,
    const Vec3& ballVel,
    const Vec3& ballAng,
    const BallCarBody& car,
    const Vec3& ballOffset,
    const Vec3& carOffset)
{
    const Vec3 ballPoint = ballVel + ballAng.cross(ballOffset);
    const Vec3 carPoint = car.vel + car.ang.cross(carOffset);
    return row.axis.dot(ballPoint - carPoint);
}

CARL_D CARL_FI void applyBallCarRow(
    Vec3& ballVel,
    Vec3& ballAng,
    BallCarBody& car,
    const BallCarRow& row,
    float impulse)
{
    ballVel = ballVel + row.axis * (impulse * BALL_INV_MASS);
    ballAng = ballAng + row.ballTorque * (impulse * BALL_INV_INERTIA);
    car.vel = car.vel - row.axis * (impulse * CAR_INV_MASS);
    car.ang = car.ang - row.carAngular * impulse;
}

CARL_D CARL_FI BallCarConstraint makeBallCarConstraint(
    const BallCarBody& car,
    const Vec3& ballVel,
    const Vec3& ballAng,
    const BallCarContact& contact)
{
    BallCarConstraint constraint{};
    constraint.ballOffset = contact.normal * -BALL_RADIUS;
    constraint.carOffset = contact.point - car.pos;
    constraint.normal = makeBallCarRow(
        car, contact.normal, constraint.ballOffset, constraint.carOffset);

    const float relNormal = ballCarRowVelocity(
        constraint.normal,
        ballVel,
        ballAng,
        car,
        constraint.ballOffset,
        constraint.carOffset);

    Vec3 tangent = (ballVel + ballAng.cross(constraint.ballOffset))
        - (car.vel + car.ang.cross(constraint.carOffset));
    tangent = tangent - contact.normal * relNormal;

    const float tangentLenSq = tangent.lenSq();

    constraint.restitution = fabsf(relNormal) > CAR_RESTITUTION_VEL_THRESH
        ? CAR_BALL_RESTITUTION * -relNormal
        : 0.f;
    tangent = tangentLenSq > 1.192092896e-07f
        ? tangent * rsqrtf(tangentLenSq)
        : fallbackTangent(contact.normal);
    constraint.friction = makeBallCarRow(
        car, tangent, constraint.ballOffset,
        constraint.carOffset);

    return constraint;
}

CARL_D CARL_FI void solveBallCarConstraint(
    Vec3& ballVel,
    Vec3& ballAng,
    BallCarBody& car,
    BallCarConstraint& constraint)
{
    for (int iter = 0; iter < CAR_SOLVER_ITERS; iter++)
    {
        BallCarRow& normal = constraint.normal;
        const float relNormal = ballCarRowVelocity(
            normal, ballVel, ballAng, car,
            constraint.ballOffset,
            constraint.carOffset);

        const float normalImpulse = solveLowerImpulse(
            constraint.restitution * normal.jacInv,
            0.f, normal.jacInv, relNormal,
            0.f, normal.applied);

        applyBallCarRow(
            ballVel, ballAng, car, normal, normalImpulse);

        BallCarRow& friction = constraint.friction;
        const float relTangent = ballCarRowVelocity(
            friction, ballVel, ballAng, car,
            constraint.ballOffset,
            constraint.carOffset);

        const float maxFriction = CAR_BALL_FRICTION * constraint.normal.applied;
        const float frictionImpulse = solveImpulse(
            0.f, 0.f, friction.jacInv,
            relTangent, -maxFriction,
            maxFriction, friction.applied);

        applyBallCarRow(
            ballVel, ballAng, car, friction, frictionImpulse);
    }
}

CARL_D CARL_FI void applyBallCarSplitCorrection(
    GameState* state,
    int ballIdx,
    int carIdx,
    const BallCarContact& contact,
    const BallCarRow& normal)
{
    const float penetration = contact.depth - CAR_SOLVER_LINEAR_SLOP;
    if (penetration <= 0.f) return;

    const float pushImpulse = penetration * CAR_CONTACT_ERP
        / PHYS_DT * normal.jacInv;
    const Vec3 ballCorrection = contact.normal
        * (pushImpulse * BALL_INV_MASS * PHYS_DT);
    const Vec3 carCorrection = contact.normal
        * (-pushImpulse * CAR_INV_MASS * PHYS_DT);
    const Vec3 carTurnVelocity = normal.carAngular * -pushImpulse;

    state->ball.pos[ballIdx] = Vec3::ldg(state->ball.pos[ballIdx]) + ballCorrection;
    const Vec3 carPos = Vec3::ldg(state->cars.pos[carIdx]) + carCorrection;
    state->cars.pos[carIdx] = carPos;
    const Quat carRot = integrateQuat(
        Quat::ldg(state->cars.rot[carIdx]),
        carTurnVelocity * CAR_SPLIT_IMPULSE_TURN_ERP);
    state->cars.rot[carIdx] = carRot;
    state->cars.cen[carIdx] = carPos + carRot.toWorld(CAR_OFFSETS);
}

CARL_D CARL_FI void solveBallCarPair(
    GameState* state,
    int ballIdx,
    int carIdx,
    const Vec3& preSolveBallVel)
{
    if (__ldg(&state->cars.isDemoed[carIdx])) return;

    const Vec3 ballPos = Vec3::ldg(state->ball.pos[ballIdx]);

    Vec3 ballVel = Vec3::ldg(state->ball.vel[ballIdx]);
    Vec3 ballAng = Vec3::ldg(state->ball.ang[ballIdx]);
    BallCarBody car = loadBallCarBody(state, carIdx);

    const Vec3 preSolveCarVel = car.vel;
    const BallCarContact contact = sphereOBBContact(ballPos, car.cen, car.rot);
    if (!contact.hit) return;

    BallCarConstraint constraint = makeBallCarConstraint(car, ballVel, ballAng, contact);

    solveBallCarConstraint(
        ballVel, ballAng, car, constraint);

    state->ball.vel[ballIdx] = ballVel;
    state->ball.ang[ballIdx] = ballAng;
    state->cars.vel[carIdx] = car.vel;
    state->cars.ang[carIdx] = car.ang;

    applyBallCarSplitCorrection(
        state, ballIdx, carIdx,
        contact, constraint.normal);

    applyBallCarExtraImpulse(
        state, ballIdx, carIdx, ballPos, car.pos,
        preSolveCarVel, car.rot, preSolveBallVel);
}

CARL_D __noinline__ void solveBallCarForSim(
    GameState* state,
    int simIdx,
    const Vec3& preSolveBallVel)
{
    const int ballIdx = simIdx;
    const int carBase = simIdx * state->nCars;
    const Vec3 ballPos = Vec3::ldg(state->ball.pos[ballIdx]);
    const float maxDist = BALL_RADIUS + CAR_BALL_SOLVER_HALF_EX.len();

    // Pair solves stay serial because each car sees prior pair results
    for (int carOffset = 0; carOffset < state->nCars; carOffset++)
    {
        const int carIdx = carBase + carOffset;
        if (__ldg(&state->cars.isDemoed[carIdx])) continue;

        const Vec3 carCen = Vec3::ldg(state->cars.cen[carIdx]);
        if ((ballPos - carCen).lenSq() > maxDist * maxDist) continue;

        solveBallCarPair(state, ballIdx, carIdx, preSolveBallVel);
    }
}
