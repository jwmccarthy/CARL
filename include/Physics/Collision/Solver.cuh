#pragma once

#include "RLConstants.cuh"

CARL_D CARL_FI Vec3 fallbackTangent(const Vec3& normal)
{
    if (fabsf(normal.z) > SQRT_1_2)
    {
        const float lenSq = normal.y * normal.y + normal.z * normal.z;
        const float invLen = rsqrtf(lenSq);
        return { 0.f, -normal.z * invLen, normal.y * invLen };
    }

    const float lenSq = normal.x * normal.x + normal.y * normal.y;
    const float invLen = rsqrtf(lenSq);
    return { -normal.y * invLen, normal.x * invLen, 0.f };
}

CARL_D CARL_FI float restitutionVelocity(
    float relVel,
    float restitution)
{
    if (fabsf(relVel) < CAR_RESTITUTION_VEL_THRESH) return 0.f;
    return restitution * -relVel;
}

CARL_D CARL_FI float solveImpulse(
    float rhs,
    float cfm,
    float jacInv,
    float relVel,
    float lower,
    float upper,
    float& applied)
{
    const float previous = applied;
    const float delta = rhs - applied * cfm - relVel * jacInv;

    applied = clampf(previous + delta, lower, upper);
    return applied - previous;
}

CARL_D CARL_FI float solveLowerImpulse(
    float rhs,
    float cfm,
    float jacInv,
    float relVel,
    float lower,
    float& applied)
{
    const float previous = applied;
    const float delta = rhs - applied * cfm - relVel * jacInv;

    applied = fmaxf(previous + delta, lower);
    return applied - previous;
}
