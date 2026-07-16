#pragma once

#include "RLConstants.cuh"

struct BallCarContact
{
    Vec3 point;
    Vec3 normal;
    float depth;
    bool hit;
};

CARL_D CARL_FI BallCarContact sphereOBBContact(
    const Vec3& ballPos,
    const Vec3& carCen,
    const Quat& carRot)
{
    BallCarContact contact{};
    const Vec3 localBall = carRot.toLocal(ballPos - carCen);
    const Vec3 clamped = {
        clampf(localBall.x, -CAR_BALL_CORE_HALF_EX.x,
            CAR_BALL_CORE_HALF_EX.x),
        clampf(localBall.y, -CAR_BALL_CORE_HALF_EX.y,
            CAR_BALL_CORE_HALF_EX.y),
        clampf(localBall.z, -CAR_BALL_CORE_HALF_EX.z,
            CAR_BALL_CORE_HALF_EX.z)
    };
    const Vec3 delta = localBall - clamped;
    const float distSq = delta.lenSq();
    const float intersectDist = BALL_RADIUS + CAR_BALL_SHAPE_MARGIN;
    const float acceptDist = intersectDist + CAR_BALL_CONTACT_BREAK;

    // Bullet clamps against the core box, then pushes out by its margin
    if (distSq > 1e-8f)
    {
        if (distSq >= acceptDist * acceptDist) return contact;

        const float dist = sqrtf(distSq);
        const Vec3 localNormal = delta * (1.f / dist);

        contact.point = carCen + carRot.toWorld(
            clamped + localNormal * CAR_BALL_SHAPE_MARGIN);
        contact.normal = carRot.toWorld(localNormal);
        contact.depth = intersectDist - dist;
        contact.hit = true;
        return contact;
    }

    const Vec3 faceDist = CAR_BALL_CORE_HALF_EX - localBall.abs();
    int axis = 0;
    float minDist = faceDist.x;

    if (faceDist.y < minDist)
    {
        axis = 1;
        minDist = faceDist.y;
    }
    if (faceDist.z < minDist)
    {
        axis = 2;
        minDist = faceDist.z;
    }

    Vec3 localNormal = Vec3::zero();
    localNormal[axis] = localBall[axis] >= 0.f ? 1.f : -1.f;

    contact.point = carCen + carRot.toWorld(
        clamped + localNormal * CAR_BALL_SHAPE_MARGIN);
    contact.normal = carRot.toWorld(localNormal);
    contact.depth = intersectDist + minDist;
    contact.hit = true;
    return contact;
}
