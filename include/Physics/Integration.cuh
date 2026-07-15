#pragma once

#include "../DataUtils.cuh"
#include "../State/GameState.cuh"

CARL_HD CARL_FI Quat integrateQuat(
    const Quat& rot,
    const Vec3& ang)
{
    constexpr float dt = PHYS_DT;
    constexpr float maxMotion = CAR_ANGULAR_MOTION_THRESH;

    const float angSpeedSq = ang.lenSq();
    float angSpeed = angSpeedSq > 1e-8f ? sqrtf(angSpeedSq) : 0.f;

    if (angSpeed * dt > maxMotion)
    {
        angSpeed = maxMotion / dt;
    }

    Vec3 axis;

    if (angSpeed < 0.001f)
    {
        // Taylor series fallback for degenerate angles
        const float dtSq = dt * dt;
        const float scale = 0.5f * dt - dtSq * dt * 0.020833333333f
                          * angSpeed * angSpeed;
        axis = ang * scale;
    }
    else
    {
        axis = ang * (sinf(0.5f * angSpeed * dt) / angSpeed);
    }

    const Quat delta = { axis.x, axis.y, axis.z, cosf(0.5f * angSpeed * dt) };
    return delta.comp(rot).norm();
}

CARL_D CARL_FI void integrateCarState(
    Vec3& pos,
    Vec3& cen,
    Quat& rot,
    const Vec3& vel,
    const Vec3& ang)
{
    pos = pos + vel * PHYS_DT;
    rot = integrateQuat(rot, ang);
    cen = pos + rot.toWorld(CAR_OFFSETS);
}

CARL_D CARL_FI CarPose predictCarPose(
    GameState* __restrict__ state,
    const int carIdx,
    const Quat& currRot)
{
    Vec3 cen;
    Vec3 pos = Vec3::ldg(state->cars.pos[carIdx]);
    Quat rot = currRot;

    integrateCarState(
        pos, cen, rot,
        Vec3::ldg(state->cars.vel[carIdx]),
        Vec3::ldg(state->cars.ang[carIdx]));

    return { cen, rot };
}
