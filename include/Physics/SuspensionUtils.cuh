#pragma once

#include "../RLConstants.cuh"

CARL_HD CARL_FI Vec3 suspConnection(int wheel)
{
    const float x = wheel < 2 ? FRONT_SUSP_X : BACK_SUSP_X;
    const float absY = wheel < 2 ? FRONT_SUSP_Y : BACK_SUSP_Y;
    const float y = wheel & 1 ? -absY : absY;
    return { x, y, 0.f };
}

CARL_HD CARL_FI float suspRayLength(int wheel)
{
    return wheel < 2 ? FRONT_SUSP_RAY_LEN : BACK_SUSP_RAY_LEN;
}

CARL_HD CARL_FI float suspRadius(int wheel)
{
    return wheel < 2 ? FRONT_WHEEL_RADIUS : BACK_WHEEL_RADIUS;
}

CARL_HD CARL_FI float suspRestLength(int wheel)
{
    return wheel < 2 ? FRONT_SUSPENSION_REST : BACK_SUSPENSION_REST;
}

CARL_HD CARL_FI float suspForceScale(int wheel)
{
    return wheel < 2 ? SUSP_FORCE_SCALE_FRONT : SUSP_FORCE_SCALE_BACK;
}
