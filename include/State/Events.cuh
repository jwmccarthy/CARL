#pragma once

#include "Cuda/Math.cuh"

enum CarlEventType
{
    EVENT_BALL_TOUCH = 1 << 0,
    EVENT_JUMP = 1 << 1,
    EVENT_DOUBLE_JUMP = 1 << 2,
    EVENT_DODGE = 1 << 3,
    EVENT_CONTACT_SURFACE = 1 << 4,
    EVENT_LEAVE_SURFACE = 1 << 5,
    EVENT_FLIP_RESET = 1 << 8,
    EVENT_DEMOED = 1 << 9,
    EVENT_GOAL = 1 << 10,
    EVENT_CEILING_BODY_CONTACT = 1 << 11,
    EVENT_BALL_ACCELERATES = 1 << 14
};

enum ArenaSurface
{
    ARENA_FLOOR = 1 << 0,
    ARENA_SIDE_WALL = 1 << 1,
    ARENA_BACK_WALL = 1 << 2,
    ARENA_CEILING = 1 << 3,
    ARENA_OTHER = 1 << 4
};

CARL_HD CARL_FI int arenaSurface(const Vec3& normal)
{
    const float x = fabsf(normal.x);
    const float y = fabsf(normal.y);
    const float z = fabsf(normal.z);

    if (z >= x && z >= y)
    {
        return normal.z >= 0.f ? ARENA_FLOOR : ARENA_CEILING;
    }

    return x >= y ? ARENA_SIDE_WALL : ARENA_BACK_WALL;
}

CARL_HD CARL_FI int carArenaSurface(const Vec3& normal)
{
    const int surface = arenaSurface(normal);

    if (surface == ARENA_SIDE_WALL
        && (fabsf(normal.y) > 1e-3f || fabsf(normal.z) > 1e-3f))
    {
        if (normal.z > 1e-3f) return ARENA_FLOOR;
        if (normal.z < -1e-3f) return ARENA_CEILING;
        return ARENA_OTHER;
    }

    return surface;
}

enum BallTouchSurface
{
    TOUCH_FRONT,
    TOUCH_REAR,
    TOUCH_LEFT,
    TOUCH_RIGHT,
    TOUCH_ROOF,
    TOUCH_UNDERSIDE
};

struct BallTouchBuffer
{
    int* surfaces = nullptr;
    int* tickSurfaces = nullptr;
    float* tickSpeedBefore = nullptr;

    CARL_HD CARL_FI bool enabled() const { return surfaces != nullptr; }

    CARL_D CARL_FI void record(int carIdx, BallTouchSurface surface, float speedBefore) const
    {
        if (!enabled()) return;
        surfaces[carIdx] |= 1 << static_cast<int>(surface);
        if (tickSurfaces) tickSurfaces[carIdx] |= 1 << static_cast<int>(surface);
        if (tickSpeedBefore) tickSpeedBefore[carIdx] = speedBefore;
    }
};

struct EventBuffer
{
    int* carFlags = nullptr;
    int* ballFlags = nullptr;
    int* carSurfaces = nullptr;
    int* carSurfaceContacted = nullptr;
    int* carSurfaceLeft = nullptr;
    int* ballSurfaces = nullptr;
    int* ballSurfaceEntered = nullptr;
    int* ballSurfaceLeft = nullptr;
    int* goalSide = nullptr;
    float* ballSpeedChange = nullptr;
    float* ballSpeedBefore = nullptr;

    BallTouchBuffer touches;
    float* touchSpeed = nullptr;

    int nSim = 0;
    int nCars = 0;

    CARL_HD CARL_FI bool enabled() const { return carFlags != nullptr; }

    CARL_D CARL_FI void car(int carIdx, int flags) const
    {
        if (!enabled()) return;

        carFlags[carIdx] |= flags;
    }

    CARL_D CARL_FI void ball(int simIdx, int flags) const
    {
        if (!enabled()) return;

        ballFlags[simIdx] |= flags;
    }

    CARL_D CARL_FI void invalidate(int simIdx) const
    {
        if (!enabled()) return;

        ballSurfaces[simIdx] = -1;

        for (int local = 0; local < nCars; local++)
        {
            carSurfaces[simIdx * nCars + local] = -1;
        }
    }

    CARL_D CARL_FI void carSurface(int carIdx, int surfaces) const
    {
        if (!enabled()) return;

        const int previous = carSurfaces[carIdx];
        carSurfaces[carIdx] = surfaces;

        if (previous < 0) return;

        carSurfaceContacted[carIdx] |= surfaces & ~previous;
        carSurfaceLeft[carIdx] |= previous & ~surfaces;

        const bool wasGrounded = previous != 0;
        const bool isGrounded = surfaces != 0;

        if (!wasGrounded && isGrounded)
        {
            car(carIdx, EVENT_CONTACT_SURFACE);
        }

        if (wasGrounded && !isGrounded)
        {
            car(carIdx, EVENT_LEAVE_SURFACE);
        }
    }

    CARL_D CARL_FI void ballSurface(int simIdx, int surfaces) const
    {
        if (!enabled()) return;

        const int previous = ballSurfaces[simIdx];
        ballSurfaces[simIdx] = surfaces;

        if (previous < 0) return;

        const int entered = surfaces & ~previous;
        const int left = previous & ~surfaces;

        ballSurfaceEntered[simIdx] |= entered;
        ballSurfaceLeft[simIdx] |= left;

    }
};
