#pragma once

#include "Cuda/Math.cuh"
#include "RLConstants.cuh"

struct CarBox
{
    Vec3 cen;
    Quat rot;

    CARL_D CARL_FI Vec3 axis(int i) const
    {
        return rot.toWorld(WORLD_AXES[i]);
    }

    CARL_D CARL_FI Vec3 support(const Vec3& dir) const
    {
        const Vec3 localDir = rot.toLocal(dir);
        const Vec3 local = {
            copysignf(CAR_HALF_EX.x, localDir.x),
            copysignf(CAR_HALF_EX.y, localDir.y),
            copysignf(CAR_HALF_EX.z, localDir.z)
        };

        return cen + rot.toWorld(local);
    }
};

struct CarCarContact
{
    Vec3 carPosA;
    Vec3 carPosB;
    Vec3 normal;
    float depth;
};

struct CarPairCollision
{
    int pairIdx;
    int carA;
    int carB;
    CarBox boxA;
    CarBox boxB;
    CarCarContact contacts[MAX_CAR_MANIFOLD_POINTS];
    int count;
};

struct DemoHit
{
    int bumper = -1;
    int victim = -1;

    CARL_D CARL_FI bool valid() const
    {
        return victim >= 0;
    }
};
