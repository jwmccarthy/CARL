#pragma once

#include "DataUtils.cuh"

#include "Cuda/Math.cuh"
#include "RLConstants.cuh"

struct AABB
{
    Vec3 min, max;
};

CARL_HD CARL_FI Vec3 carAABBHalfEx(const CarPose& pose)
{
    Vec3 rx = pose.rot.toWorld(WORLD_X);
    Vec3 ry = pose.rot.toWorld(WORLD_Y);
    Vec3 rz = pose.rot.toWorld(WORLD_Z);

    return rx.abs() * CAR_HALF_EX.x + 
           ry.abs() * CAR_HALF_EX.y + 
           rz.abs() * CAR_HALF_EX.z;
}

CARL_HD CARL_FI AABB carAABB(const CarPose& pose)
{
    Vec3 halfEx = carAABBHalfEx(pose);
    return { pose.cen - halfEx, pose.cen + halfEx };
}

CARL_HD CARL_FI Vec3 carAABBMin(const CarPose& pose)
{
    Vec3 halfEx = carAABBHalfEx(pose);
    return pose.cen - halfEx;
}

CARL_HD CARL_FI AABB triAABB(const Vec3& v0, const Vec3& v1, const Vec3& v2)
{
    return { v0.min(v1).min(v2), v0.max(v1).max(v2) };
}

CARL_HD CARL_FI bool testAABBOverlap(const AABB& a, const AABB& b)
{
    return a.min.x <= b.max.x && a.max.x >= b.min.x &&
           a.min.y <= b.max.y && a.max.y >= b.min.y &&
           a.min.z <= b.max.z && a.max.z >= b.min.z;
}