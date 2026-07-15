#pragma once

#include "../../Cuda/Math.cuh"

struct LocalTriangle
{
    int triIdx;
    Vec3 v0, v1, v2;
    Vec3 e0, e1, e2;

    CARL_D CARL_FI void set(int idx, Vec3 a, Vec3 b, Vec3 c)
    {
        triIdx = idx;
        v0 = a;
        v1 = b;
        v2 = c;

        e0 = v1 - v0;
        e1 = v2 - v1;
        e2 = v0 - v2;
    }

    CARL_D CARL_FI Vec3 getVert(int i) const
    {
        return i == 0 ? v0 : (i == 1 ? v1 : v2);
    }

    CARL_D CARL_FI Vec3 getEdge(int i) const
    {
        return i == 0 ? e0 : (i == 1 ? e1 : e2);
    }

    CARL_D CARL_FI Vec3 center() const
    {
        return (v0 + v1 + v2) / 3.f;
    }

    CARL_D CARL_FI Vec3 normal() const
    {
        return e0.cross(v2 - v0).norm();
    }
};
