#pragma once

#include <vector>
#include <utility>
#include <stdexcept>

#include "../Cuda/Math.cuh"
#include "../Cuda/Common.cuh"
#include "../RLConstants.cuh"

constexpr float BOUNDARY_EDGE_ANGLE = 2.f * PI;

struct Tri
{
    Vec3 v0, v1, v2;

    CARL_HD CARL_FI Vec3& operator[](int i) { return (&v0)[i]; }
    CARL_HD CARL_FI Vec3 operator[](int i) const { return (&v0)[i]; }

    CARL_HD CARL_FI Vec3 getNormal()
    {
        return (v1 - v0).cross(v2 - v0).norm();
    }
};

inline std::vector<Vec3> triNormals(
    const std::vector<Vec3>& verts,
    const std::vector<Int3>& tris)
{
    std::vector<Vec3> norms;
    norms.reserve(tris.size());

    for (size_t i = 0; i < tris.size(); i++)
    {
        const Int3& tri = tris[i];
        const Vec3 cross = (verts[tri.y] - verts[tri.x])
            .cross(verts[tri.z] - verts[tri.x]);
        const float lenSq = cross.lenSq();

        if (lenSq <= 1e-12f)
            throw std::runtime_error("degenerate triangle in arena mesh");

        norms.push_back(cross * (1.f / sqrtf(lenSq)));
    }

    return norms;
}

inline std::pair<std::vector<Vec3>, std::vector<Vec3>> triBounds(
    const std::vector<Vec3>& verts,
    const std::vector<Int3>& tris)
{
    std::vector<Vec3> aabbMin;
    std::vector<Vec3> aabbMax;
    aabbMin.reserve(tris.size());
    aabbMax.reserve(tris.size());

    for (const Int3& tri : tris)
    {
        const Vec3 v0 = verts[tri.x];
        const Vec3 v1 = verts[tri.y];
        const Vec3 v2 = verts[tri.z];

        aabbMin.push_back(v0.min(v1).min(v2));
        aabbMax.push_back(v0.max(v1).max(v2));
    }

    return { std::move(aabbMin), std::move(aabbMax) };
}


// Rodrigues' rotation formula - `ax` is a unit vector
CARL_HD CARL_FI Vec3 rotateAroundAxis(Vec3 v, Vec3 ax, float a)
{
    return v * cosf(a) + ax.cross(v) * sinf(a) + ax * (ax.dot(v) * (1.f - cosf(a)));
}
