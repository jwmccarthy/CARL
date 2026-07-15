#pragma once

#include "../../RLConstants.cuh"
#include "../../Cuda/Math.cuh"
#include "../../State/Workspace.cuh"
#include "../../Arena/ArenaMesh.cuh"
#include "../../DataUtils.cuh"
#include "../Collision/SAT.cuh"
#include "Triangle.cuh"

struct TriSAT : LocalTriangle
{
    Vec3 delta;
};

CARL_D CARL_FI TriSAT loadSATContext(
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena,
    const int carIdx,
    const int offset,
    const Vec3& cen,
    const Quat& rot)
{
    TriSAT tri;

    const int cellIdx  = __ldg(&space->bp.cellIdx[carIdx]);
    const int triStart = __ldg(&arena->prefix[cellIdx]);
    const int triIdx   = __ldg(&arena->triIdx[triStart + offset]);

    auto [i0, i1, i2, _] = Int3::ldg(arena->tris[triIdx]);
    tri.set(
        triIdx,
        rot.toLocal(Vec3::ldg(arena->verts[i0]) - cen),
        rot.toLocal(Vec3::ldg(arena->verts[i1]) - cen),
        rot.toLocal(Vec3::ldg(arena->verts[i2]) - cen));

    tri.delta = tri.center().neg();

    return tri;
}

CARL_D CARL_FI AxisProj projectCarRadius(const Vec3& ax)
{
    float r = ax.abs().dot(CAR_HALF_EX);

    return { -r, r };
}

CARL_D CARL_FI AxisProj projectTriRadius(const Vec3& ax, const TriSAT& tri)
{
    float d0 = tri.v0.dot(ax);
    float d1 = tri.v1.dot(ax);
    float d2 = tri.v2.dot(ax);

    return {
        fminf(fminf(d0, d1), d2),
        fmaxf(fmaxf(d0, d1), d2)
    };
}

CARL_D CARL_FI void testCarTriAxis(
    Vec3 ax,
    const TriSAT& tri,
    const int idx,
    const bool norm,
    SATResult& res)
{
    if (!res.overlap) return;

    float lenSq = ax.lenSq();
    if (lenSq < 1e-12f) return;

    if (norm) ax = ax * rsqrtf(lenSq);

    ax = ax * copysignf(1.f, ax.dot(tri.delta));
    
    auto [cMin, cMax] = projectCarRadius(ax);
    auto [tMin, tMax] = projectTriRadius(ax, tri);

    if (cMax < tMin || tMax < cMin)
    {
        res.overlap = false;
        return;
    }

    float depth = fminf(cMax - tMin, tMax - cMin);

    if (depth < res.minPen)
    {
        res.minPen = depth;
        res.minAxis = ax;
        res.axisIdx = idx;
    }
}

CARL_D CARL_FI SATResult carTriSAT(const TriSAT& tri)
{
    SATResult res;

    testCarTriAxis(tri.e0.cross(tri.e1), tri, 0, true, res);

    if (!res.overlap) return res;  // Early out on likely tri-normal separating axis

    testCarTriAxis(WORLD_X, tri, 1, false, res);
    testCarTriAxis(WORLD_Y, tri, 2, false, res);
    testCarTriAxis(WORLD_Z, tri, 3, false, res);

    testCarTriAxis(tri.e0.cross(WORLD_X), tri, 4, true, res);
    testCarTriAxis(tri.e1.cross(WORLD_X), tri, 5, true, res);
    testCarTriAxis(tri.e2.cross(WORLD_X), tri, 6, true, res);

    testCarTriAxis(tri.e0.cross(WORLD_Y), tri, 7, true, res);
    testCarTriAxis(tri.e1.cross(WORLD_Y), tri, 8, true, res);
    testCarTriAxis(tri.e2.cross(WORLD_Y), tri, 9, true, res);

    testCarTriAxis(tri.e0.cross(WORLD_Z), tri, 10, true, res);
    testCarTriAxis(tri.e1.cross(WORLD_Z), tri, 11, true, res);
    testCarTriAxis(tri.e2.cross(WORLD_Z), tri, 12, true, res);

    return res;
}

CARL_D CARL_FI void writeCarTriPairResult(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena,
    const int carIdx,
    const int offset)
{
    const CarPose pose = loadCarPose(state, carIdx);

    const TriSAT tri = loadSATContext(
        space, arena, carIdx, offset, pose.cen, pose.rot);
    const SATResult sat = carTriSAT(tri);

    if (!sat.overlap) return;

    const int slot = atomicAdd(&space->ctHit.carHitCount[carIdx], 1);
    if (slot >= MAX_CAR_TRI_PAIRS) return;

    const int pairIdx = carIdx * MAX_CAR_TRI_PAIRS + slot;

    space->ctNrw.pairV0[pairIdx] = tri.v0;
    space->ctNrw.pairV1[pairIdx] = tri.v1;
    space->ctNrw.pairV2[pairIdx] = tri.v2;

    space->ctNrw.triIdx[pairIdx] = tri.triIdx;

    space->ctNrw.axisIdx[pairIdx] = sat.axisIdx;
    space->ctNrw.minAxis[pairIdx] = sat.minAxis;
    space->ctNrw.minPen[pairIdx] = sat.minPen;
}
