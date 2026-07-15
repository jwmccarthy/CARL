#pragma once

#include "../AABB.cuh"
#include "TriMesh.cuh"
#include "BroadPhaseGrid.cuh"

#include "../Cuda/Math.cuh"
#include "../Cuda/Common.cuh"
#include "../Cuda/DeviceArray.cuh"

struct ArenaMesh
{
    int nTris;
    int nVerts;
    int nCells;

    DeviceArray<Vec3> verts;
    DeviceArray<Vec3> norms;
    DeviceArray<Int3> tris;

    DeviceArray<Vec3> aabbMin;
    DeviceArray<Vec3> aabbMax;
    DeviceArray<Int3> cellLow;
    DeviceArray<Vec3> triAngs;
    DeviceArray<int>  triIdx;
    DeviceArray<int>  prefix;

    ArenaMesh();

    CARL_D CARL_FI int numTrisInCell(int c) const
    {
        return prefix[c + 1] - prefix[c];
    }

    CARL_D CARL_FI AABB getTriAABB(int tri) const
    {
        return {
            Vec3::ldg(aabbMin[tri]),
            Vec3::ldg(aabbMax[tri])
        };
    }

    CARL_HD CARL_FI Int3 aabbToCell3D(const Vec3& p)
    {
        return ((p - ARENA_MIN) / CELL_SIZE)
            .max(0.f)
            .min(GRID_DIMS - 1.f)
            .toInt3();
    }

    CARL_HD CARL_FI int cell3DToFlatIdx(const Int3& c)
    {
        constexpr int WIDTH = (int)GRID_DIMS.x;
        constexpr int CROSS = WIDTH * (int)GRID_DIMS.y;

        return c.x + c.y * WIDTH + c.z * CROSS;
    }

    CARL_D CARL_FI Tri ldg(int tri) const
    {
        const Int3 idx = tris[tri];
        return {
            Vec3::ldg(verts[idx.x]),
            Vec3::ldg(verts[idx.y]),
            Vec3::ldg(verts[idx.z])
        };
    }
};
