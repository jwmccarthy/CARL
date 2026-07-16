#include "Cuda/DeviceArray.cuh"

#include "Arena/ArenaMesh.cuh"
#include "Arena/ObjLoader.cuh"
#include "Arena/TriMesh.cuh"
#include "Arena/BroadPhaseGrid.cuh"
#include "Arena/EdgeInfo.cuh"

ArenaMesh::ArenaMesh()
{
    static const auto geometry  = parseObj();
    const auto& [hVerts, hTris] = geometry;

    static const auto hNorms = triNormals(hVerts, hTris);
    static const auto bounds = triBounds(hVerts, hTris);
    const auto& [hAabbMin, hAabbMax] = bounds;

    static const auto hCellLow = triCellLows(hAabbMin);
    static const auto grid = buildGrid(hCellLow, hAabbMax);
    const auto& [hTriIdx, hPrefix] = grid;

    static const auto hAngles = buildEdgeAngles(hVerts, hTris, hNorms);

    nTris  = (int)hTris.size();
    nVerts = (int)hVerts.size();
    nCells = (int)GRID_DIMS.prod();

    verts = DeviceArray(hVerts);
    norms = DeviceArray(hNorms);
    tris  = DeviceArray(hTris);

    aabbMin = DeviceArray(hAabbMin);
    aabbMax = DeviceArray(hAabbMax);
    cellLow = DeviceArray(hCellLow);

    triIdx = DeviceArray(hTriIdx);
    prefix = DeviceArray(hPrefix);

    triAngs = DeviceArray(hAngles);
}
