#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

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

    carTriIdx = DeviceArray(hTriIdx);
    prefix = DeviceArray(hPrefix);

    triAngs = DeviceArray(hAngles);

    std::vector<Vec3> nativeAngles = hAngles;
    std::vector<int> flags(nTris);
    std::vector<int> mesh(nTris, 0);
    std::vector<int> order(nTris);
    std::vector<Vec3> nativeVerts[3];

    // Initialize native metadata from the existing arena geometry
    for (int j = 0; j < 3; j++)
    {
        nativeVerts[j].resize(nTris);

        for (int i = 0; i < nTris; i++)
        {
            nativeVerts[j][i] = hVerts[triangleVertex(hTris[i], j)] * .02f;
        }
    }

    for (int i = 0; i < nTris; i++)
    {
        order[i] = i;
    }

    // Override the defaults when native mesh metadata is available
    if (const char* path = std::getenv("CARL_ARENA_METADATA"))
    {
        std::ifstream file(path);
        int count = 0;
        file >> count;

        if (count != nTris)
        {
            throw std::runtime_error("CARL arena metadata triangle count mismatch");
        }

        for (int i = 0; i < nTris; i++)
        {
            auto& angles = nativeAngles[i];
            file >> mesh[i] >> order[i] >> angles.x >> angles.y >> angles.z >> flags[i];

            for (int j = 0; j < 3; j++)
            {
                auto& vertex = nativeVerts[j][i];
                file >> vertex.x >> vertex.y >> vertex.z;
            }
        }

        if (!file)
        {
            throw std::runtime_error("Invalid CARL arena metadata");
        }

        hasNativeMetadata = true;
    }

    nativeEdgeAngles = DeviceArray(nativeAngles);
    nativeEdgeFlags = DeviceArray(flags);
    nativeMesh = DeviceArray(mesh);
    nativeOrder = DeviceArray(order);
    nativeV0 = DeviceArray(nativeVerts[0]);
    nativeV1 = DeviceArray(nativeVerts[1]);
    nativeV2 = DeviceArray(nativeVerts[2]);

    // Order ball triangles by native mesh and traversal position within each cell
    std::vector<int> ballIndices = hTriIdx;
    const auto nativeOrderLess = [&](int a, int b)
    {
        return mesh[a] * 10000 + order[a] < mesh[b] * 10000 + order[b];
    };

    for (int cell = 0; cell < nCells; cell++)
    {
        std::sort(
            ballIndices.begin() + hPrefix[cell],
            ballIndices.begin() + hPrefix[cell + 1],
            nativeOrderLess);
    }

    ballTriIdx = DeviceArray(ballIndices);
}
