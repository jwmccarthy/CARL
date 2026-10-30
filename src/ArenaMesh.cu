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

    std::vector<Vec3> nativeAngles = hAngles;
    std::vector<int> flags(nTris), mesh(nTris, 0), order(nTris);
    std::vector<Vec3> nativeVerts[3];
    for (int j = 0; j < 3; j++)
    {
        nativeVerts[j].resize(nTris);
        for (int i = 0; i < nTris; i++)
            nativeVerts[j][i] = hVerts[triangleVertex(hTris[i], j)] * .02f;
    }
    for (int i = 0; i < nTris; i++)
        order[i] = i;
    if (const char *path = std::getenv("CARL_ARENA_METADATA"))
    {
        std::ifstream file(path);
        int count = 0;
        file >> count;
        if (count != nTris)
            throw std::runtime_error("CARL arena metadata triangle count mismatch");
        for (int i = 0; i < nTris; i++)
        {
            auto &a = nativeAngles[i];
            file >> mesh[i] >> order[i] >> a.x >> a.y >> a.z >> flags[i];
            for (int j = 0; j < 3; j++)
            {
                auto &v = nativeVerts[j][i];
                file >> v.x >> v.y >> v.z;
            }
        }
        if (!file)
            throw std::runtime_error("Invalid CARL arena metadata");
        hasNativeMetadata = true;
    }
    nativeEdgeAngles = DeviceArray(nativeAngles);
    nativeEdgeFlags = DeviceArray(flags);
    nativeMesh = DeviceArray(mesh);
    nativeOrder = DeviceArray(order);
    nativeV0 = DeviceArray(nativeVerts[0]);
    nativeV1 = DeviceArray(nativeVerts[1]);
    nativeV2 = DeviceArray(nativeVerts[2]);
    std::vector<int> ballIndices = hTriIdx;
    for (int cell = 0; cell < nCells; cell++)
    {
        std::sort(
            ballIndices.begin() + hPrefix[cell], ballIndices.begin() + hPrefix[cell + 1],
            [&](int a, int b) { return mesh[a] * 10000 + order[a] < mesh[b] * 10000 + order[b]; });
    }
    ballTriIdx = DeviceArray(ballIndices);
}
