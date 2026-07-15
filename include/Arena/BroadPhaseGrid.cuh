#pragma once

#include <vector>
#include <utility>
#include <algorithm>

#include "../Cuda/Math.cuh"
#include "../Cuda/Common.cuh"
#include "../RLConstants.cuh"

constexpr CARL_HD Vec3 GRID_DIMS = { 48.f, 48.f, 10.f };
constexpr CARL_HD Vec3 CELL_SIZE = (ARENA_MAX - ARENA_MIN) / GRID_DIMS;

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

inline std::vector<Int3> triCellLows(const std::vector<Vec3>& aabbMin)
{
    std::vector<Int3> lows;
    lows.reserve(aabbMin.size());

    for (const Vec3& lo : aabbMin)
    {
        lows.push_back(aabbToCell3D(lo).sub(1).max(0));
    }

    return lows;
}

inline std::pair<std::vector<int>, std::vector<int>> buildGrid(
    const std::vector<Int3>& cellLow,
    const std::vector<Vec3>& aabbMax)
{
    const int nTris  = (int)cellLow.size();
    const int nCells = (int)GRID_DIMS.prod();

    std::vector<std::vector<int>> cells(nCells);

    for (int i = 0; i < nTris; i++)
    {
        const Int3 lo = cellLow[i];
        const Int3 hi = aabbToCell3D(aabbMax[i]);

        for (int x = lo.x; x <= hi.x; x++)
        for (int y = lo.y; y <= hi.y; y++)
        for (int z = lo.z; z <= hi.z; z++)
        {
            cells[cell3DToFlatIdx({ x, y, z })].push_back(i);
        }
    }

    std::vector<int> prefix(nCells + 1, 0);
    for (int c = 0; c < nCells; c++)
    {
        prefix[c + 1] = prefix[c] + (int)cells[c].size();
    }

    std::vector<int> triIdx(prefix.back());
    for (int c = 0; c < nCells; c++)
    {
        std::copy(cells[c].begin(), cells[c].end(), triIdx.begin() + prefix[c]);
    }

    return { std::move(triIdx), std::move(prefix) };
}
