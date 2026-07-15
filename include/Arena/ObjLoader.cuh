#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../Cuda/Math.cuh"

inline std::ifstream openArenaMeshObj()
{
    const std::string sourceFile = __FILE__;
    const std::string sourceMarker = "/include/";
    const size_t markerPos = sourceFile.rfind(sourceMarker);

    const std::string sourceTreePath = markerPos == std::string::npos
        ? std::string{}
        : sourceFile.substr(0, markerPos) + "/assets/arena.obj";

    const std::string candidates[] = {
        "assets/arena.obj",
        "../assets/arena.obj",
        sourceTreePath,
    };

    for (const std::string& candidate : candidates)
    {
        if (candidate.empty()) continue;

        std::ifstream file(candidate);
        if (file.is_open())
        {
            return file;
        }
    }

    throw std::runtime_error("failed to open assets/arena.obj");
}

inline void weldExactVertices(
    std::vector<Vec3>& verts,
    std::vector<Int3>& tris)
{
    std::vector<Vec3> welded;
    std::vector<int> remap(verts.size());
    std::map<std::array<uint32_t, 3>, int> vertexIds;

    welded.reserve(verts.size());

    for (size_t i = 0; i < verts.size(); i++)
    {
        const Vec3& v = verts[i];

        const auto bits = [](float value)
        {
            return std::bit_cast<uint32_t>(value == 0.f ? 0.f : value);
        };
        
        const std::array key = { bits(v.x), bits(v.y), bits(v.z) };
        const int nextId = (int)welded.size();
        const auto [entry, inserted] = vertexIds.emplace(key, nextId);

        if (inserted) welded.push_back(v);
        remap[i] = entry->second;
    }

    for (Int3& tri : tris)
    {
        if (tri.x < 0 || tri.x >= (int)remap.size() ||
            tri.y < 0 || tri.y >= (int)remap.size() ||
            tri.z < 0 || tri.z >= (int)remap.size())
        {
            throw std::runtime_error("OBJ triangle index is out of range");
        }

        tri = { remap[tri.x], remap[tri.y], remap[tri.z] };
    }

    verts = std::move(welded);
}

inline std::pair<std::vector<Vec3>, std::vector<Int3>> parseObj()
{
    std::vector<Vec3> verts;
    std::vector<Int3> tris;

    std::string line;
    std::ifstream file = openArenaMeshObj();

    while (std::getline(file, line))
    {
        std::string type;
        std::istringstream s(line);
        if (!(s >> type)) continue;

        if (type == "v")
        {
            // The source .obj uses a swapped X/Y axis convention
            float x, y, z;
            if (!(s >> y >> x >> z))
                throw std::runtime_error("invalid OBJ vertex");
            verts.push_back({ x, y, z });
        }
        else if (type == "f")
        {
            // .obj face indices are 1-based
            int x, y, z;
            if (!(s >> x >> y >> z))
                throw std::runtime_error("invalid OBJ triangle");
            tris.push_back({ x - 1, y - 1, z - 1 });
        }
    }

    weldExactVertices(verts, tris);

    return { std::move(verts), std::move(tris) };
}
