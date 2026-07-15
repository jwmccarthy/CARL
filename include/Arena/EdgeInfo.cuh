#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "TriMesh.cuh"

#include "../Cuda/Math.cuh"

struct TriangleEdge
{
    int triangle;
    int slot;
    int start;
    int end;
};

struct EdgeAdjacency
{
    TriangleEdge first;
    bool paired;
};

using EdgeMap = std::map<std::pair<int, int>, EdgeAdjacency>;

inline int triangleVertex(const Int3& triangle, int slot)
{
    if (slot == 0) return triangle.x;
    if (slot == 1) return triangle.y;
    return triangle.z;
}

inline TriangleEdge makeTriangleEdge(
    const Int3& triangle,
    int triangleIdx,
    int slot)
{
    return {
        triangleIdx,
        slot,
        triangleVertex(triangle, slot),
        triangleVertex(triangle, (slot + 1) % 3)
    };
}

inline std::optional<TriangleEdge> registerEdge(
    EdgeMap& edges,
    const TriangleEdge& edge)
{
    const auto [lo, hi] = std::minmax(edge.start, edge.end);
    auto [entry, inserted] = edges.try_emplace(
        std::pair{ lo, hi }, EdgeAdjacency{ edge, false });

    if (inserted) return std::nullopt;
    if (entry->second.paired)
        throw std::runtime_error("non-manifold edge in arena mesh");

    entry->second.paired = true;
    return entry->second.first;
}

inline void validateWinding(
    const TriangleEdge& first,
    const TriangleEdge& second)
{
    if (first.start != second.end || first.end != second.start)
        throw std::runtime_error("inconsistent triangle winding in arena mesh");
}

inline float signedEdgeAngle(
    const Vec3& start,
    const Vec3& end,
    const Vec3& normalA,
    const Vec3& normalB)
{
    const Vec3 edgeVector = end - start;
    const float edgeLenSq = edgeVector.lenSq();
    if (edgeLenSq <= 1e-12f)
        throw std::runtime_error("degenerate edge in arena mesh");

    const Vec3 edge = edgeVector * (1.f / sqrtf(edgeLenSq));
    const Vec3 normalCross = normalA.cross(normalB);
    const float normalDot = std::clamp(normalA.dot(normalB), -1.f, 1.f);

    if (normalCross.lenSq() <= 1e-4f)
    {
        if (normalDot < 0.f)
            throw std::runtime_error("opposing triangle normals in arena mesh");
        return 0.f;
    }

    return atan2f(edge.dot(normalCross), normalDot);
}

inline std::vector<Vec3> buildEdgeAngles(
    const std::vector<Vec3>& verts,
    const std::vector<Int3>& tris,
    const std::vector<Vec3>& norms)
{
    std::vector<Vec3> angles(tris.size(), Vec3::fill(BOUNDARY_EDGE_ANGLE));
    EdgeMap edges;

    for (int triangle = 0; triangle < (int)tris.size(); triangle++)
    for (int slot = 0; slot < 3; slot++)
    {
        const TriangleEdge edge = makeTriangleEdge(tris[triangle], triangle, slot);
        const std::optional<TriangleEdge> other = registerEdge(edges, edge);
        if (!other) continue;

        validateWinding(*other, edge);
        const float angle = signedEdgeAngle(
            verts[other->start], verts[other->end],
            norms[other->triangle], norms[edge.triangle]);

        angles[other->triangle][other->slot] = angle;
        angles[triangle][slot] = angle;
    }

    return angles;
}
