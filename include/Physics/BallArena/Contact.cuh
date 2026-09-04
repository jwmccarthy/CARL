#pragma once

#include "Arena/ArenaMesh.cuh"
#include "Cuda/Math.cuh"
#include "RLConstants.cuh"

struct BallArenaContact
{
    Vec3 point;
    Vec3 relPos;
    Vec3 normal;
    float depth;
};

CARL_D CARL_FI Vec3 closestPointOnBallArenaEdge(
    const Vec3& point,
    const Vec3& start,
    const Vec3& end)
{
    const Vec3 edge = end - start;
    const float t = clampf((point - start).dot(edge) / edge.lenSq(), 0.f, 1.f);
    return start + edge * t;
}

CARL_D CARL_FI bool clampBallArenaNormal(
    const Vec3& edgeAxis,
    const Vec3& faceNormal,
    const Vec3& contactNormal,
    float edgeAngle,
    Vec3& clampedNormal)
{
    Vec3 edgeCross = edgeAxis.cross(faceNormal);
    const float crossLenSq = edgeCross.lenSq();
    if (crossLenSq <= 1e-12f) return false;

    edgeCross = edgeCross * rsqrtf(crossLenSq);
    const float currentAngle = atan2f(
        contactNormal.dot(edgeCross),
        contactNormal.dot(faceNormal));
    const bool outside = edgeAngle < 0.f
        ? currentAngle < edgeAngle
        : currentAngle > edgeAngle;
    if (!outside) return false;

    clampedNormal = rotateAroundAxis(
        contactNormal, edgeAxis, edgeAngle - currentAngle);
    return true;
}

CARL_D CARL_FI void adjustBallArenaEdgeNormal(
    BallArenaContact& contact,
    const Vec3& point,
    const Tri& tri,
    const Vec3& angles)
{
    int bestEdge = -1;
    float bestDistSq = 100.f;

    const Vec3 verts[3] = { tri.v0, tri.v1, tri.v2 };
    for (int edge = 0; edge < 3; edge++)
    {
        if (fabsf(angles[edge]) >= BOUNDARY_EDGE_ANGLE) continue;
        const Vec3 nearest = closestPointOnBallArenaEdge(
            point, verts[edge], verts[(edge + 1) % 3]);
        const float distSq = (point - nearest).lenSq();
        if (distSq < bestDistSq)
        {
            bestDistSq = distSq;
            bestEdge = edge;
        }
    }

    if (bestEdge < 0) return;

    Vec3 faceNormal = (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).norm();
    if (faceNormal.dot(contact.normal) < 0.f) faceNormal = faceNormal.neg();

    const float angle = angles[bestEdge];
    if (angle == 0.f)
    {
        contact.normal = faceNormal;
        return;
    }

    const float side = angle > 0.f ? 1.f : -1.f;
    const Vec3 edgeAxis = (verts[(bestEdge + 1) % 3] - verts[bestEdge]).norm();
    const Vec3 normalA = faceNormal * side;
    const Vec3 normalB = rotateAroundAxis(faceNormal, edgeAxis, angle) * side;

    if (contact.normal.dot(normalA) < 0.f
        && contact.normal.dot(normalB) < 0.f)
    {
        contact.normal = faceNormal;
        return;
    }

    Vec3 clamped;
    if (clampBallArenaNormal(
        edgeAxis, normalA, contact.normal, angle, clamped)
        && clamped.dot(faceNormal) > 0.f)
    {
        contact.normal = clamped.norm();
    }
}

CARL_D CARL_FI Vec3 closestPointOnTriangle(
    const Vec3& p,
    const Vec3& a,
    const Vec3& b,
    const Vec3& c)
{
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    const float d1 = ab.dot(ap);
    const float d2 = ac.dot(ap);
    if (d1 <= 0.f && d2 <= 0.f) return a;

    const Vec3 bp = p - b;
    const float d3 = ab.dot(bp);
    const float d4 = ac.dot(bp);
    if (d3 >= 0.f && d4 <= d3) return b;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
    {
        return a + ab * (d1 / (d1 - d3));
    }

    const Vec3 cp = p - c;
    const float d5 = ab.dot(cp);
    const float d6 = ac.dot(cp);
    if (d6 >= 0.f && d5 <= d6) return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
    {
        return a + ac * (d2 / (d2 - d6));
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && d4 >= d3 && d5 >= d6)
    {
        const float edge = d4 - d3;
        return b + (c - b) * (edge / (edge + d5 - d6));
    }

    const float inv = 1.f / (va + vb + vc);
    return a + ab * (vb * inv) + ac * (vc * inv);
}

CARL_D CARL_FI bool sphereTriangleContact(
    BallArenaContact& contact,
    const Vec3& center,
    float radius,
    const Tri& tri)
{
    const Vec3 point = closestPointOnTriangle(
        center, tri.v0, tri.v1, tri.v2);
    const Vec3 delta = center - point;
    const float distSq = delta.lenSq();
    if (distSq >= radius * radius) return false;

    // Use face normals by default to avoid tessellation-edge contacts.
    Vec3 normal = (tri.v1 - tri.v0).cross(tri.v2 - tri.v0);
    const float normalLenSq = normal.lenSq();
    if (normalLenSq < 1e-12f) return false;

    normal = normal * rsqrtf(normalLenSq);
    if (distSq > 1e-8f && normal.dot(delta) < 0.f)
    {
        normal = normal.neg();
    }

    contact.point = point;
    contact.relPos = normal * -BALL_COLLISION_RADIUS;
    contact.normal = normal;
    contact.depth = distSq > 1e-8f
        ? radius - sqrtf(distSq)
        : radius;

    return true;
}

CARL_D CARL_FI int gatherBallArenaContacts(
    BallArenaContact* contacts,
    const Vec3& center,
    ArenaMesh* arena)
{
    constexpr float radius = BALL_REST_Z;
    const Vec3 extent = Vec3::fill(radius);
    const Int3 cellLo = arena->aabbToCell3D(center - extent);
    const Int3 cellHi = arena->aabbToCell3D(center + extent);

    int count = 0;
    int rawCount = 0;
    Vec3 rawRelPosSum = Vec3::zero();
    Vec3 rawNormalSum = Vec3::zero();
    float rawDepthSum = 0.f;
    const bool curvedCorner = fabsf(center.x) > 3000.f
        && fabsf(center.y) > 3000.f;
    const bool backWallSeam = fabsf(center.y) > 4900.f;

    for (int z = cellLo.z; z <= cellHi.z; z++)
    for (int y = cellLo.y; y <= cellHi.y; y++)
    for (int x = cellLo.x; x <= cellHi.x; x++)
    {
        const Int3 cell = { x, y, z };
        const int cellIdx = arena->cell3DToFlatIdx(cell);
        const int begin = __ldg(&arena->prefix[cellIdx]);
        const int end = __ldg(&arena->prefix[cellIdx + 1]);

        for (int i = begin; i < end; i++)
        {
            const int triIdx = __ldg(&arena->triIdx[i]);
            const Int3 owner = Int3::ldg(arena->cellLow[triIdx]).max(cellLo);
            if (owner.x != x || owner.y != y || owner.z != z) continue;

            BallArenaContact contact;

            const Tri tri = arena->ldg(triIdx);
            if (!sphereTriangleContact(contact, center, radius, tri)) continue;

            if (backWallSeam)
            {
                const Vec3 delta = center - contact.point;
                contact.normal = delta.norm();
                contact.relPos = contact.normal * -radius;
                adjustBallArenaEdgeNormal(
                    contact, contact.point,
                    tri, Vec3::ldg(arena->triAngs[triIdx]));
            }

            if (backWallSeam)
            {
                rawCount++;
                rawRelPosSum = rawRelPosSum + contact.relPos;
                rawNormalSum = rawNormalSum + contact.normal;
                rawDepthSum += contact.depth;
            }

            int slot = -1;

            for (int c = 0; c < count; c++)
            {
                if (contacts[c].normal.dot(contact.normal) > 0.9999f)
                {
                    slot = c;
                    break;
                }
            }

            if (slot >= 0)
            {
                if (contact.depth > contacts[slot].depth)
                {
                    contacts[slot] = contact;
                }
            }
            else if (count < 4)
            {
                contacts[count++] = contact;
            }
        }
    }

    if (count > 1 && backWallSeam)
    {
        contacts[0].relPos = rawRelPosSum * (1.f / rawCount);
        contacts[0].normal = rawNormalSum * (1.f / rawCount);
        contacts[0].depth = rawDepthSum / rawCount;
        return 1;
    }

    if (count > 1 && curvedCorner)
    {
        Vec3 normal = Vec3::zero();
        float depth = 0.f;

        for (int i = 0; i < count; i++)
        {
            normal = normal + contacts[i].normal;
            depth += contacts[i].depth;
        }

        contacts[0].normal = normal * (1.f / count);
        contacts[0].relPos = contacts[0].normal * -BALL_COLLISION_RADIUS;
        contacts[0].depth = depth / count;
        return 1;
    }

    return count;
}
