#pragma once

/* Collision routines below contain altered CUDA adaptations of Bullet's
   sphere/triangle, internal-edge, and manifold algorithms.

   Bullet Continuous Collision Detection and Physics Library
   Copyright (c) 2003-2009 Erwin Coumans http://bulletphysics.org

   This software is provided 'as-is', without any express or implied warranty.
   In no event will the authors be held liable for any damages arising from the
   use of this software. Permission is granted to anyone to use this software
   for any purpose, including commercial applications, and to alter it and
   redistribute it freely, subject to the following restrictions:
   1. The origin of this software must not be misrepresented; you must not claim
      that you wrote the original software. If you use this software in a
      product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
*/

#include "Arena/ArenaMesh.cuh"
#include "Cuda/Math.cuh"
#include "RLConstants.cuh"
#include <cassert>

constexpr int MAX_BALL_ARENA_CONTACTS = 68; // 16 native meshes x 4, plus planes.
constexpr float BALL_ARENA_CONTACT_BREAK = 1.905f;

struct BallArenaContact
{
    Vec3 point;
    Vec3 relPos;
    Vec3 normal;
    float depth;
    int triangle = -1;
    Vec3 reductionPos;
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
    float bestDistSq = 25.f; // Bullet's 0.1 m edge-distance threshold, in UU.

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

    const float angle = angles[bestEdge];
    if (angle == 0.f)
    {
        if (faceNormal.dot(contact.normal) >= 0.f)
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
        if (faceNormal.dot(contact.normal) >= 0.f)
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

CARL_D CARL_FI void adjustNativeBallEdgeNormal(BallArenaContact &contact, const Tri &tri,
                                               const Vec3 &angles, int flags,
                                               float edgeDistance = 5.f)
{
    const Vec3 verts[3] = {tri.v0, tri.v1, tri.v2};
    int best = -1;
    float bestSq = edgeDistance * edgeDistance;
    for (int e = 0; e < 3; e++)
    {
        if (fabsf(angles[e]) >= 2.f * PI)
            continue;
        const Vec3 near = closestPointOnBallArenaEdge(contact.point, verts[e], verts[(e + 1) % 3]);
        const float dist = (contact.point - near).lenSq();
        if (dist < bestSq)
        {
            best = e;
            bestSq = dist;
        }
    }
    if (best < 0)
        return;
    const Vec3 face = (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).norm();
    const Vec3 edge = (verts[best] - verts[(best + 1) % 3]).norm();
    const float angle = angles[best];
    bool concave = angle == 0.f;
    if (!concave)
    {
        const float sign = (flags & (1 << best)) ? 1.f : -1.f;
        Vec3 other = rotateAroundAxis(face, edge, angle);
        if (flags & (1 << (best + 3)))
            other = other.neg();
        concave = contact.normal.dot(face * sign) < 0.f && contact.normal.dot(other * sign) < 0.f;
        if (!concave)
        {
            Vec3 clamped;
            if (clampBallArenaNormal(edge, face * sign, contact.normal, angle, clamped) &&
                clamped.dot(face) > 0.f)
                contact.normal = clamped;
        }
    }
    if (concave && face.dot(contact.normal) >= 0.f)
        contact.normal = face;
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

    const float contactRadius = radius + BALL_ARENA_CONTACT_BREAK;
    if (distSq >= contactRadius * contactRadius)
        return false;

    Vec3 normal = (tri.v1 - tri.v0).cross(tri.v2 - tri.v0);
    const float normalLenSq = normal.lenSq();
    
    if (normalLenSq < 1e-12f) return false;

    normal = normal * rsqrtf(normalLenSq);
    if (distSq > 1e-8f && normal.dot(delta) < 0.f)
    {
        normal = normal.neg();
    }
    if (distSq > 1e-8f)
        normal = delta * rsqrtf(distSq);

    contact.point = point;
    contact.relPos = normal * -radius;
    contact.normal = normal;
    contact.depth = distSq > 1e-8f
        ? radius - sqrtf(distSq)
        : radius;

    return true;
}

CARL_D CARL_FI bool nativeSphereTriangleContact(BallArenaContact &c, const Vec3 &center,
                                                const Tri &t)
{
    constexpr float radius = BALL_COLLISION_RADIUS * .02f;
    constexpr float extent = radius + BALL_ARENA_CONTACT_BREAK * .02f;
    const Vec3 u = t.v1 - t.v0, v = t.v2 - t.v0, w = center - t.v0, n = u.cross(v);
    const float n2 = n.lenSq();
    if (n2 < 1e-14f)
        return false;
    Vec3 normal = n * (1.f / sqrtf(n2));
    float plane = w.dot(normal);
    if (plane < 0.f)
    {
        plane = -plane;
        normal = normal.neg();
    }
    if (plane >= extent)
        return false;
    const float gamma = u.cross(w).dot(n) / n2, beta = w.cross(v).dot(n) / n2,
                alpha = 1.f - gamma - beta;
    const bool inside =
        alpha >= 0.f && alpha <= 1.f && beta >= 0.f && beta <= 1.f && gamma >= 0.f && gamma <= 1.f;
    c.point = inside ? center - normal * plane : closestPointOnTriangle(center, t.v0, t.v1, t.v2);
    const Vec3 delta = center - c.point;
    const float d2 = delta.lenSq();
    if (d2 >= extent * extent)
        return false;
    const float distance = sqrtf(d2);
    c.normal = d2 > 1.1920929e-7f ? delta * (1.f / distance) : normal;
    c.depth = d2 > 1.1920929e-7f ? radius - distance : radius;
    c.relPos = (c.point - c.normal * c.depth) - center;
    c.reductionPos = c.relPos;
    return true;
}

CARL_D CARL_FI int ballContactReplacement(const BallArenaContact *p, const BallArenaContact &c)
{
    int deepest = -1;
    float depth = c.depth;
    for (int i = 0; i < 4; i++)
        if (p[i].depth > depth)
        {
            deepest = i;
            depth = p[i].depth;
        }
    float area[4] = {0, 0, 0, 0};
    const Vec3 x = c.reductionPos;
    if (deepest != 0)
        area[0] = (x - p[1].reductionPos).cross(p[3].reductionPos - p[2].reductionPos).lenSq();
    if (deepest != 1)
        area[1] = (x - p[0].reductionPos).cross(p[3].reductionPos - p[2].reductionPos).lenSq();
    if (deepest != 2)
        area[2] = (x - p[0].reductionPos).cross(p[3].reductionPos - p[1].reductionPos).lenSq();
    if (deepest != 3)
        area[3] = (x - p[0].reductionPos).cross(p[2].reductionPos - p[1].reductionPos).lenSq();
    int best = 0;
    for (int i = 1; i < 4; i++)
        if (area[i] > area[best])
            best = i;
    return best;
}

CARL_D CARL_FI int gatherBallArenaContacts(
    BallArenaContact* contacts,
    const Vec3& center,
    ArenaMesh* arena)
{
    constexpr float radius = BALL_COLLISION_RADIUS;
    const Vec3 extent = Vec3::fill(radius + BALL_ARENA_CONTACT_BREAK);
    const Int3 cellLo = arena->aabbToCell3D(center - extent);
    const Int3 cellHi = arena->aabbToCell3D(center + extent);

    constexpr int maxCells = 27;
    static_assert(2.f * (BALL_COLLISION_RADIUS + BALL_ARENA_CONTACT_BREAK) <= 2.f * CELL_SIZE.x &&
                  2.f * (BALL_COLLISION_RADIUS + BALL_ARENA_CONTACT_BREAK) <= 2.f * CELL_SIZE.y &&
                  2.f * (BALL_COLLISION_RADIUS + BALL_ARENA_CONTACT_BREAK) <= 2.f * CELL_SIZE.z);
    int cursor[maxCells], end[maxCells], cells = 0;
    for (int z = cellLo.z; z <= cellHi.z; z++)
    for (int y = cellLo.y; y <= cellHi.y; y++)
    for (int x = cellLo.x; x <= cellHi.x; x++)
    {
        const Int3 cell = { x, y, z };
        const int cellIdx = arena->cell3DToFlatIdx(cell);
        cursor[cells] = arena->prefix[cellIdx];
        end[cells] = arena->prefix[cellIdx + 1];
        cells++;
    }
    int count = 0, mesh = -1, base = 0, localCount = 0, lastTri = -1;
    while (true)
    {
        int selected = -1, triIdx = -1, key = 2147483647;
        for (int cell = 0; cell < cells; cell++)
        {
            if (cursor[cell] >= end[cell])
                continue;
            const int tri = arena->ballTriIdx[cursor[cell]];
            const int candidate = arena->nativeMesh[tri] * 10000 + arena->nativeOrder[tri];
            if (candidate < key)
            {
                key = candidate;
                selected = cell;
                triIdx = tri;
            }
        }
        if (selected < 0)
            break;
        cursor[selected]++;
        if (triIdx == lastTri)
            continue;
        lastTri = triIdx;
        const int current = arena->nativeMesh[triIdx];
        if (arena->hasNativeMetadata && current < 0)
            continue;
        BallArenaContact contact;
        if (arena->hasNativeMetadata)
        {
            const Tri tri = {arena->nativeV0[triIdx], arena->nativeV1[triIdx],
                             arena->nativeV2[triIdx]};
            if (!nativeSphereTriangleContact(contact, center * .02f, tri))
                continue;
        }
        else
        {
            if (!sphereTriangleContact(contact, center, radius, arena->ldg(triIdx)))
                continue;
            contact.reductionPos = contact.relPos;
        }
        contact.triangle = triIdx;
        if (current != mesh)
        {
            base = count;
            localCount = 0;
            mesh = current;
        }
        int slot;
        if (localCount < 4)
        {
            slot = localCount++;
            count++;
        }
        else
            slot = ballContactReplacement(contacts + base, contact);
        if (arena->hasNativeMetadata)
        {
            const Tri tri = {arena->nativeV0[triIdx], arena->nativeV1[triIdx],
                             arena->nativeV2[triIdx]};
            adjustNativeBallEdgeNormal(contact, tri, arena->nativeEdgeAngles[triIdx],
                                       arena->nativeEdgeFlags[triIdx], .1f);
            contact.point = center * .02f + contact.relPos + contact.normal * contact.depth;
        }
        else
        {
            adjustBallArenaEdgeNormal(contact, contact.point, arena->ldg(triIdx),
                                      arena->triAngs[triIdx]);
            contact.point = center + contact.relPos + contact.normal * contact.depth;
        }
        contacts[base + slot] = contact;
    }
    if (arena->hasNativeMetadata)
    {
        for (int i = 0; i < count; i++)
        {
            contacts[i].point = contacts[i].point * 50.f;
            contacts[i].relPos = contacts[i].relPos * 50.f;
            contacts[i].depth *= 50.f;
        }
        const Vec3 normals[4] = {{0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}};
        const float offsets[4] = {0.f, -2044.f, -4096.f, -4096.f};
        for (int p = 0; p < 4; p++)
        {
            const Vec3 normal = normals[p];
            const float depth = radius - (normal.dot(center) - offsets[p]);
            if (depth > -BALL_ARENA_CONTACT_BREAK)
            {
                assert(count < MAX_BALL_ARENA_CONTACTS);
                contacts[count++] = {center - normal * (radius - depth), normal * -radius, normal,
                                     depth};
            }
        }
    }

    return count;
}
