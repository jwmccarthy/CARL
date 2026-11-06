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

#include <cassert>

#include "Arena/ArenaMesh.cuh"
#include "Cuda/Math.cuh"
#include "RLConstants.cuh"

// Capacity for four manifold points per native mesh (16 meshes), plus four
// arena planes. Keep the contact and solver row arrays sized consistently.
constexpr int MAX_BALL_ARENA_CONTACTS = 68;
// Contact separation threshold in Unreal units: contacts farther apart are
// discarded and the same margin expands the broadphase query. This is an
// existing physics-port tuning value; its original derivation is undocumented.
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

// Both mesh formats select edges the same way, but use different winding and
// metadata when correcting the normal afterwards.
CARL_D CARL_FI int closestBallArenaInternalEdge(
    const Vec3& point,
    const Tri& tri,
    const Vec3& angles,
    float maxDistSq)
{
    int bestEdge = -1;
    float bestDistSq = maxDistSq;

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

    return bestEdge;
}

CARL_D CARL_FI void adjustBallArenaEdgeNormal(
    BallArenaContact& contact,
    const Vec3& point,
    const Tri& tri,
    const Vec3& angles)
{
    // Bullet's 0.1 m edge-distance threshold, in squared UU.
    const int bestEdge = closestBallArenaInternalEdge(point, tri, angles, 25.f);
    if (bestEdge < 0) return;

    const Vec3 verts[3] = { tri.v0, tri.v1, tri.v2 };
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

CARL_D CARL_FI void adjustNativeBallEdgeNormal(
    BallArenaContact& contact,
    const Tri& tri,
    const Vec3& angles,
    int flags,
    float edgeDistance = 5.f)
{
    const Vec3 verts[3] = { tri.v0, tri.v1, tri.v2 };
    const int bestEdge = closestBallArenaInternalEdge(
        contact.point, tri, angles, edgeDistance * edgeDistance);
    if (bestEdge < 0) return;

    const Vec3 faceNormal = (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).norm();
    const Vec3 edgeDir = (verts[bestEdge] - verts[(bestEdge + 1) % 3]).norm();
    const float angle = angles[bestEdge];
    bool concave = angle == 0.f;

    if (!concave)
    {
        const float sign = (flags & (1 << bestEdge)) ? 1.f : -1.f;
        Vec3 otherNormal = rotateAroundAxis(faceNormal, edgeDir, angle);
        if (flags & (1 << (bestEdge + 3))) otherNormal = otherNormal.neg();

        concave = contact.normal.dot(faceNormal * sign) < 0.f
               && contact.normal.dot(otherNormal * sign) < 0.f;

        if (!concave)
        {
            Vec3 clamped;
            if (clampBallArenaNormal(edgeDir, faceNormal * sign, contact.normal, angle, clamped)
                && clamped.dot(faceNormal) > 0.f)
            {
                contact.normal = clamped;
            }
        }
    }

    if (concave && faceNormal.dot(contact.normal) >= 0.f)
    {
        contact.normal = faceNormal;
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

    // Test vertex A, then vertex B and edge AB.
    const Vec3 ap = p - a;
    const float abAp = ab.dot(ap);
    const float acAp = ac.dot(ap);
    if (abAp <= 0.f && acAp <= 0.f) return a;

    const Vec3 bp = p - b;
    const float abBp = ab.dot(bp);
    const float acBp = ac.dot(bp);
    if (abBp >= 0.f && acBp <= abBp) return b;

    const float weightC = abAp * acBp - abBp * acAp;
    if (weightC <= 0.f && abAp >= 0.f && abBp <= 0.f)
    {
        return a + ab * (abAp / (abAp - abBp));
    }

    // Test vertex C, then edges AC and BC.
    const Vec3 cp = p - c;
    const float abCp = ab.dot(cp);
    const float acCp = ac.dot(cp);
    if (acCp >= 0.f && abCp <= acCp) return c;

    const float weightB = abCp * acAp - abAp * acCp;
    if (weightB <= 0.f && acAp >= 0.f && acCp <= 0.f)
    {
        return a + ac * (acAp / (acAp - acCp));
    }

    const float weightA = abBp * acCp - abCp * acBp;
    if (weightA <= 0.f && acBp >= abBp && abCp >= acCp)
    {
        const float edge = acBp - abBp;
        return b + (c - b) * (edge / (edge + abCp - acCp));
    }

    // The closest point lies inside the face; use barycentric weights.
    const float inv = 1.f / (weightA + weightB + weightC);
    return a + ab * (weightB * inv) + ac * (weightC * inv);
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

CARL_D CARL_FI bool nativeSphereTriangleContact(
    BallArenaContact& contact,
    const Vec3& center,
    const Tri& tri)
{
    constexpr float radius = BALL_COLLISION_RADIUS * .02f;
    constexpr float extent = radius + BALL_ARENA_CONTACT_BREAK * .02f;

    // Reject degenerate triangles and distant planes
    const Vec3 edge01 = tri.v1 - tri.v0;
    const Vec3 edge02 = tri.v2 - tri.v0;
    const Vec3 centerOffset = center - tri.v0;
    const Vec3 faceNormal = edge01.cross(edge02);
    const float normalLenSq = faceNormal.lenSq();

    if (normalLenSq < 1e-14f) return false;

    Vec3 normal = faceNormal * (1.f / sqrtf(normalLenSq));
    float planeDistance = centerOffset.dot(normal);

    if (planeDistance < 0.f)
    {
        planeDistance = -planeDistance;
        normal = normal.neg();
    }

    if (planeDistance >= extent) return false;

    // Project onto the face when inside, otherwise use the nearest edge
    const float gamma = edge01.cross(centerOffset).dot(faceNormal) / normalLenSq;
    const float beta = centerOffset.cross(edge02).dot(faceNormal) / normalLenSq;
    const float alpha = 1.f - gamma - beta;

    const bool inside = alpha >= 0.f && alpha <= 1.f
        && beta >= 0.f && beta <= 1.f
        && gamma >= 0.f && gamma <= 1.f;

    contact.point = inside ? center - normal * planeDistance
                           : closestPointOnTriangle(center, tri.v0, tri.v1, tri.v2);
    const Vec3 delta = center - contact.point;
    const float distSq = delta.lenSq();

    if (distSq >= extent * extent) return false;

    const float distance = sqrtf(distSq);
    contact.normal = distSq > FLOAT_EPSILON ? delta * (1.f / distance) : normal;
    contact.depth = distSq > FLOAT_EPSILON ? radius - distance : radius;
    contact.relPos = (contact.point - contact.normal * contact.depth) - center;
    contact.reductionPos = contact.relPos;

    return true;
}

CARL_D CARL_FI int ballContactReplacement(
    const BallArenaContact* contacts,
    const BallArenaContact& candidate)
{
    // Find the deepest contact among the existing points and the candidate
    int deepest = -1;
    float depth = candidate.depth;

    for (int i = 0; i < 4; i++)
    {
        if (contacts[i].depth > depth)
        {
            deepest = i;
            depth = contacts[i].depth;
        }
    }

    // Choose the replacement that gives the widest contact spread
    float areaSq[4] = { 0, 0, 0, 0 };
    const Vec3 position = candidate.reductionPos;

    if (deepest != 0)
    {
        areaSq[0] = (position - contacts[1].reductionPos)
            .cross(contacts[3].reductionPos - contacts[2].reductionPos).lenSq();
    }
    if (deepest != 1)
    {
        areaSq[1] = (position - contacts[0].reductionPos)
            .cross(contacts[3].reductionPos - contacts[2].reductionPos).lenSq();
    }
    if (deepest != 2)
    {
        areaSq[2] = (position - contacts[0].reductionPos)
            .cross(contacts[3].reductionPos - contacts[1].reductionPos).lenSq();
    }
    if (deepest != 3)
    {
        areaSq[3] = (position - contacts[0].reductionPos)
            .cross(contacts[2].reductionPos - contacts[1].reductionPos).lenSq();
    }

    int best = 0;
    for (int i = 1; i < 4; i++)
    {
        if (areaSq[i] > areaSq[best]) best = i;
    }

    return best;
}

CARL_D CARL_FI int nextBallArenaTriangle(
    int* cursor,
    const int* end,
    int cells,
    ArenaMesh* arena)
{
    int selected = -1;
    int triIdx = -1;
    int key = 2147483647;

    // Merge cell lists in native mesh and traversal order
    for (int cell = 0; cell < cells; cell++)
    {
        if (cursor[cell] >= end[cell]) continue;

        const int tri = arena->ballTriIdx[cursor[cell]];
        const int candidate = arena->nativeMesh[tri] * 10000 + arena->nativeOrder[tri];

        if (candidate < key)
        {
            key = candidate;
            selected = cell;
            triIdx = tri;
        }
    }

    if (selected >= 0) cursor[selected]++;

    return triIdx;
}

CARL_D CARL_FI bool makeBallArenaContact(
    BallArenaContact& contact,
    const Vec3& center,
    int triIdx,
    ArenaMesh* arena)
{
    if (arena->hasNativeMetadata)
    {
        const Tri tri = {
            arena->nativeV0[triIdx], arena->nativeV1[triIdx], arena->nativeV2[triIdx]
        };

        if (!nativeSphereTriangleContact(contact, center * .02f, tri)) return false;
    }
    else
    {
        if (!sphereTriangleContact(contact, center, BALL_COLLISION_RADIUS, arena->ldg(triIdx)))
            return false;

        contact.reductionPos = contact.relPos;
    }

    contact.triangle = triIdx;

    return true;
}

CARL_D CARL_FI void adjustBallArenaContact(
    BallArenaContact& contact,
    const Vec3& center,
    int triIdx,
    ArenaMesh* arena)
{
    if (arena->hasNativeMetadata)
    {
        const Tri tri = {
            arena->nativeV0[triIdx], arena->nativeV1[triIdx], arena->nativeV2[triIdx]
        };

        adjustNativeBallEdgeNormal(
            contact, tri, arena->nativeEdgeAngles[triIdx], arena->nativeEdgeFlags[triIdx], .1f);
        contact.point = center * .02f + contact.relPos + contact.normal * contact.depth;
    }
    else
    {
        adjustBallArenaEdgeNormal(
            contact, contact.point, arena->ldg(triIdx), arena->triAngs[triIdx]);
        contact.point = center + contact.relPos + contact.normal * contact.depth;
    }
}

CARL_D CARL_FI int finishNativeBallArenaContacts(
    BallArenaContact* contacts,
    int count,
    const Vec3& center)
{
    // Convert native contacts back to Unreal units
    for (int i = 0; i < count; i++)
    {
        contacts[i].point = contacts[i].point * 50.f;
        contacts[i].relPos = contacts[i].relPos * 50.f;
        contacts[i].depth *= 50.f;
    }

    // Add contacts for the analytic arena planes
    const float offsets[4] = { 0.f, -2044.f, -4096.f, -4096.f };

    for (int plane = 0; plane < 4; plane++)
    {
        const Vec3 normal = ARENA_PLANE_NORMALS[plane];
        const float depth = BALL_COLLISION_RADIUS - (normal.dot(center) - offsets[plane]);

        if (depth > -BALL_ARENA_CONTACT_BREAK)
        {
            assert(count < MAX_BALL_ARENA_CONTACTS);
            contacts[count++] = {
                center - normal * (BALL_COLLISION_RADIUS - depth),
                normal * -BALL_COLLISION_RADIUS,
                normal, depth
            };
        }
    }

    return count;
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
    constexpr float diameter = 2.f * (BALL_COLLISION_RADIUS + BALL_ARENA_CONTACT_BREAK);
    static_assert(
        diameter <= 2.f * CELL_SIZE.x &&
        diameter <= 2.f * CELL_SIZE.y &&
        diameter <= 2.f * CELL_SIZE.z);

    int cursor[maxCells];
    int end[maxCells];
    int cells = 0;

    // Collect the sorted triangle range for each overlapping grid cell
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

    int count = 0;
    int mesh = -1;
    int base = 0;
    int localCount = 0;
    int lastTri = -1;

    // Reduce contacts separately for each native mesh
    while (true)
    {
        const int triIdx = nextBallArenaTriangle(cursor, end, cells, arena);

        if (triIdx < 0) break;
        if (triIdx == lastTri) continue;

        lastTri = triIdx;
        const int current = arena->nativeMesh[triIdx];

        if (arena->hasNativeMetadata && current < 0) continue;

        BallArenaContact contact;

        if (!makeBallArenaContact(contact, center, triIdx, arena)) continue;

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
        {
            slot = ballContactReplacement(contacts + base, contact);
        }

        adjustBallArenaContact(contact, center, triIdx, arena);
        contacts[base + slot] = contact;
    }

    return arena->hasNativeMetadata ? finishNativeBallArenaContacts(contacts, count, center) : count;
}
