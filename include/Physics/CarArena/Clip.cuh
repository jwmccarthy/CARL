#pragma once

#include <cfloat>

#include "Triangle.cuh"

#include "../../Arena/ArenaMesh.cuh"
#include "../../Arena/TriMesh.cuh"
#include "../../RLConstants.cuh"
#include "../../State/Workspace.cuh"
#include "../Collision/Clip.cuh"

constexpr float TRI_EDGE_DIST_MAX = 5.f;
constexpr float TRI_EDGE_DIST_MAX_SQ = TRI_EDGE_DIST_MAX * TRI_EDGE_DIST_MAX;

struct TriClip : LocalTriangle
{
    Vec3 faceNormal;
    Vec3 angles;
};

struct Contact
{
    Vec3 carPos;
    Vec3 arenaPos;
    Vec3 normal;
    float depth;
};

CARL_D CARL_FI TriClip loadTriangleClip(
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena,
    int pairIdx)
{
    TriClip tri;
    const int triIdx = __ldg(&space->ctNrw.triIdx[pairIdx]);

    tri.set(
        triIdx,
        Vec3::ldg(space->ctNrw.pairV0[pairIdx]),
        Vec3::ldg(space->ctNrw.pairV1[pairIdx]),
        Vec3::ldg(space->ctNrw.pairV2[pairIdx]));

    tri.faceNormal = tri.normal();
    tri.angles = Vec3::ldg(arena->triAngs[triIdx]);

    return tri;
}

CARL_D CARL_FI Vec3 nearestPointOnSegment(
    Vec3 point,
    Vec3 start,
    Vec3 end)
{
    const Vec3 segment = end - start;
    const Vec3 pointOffset = point - start;
    const float proj = pointOffset.dot(segment) / segment.lenSq();
    const float t = clampf(proj, 0.f, 1.f);

    return start + segment * t;
}

CARL_D CARL_FI bool clampNormal(
    const Vec3& edgeAxis,
    const Vec3& triNormal,
    const Vec3& contactNormal,
    float edgeAngle,
    Vec3& clampedNormal)
{
    Vec3 edgeCross = edgeAxis.cross(triNormal);
    const float crossLenSq = edgeCross.lenSq();
    if (crossLenSq <= 1e-12f) return false;

    edgeCross = edgeCross * rsqrtf(crossLenSq);
    const float tangentProj = contactNormal.dot(edgeCross);
    const float normalProj = contactNormal.dot(triNormal);
    const float currentAngle = atan2f(tangentProj, normalProj);

    const bool outside = edgeAngle < 0.f
        ? currentAngle < edgeAngle
        : currentAngle > edgeAngle;

    if (!outside) return false;

    clampedNormal = rotateAroundAxis(
        contactNormal, edgeAxis, edgeAngle - currentAngle);

    return true;
}

CARL_D CARL_FI void adjustInternalEdgeContact(
    const TriClip& tri,
    Contact& contact)
{
    if (tri.angles.gte(BOUNDARY_EDGE_ANGLE)) return;

    int bestEdge = -1;
    float bestDistSq = FLT_MAX;

    for (int edge = 0; edge < 3; edge++)
    {
        if (fabsf(tri.angles[edge]) >= BOUNDARY_EDGE_ANGLE) continue;

        const Vec3 nearest = nearestPointOnSegment(
            contact.arenaPos,
            tri.getVert(edge),
            tri.getVert((edge + 1) % 3));
        const float distSq = (contact.arenaPos - nearest).lenSq();

        if (distSq < bestDistSq)
        {
            bestDistSq = distSq;
            bestEdge = edge;
        }
    }

    // Only adjust contacts close enough to a shared triangle edge.
    if (bestEdge < 0 || bestDistSq >= TRI_EDGE_DIST_MAX_SQ) return;

    const Vec3 localNormal = contact.normal.norm();
    const float angle = tri.angles[bestEdge];

    auto snapToFace = [&]()
    {
        if (tri.faceNormal.dot(localNormal) >= 0.f)
        {
            contact.normal = tri.faceNormal;
            contact.arenaPos = contact.carPos + tri.faceNormal * contact.depth;
        }
    };

    if (angle == 0.f)
    {
        snapToFace();
        return;
    }

    const float side = angle > 0.f ? 1.f : -1.f;
    const Vec3 edgeAxis = tri.getEdge(bestEdge).norm();
    const Vec3 normalA = tri.faceNormal * side;
    const Vec3 normalB = rotateAroundAxis(
        tri.faceNormal, edgeAxis, angle) * side;

    // Normals behind both adjoining faces should use the current face normal.
    if (localNormal.dot(normalA) < 0.f && localNormal.dot(normalB) < 0.f)
    {
        snapToFace();
        return;
    }

    Vec3 clamped;

    // Accept the edge clamp only when the result still points out of the face.
    if (clampNormal(edgeAxis, normalA, localNormal, angle, clamped)
        && clamped.dot(tri.faceNormal) > 0.f)
    {
        contact.normal = clamped.norm();
        contact.arenaPos = contact.carPos + contact.normal * contact.depth;
    }
}

CARL_D CARL_FI Contact edgeEdgeContact(
    const TriClip& tri,
    float minPen,
    int axisIdx,
    Vec3 sepNormal)
{
    const int carAxisIdx = (axisIdx - 4) / 3;
    const int triEdgeIdx = (axisIdx - 4) % 3;
    const Vec3 triCenter = tri.center();

    if (sepNormal.dot(triCenter) > 0.f) sepNormal = sepNormal.neg();

    Vec3 carPoint = sepNormal.neg().sign() * CAR_HALF_EX;
    carPoint[carAxisIdx] = 0.f;
    const Vec3 carAxis = WORLD_AXES[carAxisIdx];

    const Vec3 triPoint = tri.getVert(triEdgeIdx);
    const Vec3 triEdge = tri.getEdge(triEdgeIdx);
    const float triLength = triEdge.len();
    const Vec3 triAxis = triEdge / triLength;

    const Vec3 carToTri = triPoint - carPoint;
    const float axisDot = carAxis.dot(triAxis);
    const float denom = 1.f - axisDot * axisDot;

    Vec3 pointOnTri = triPoint;
    if (denom > 1e-6f)
    {
        const float carProj = carToTri.dot(carAxis);
        const float triProj = carToTri.dot(triAxis);
        const float unclampedParam =
            (axisDot * carProj - triProj) / denom;
        const float triParam = clampf(unclampedParam, 0.f, triLength);

        pointOnTri = triPoint + triAxis * triParam;
    }

    const float carParamProj = carAxis.dot(pointOnTri - carPoint);
    const float carParam = clampf(
        carParamProj,
        -CAR_HALF_EX[carAxisIdx],
        CAR_HALF_EX[carAxisIdx]);
    const Vec3 pointOnCar = carPoint + carAxis * carParam;

    const Vec3 separation = pointOnTri - pointOnCar;
    const float separationSq = separation.lenSq();

    if (separationSq > 1e-8f)
    {
        Vec3 normal = separation * rsqrtf(separationSq);
        if (normal.dot(triCenter) > 0.f) normal = normal.neg();
        return { pointOnCar, pointOnTri, normal, minPen };
    }

    return { pointOnCar, pointOnTri, sepNormal, minPen };
}

CARL_D CARL_FI int carHitPairCount(
    Workspace* __restrict__ space,
    int carIdx)
{
    const int count = __ldg(&space->ctHit.carHitCount[carIdx]);
    return count < MAX_CAR_TRI_PAIRS ? count : MAX_CAR_TRI_PAIRS;
}

CARL_D CARL_FI Contact readCarTriContact(
    const CarTriContact& contacts,
    int idx)
{
    return {
        Vec3::ldg(contacts.carPos[idx]),
        Vec3::ldg(contacts.arenaPos[idx]),
        Vec3::ldg(contacts.normal[idx]),
        __ldg(&contacts.depth[idx])
    };
}

CARL_D CARL_FI void writeCarTriContact(
    CarTriContact& contacts,
    const Contact& contact,
    int idx)
{
    contacts.carPos[idx] = contact.carPos;
    contacts.arenaPos[idx] = contact.arenaPos;
    contacts.normal[idx] = contact.normal;
    contacts.depth[idx] = contact.depth;
}

CARL_D CARL_FI Vec3 witnessFaceNormal(const Vec3& sepNormal)
{
    const int axis = sepNormal.abs().argMax();

    return WORLD_AXES[axis] * -copysignf(1.f, sepNormal[axis]);
}

CARL_D CARL_FI int clipTriangleToCarFace(
    Vec3*& input,
    Vec3*& output,
    const TriClip& tri,
    int faceAxis)
{
    int count = 3;

    input[0] = tri.v0;
    input[1] = tri.v1;
    input[2] = tri.v2;

    for (int axis = 0; axis < 3 && count > 0; axis++)
    {
        if (axis == faceAxis) continue;

        for (int sign = 1; sign >= -1 && count > 0; sign -= 2)
        {
            count = clipPlane(
                input, count, output,
                WORLD_AXES[axis] * (float)sign,
                CAR_HALF_EX[axis]);

            Vec3* swap = input;
            input = output;
            output = swap;
        }
    }

    return count;
}

CARL_D CARL_FI void writeFaceContacts(
    Workspace* __restrict__ space,
    int pairIdx,
    const Vec3* polygon,
    int vertexCount,
    Vec3 sepNormal,
    float minPen)
{
    const int contactBase = pairIdx * MAX_PAIR_CONTACTS;
    const int faceAxis = sepNormal.abs().argMax();
    const Vec3 faceNormal = witnessFaceNormal(sepNormal);
    const float faceDistance = CAR_HALF_EX[faceAxis];
    const float minDepth = -(minPen + CAR_CONTACT_BREAK);

    int count = 0;

    for (int i = 0; i < vertexCount; i++)
    {
        float depth = faceNormal.dot(polygon[i]) - faceDistance;
        depth = fmaxf(depth, minDepth);
        if (depth > CAR_CONTACT_BREAK) continue;

        const Contact contact = {
            polygon[i] + sepNormal * depth,
            polygon[i],
            sepNormal,
            -depth
        };

        writeCarTriContact(
            space->ctCon, contact, contactBase + count++);
    }

    space->ctNrw.conPairCount[pairIdx] = count;
}

CARL_D CARL_FI void carArenaClipPair(
    ArenaMesh* __restrict__ arena,
    Workspace* __restrict__ space,
    int pairIdx)
{
    const Vec3 sepNormal = Vec3::ldg(space->ctNrw.minAxis[pairIdx]);
    const float minPen = __ldg(&space->ctNrw.minPen[pairIdx]);
    const int axisIdx = __ldg(&space->ctNrw.axisIdx[pairIdx]);

    const TriClip tri = loadTriangleClip(space, arena, pairIdx);

    if (axisIdx >= 4)
    {
        const Contact contact = edgeEdgeContact(
            tri, minPen, axisIdx, sepNormal);

        writeCarTriContact(
            space->ctCon, contact, pairIdx * MAX_PAIR_CONTACTS);

        space->ctNrw.conPairCount[pairIdx] = 1;
        return;
    }

    Vec3 polygonA[MAX_PAIR_CONTACTS];
    Vec3 polygonB[MAX_PAIR_CONTACTS];
    Vec3* input = polygonA;
    Vec3* output = polygonB;

    const int faceAxis = sepNormal.abs().argMax();
    const int vertexCount = clipTriangleToCarFace(
        input, output, tri, faceAxis);

    writeFaceContacts(
        space, pairIdx, input, vertexCount, sepNormal, minPen);
}

CARL_D CARL_FI void carArenaClip(
    ArenaMesh* __restrict__ arena,
    Workspace* __restrict__ space,
    int hitIdx)
{
    if (hitIdx >= space->ctNrw.maxCarTriPairs) return;

    const int carIdx = hitIdx / MAX_CAR_TRI_PAIRS;
    const int pairBase = __ldg(&space->ctHit.carHitStart[carIdx]);
    const int pairOffset = hitIdx - pairBase;

    if (pairOffset >= carHitPairCount(space, carIdx)) return;

    carArenaClipPair(arena, space, pairBase + pairOffset);
}
