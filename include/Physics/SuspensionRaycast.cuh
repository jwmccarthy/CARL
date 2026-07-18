#pragma once

#include "../Arena/ArenaMesh.cuh"
#include "../DataUtils.cuh"
#include "../RLConstants.cuh"
#include "../State/Workspace.cuh"
#include "SuspensionUtils.cuh"

constexpr float RAY_PARALLEL_EPS = 1e-8f;

struct RayHit
{
    float dist;
    Vec3 normal;
};

// --- Ray-shape intersection primitives ---

CARL_D CARL_FI Vec3 suspRayStart(const CarPose& pose, int wheel)
{
    return pose.cen + pose.rot.toWorld(suspConnection(wheel));
}

CARL_D CARL_FI bool raySphereImpact(
    const Vec3& rayStart,
    const Vec3& rayDir,
    const Vec3& center,
    float radius,
    float maxDist,
    float& dist)
{
    const Vec3 offset = rayStart - center;
    const float b = offset.dot(rayDir);
    const float c = offset.lenSq() - radius * radius;

    if (c < 0.f) return false;

    const float discriminant = b * b - c;
    if (discriminant < 0.f) return false;

    dist = -b - sqrtf(discriminant);
    return dist > 0.f && dist <= maxDist;
}

CARL_D CARL_FI bool rayOBBImpact(
    const Vec3& rayStart, 
    const Vec3& rayDir,
    const Vec3& center, 
    const Quat& rot,
    const Vec3& halfExtents, 
    float maxDist, 
    float& dist, 
    Vec3& normal)
{
    const Vec3 localStart = rot.toLocal(rayStart - center);
    const Vec3 localDir = rot.toLocal(rayDir);

    float entryDist = 0.f;
    float exitDist = maxDist;
    int hitAxis = -1;

    for (int axis = 0; axis < 3; axis++)
    {
        if (fabsf(localDir[axis]) < RAY_PARALLEL_EPS)
        {
            if (fabsf(localStart[axis]) > halfExtents[axis]) return false;
            continue;
        }

        const float invDir = 1.f / localDir[axis];
        const float t1 = (-halfExtents[axis] - localStart[axis]) * invDir;
        const float t2 = ( halfExtents[axis] - localStart[axis]) * invDir;

        const float tNear = fminf(t1, t2);
        if (tNear > entryDist)
        {
            entryDist = tNear;
            hitAxis = axis;
        }

        exitDist = fminf(exitDist, fmaxf(t1, t2));
        if (entryDist > exitDist) return false;
    }

    if (hitAxis < 0) return false;

    dist = entryDist;

    Vec3 localNormal = Vec3::zero();
    localNormal[hitAxis] = localDir[hitAxis] > 0.f ? -1.f : 1.f;
    normal = rot.toWorld(localNormal);
    return true;
}

// --- Per-shape wheel ray tests ---

CARL_D CARL_FI void testTriangleWheels(
    const CarPose& pose,
    const Vec3& rayDir,
    const Vec3& v0,
    const Vec3& edge1,
    const Vec3& edge2,
    RayHit best[NUM_WHEELS])
{
    const Vec3 pVec = rayDir.cross(edge2);
    const float det = edge1.dot(pVec);
    if (fabsf(det) < RAY_PARALLEL_EPS) return;

    const float invDet = 1.f / det;

    #pragma unroll
    for (int w = 0; w < NUM_WHEELS; w++)
    {
        const Vec3 tVec = suspRayStart(pose, w) - v0;
        const float u = invDet * tVec.dot(pVec);
        if (u < 0.f || u > 1.f) continue;

        const Vec3 qVec = tVec.cross(edge1);
        const float v = invDet * rayDir.dot(qVec);
        if (v < 0.f || u + v > 1.f) continue;

        const float hitDist = invDet * edge2.dot(qVec);
        if (hitDist <= 0.f || hitDist >= best[w].dist) continue;

        best[w].dist = hitDist;

        Vec3 faceN = edge1.cross(edge2);
        const float nl = faceN.lenSq();
        if (nl > 1e-12f)
        {
            faceN = faceN * rsqrtf(nl);
            if (faceN.dot(rayDir) > 0.f) faceN = faceN.neg();
            best[w].normal = faceN;
        }
    }
}

CARL_D CARL_FI void testOBBWheels(
    const CarPose& pose,
    const Vec3& rayDir,
    const Vec3& center,
    const Quat& rot,
    RayHit best[NUM_WHEELS])
{
    #pragma unroll
    for (int w = 0; w < NUM_WHEELS; w++)
    {
        float dist;
        Vec3 normal;

        const bool hit = rayOBBImpact(
            suspRayStart(pose, w), rayDir, center, rot,
            CAR_HALF_EX, suspRayLength(w), dist, normal);
        if (!hit || dist >= best[w].dist) continue;

        best[w] = { dist, normal };
    }
}

CARL_D CARL_FI void testSphereWheels(
    const CarPose& pose, 
    const Vec3& rayDir,
    const Vec3& center, 
    float radius,
    RayHit best[NUM_WHEELS])
{
    const float invRadius = 1.f / radius;

    #pragma unroll
    for (int w = 0; w < NUM_WHEELS; w++)
    {
        float dist;
        const Vec3 rayStart = suspRayStart(pose, w);

        const bool hit = raySphereImpact(
            rayStart, rayDir, center, radius, suspRayLength(w), dist);
        if (!hit || dist >= best[w].dist) continue;

        const Vec3 hitDir = rayStart + rayDir * dist - center;
        best[w] = { dist, hitDir * invRadius };
    }
}

// --- Setup and storage ---

CARL_D CARL_FI void initWheelHits(
    RayHit best[NUM_WHEELS],
    const Vec3& carUp)
{
    #pragma unroll
    for (int w = 0; w < NUM_WHEELS; w++)
    {
        best[w] = { suspRayLength(w) + 1.f, carUp };
    }
}

CARL_D CARL_FI void storeWheelHits(
    CarSuspension& susp,
    int carIdx,
    const RayHit best[NUM_WHEELS])
{
    const int wheelBase = carIdx * NUM_WHEELS;

    #pragma unroll
    for (int w = 0; w < NUM_WHEELS; w++)
    {
        susp.rayDist[wheelBase + w] = best[w].dist;
        susp.rayNormal[wheelBase + w] = best[w].normal;
    }
}

// --- Per-shape batch tests ---

CARL_D CARL_FI void testArenaWheels(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena,
    const CarPose& pose,
    const Vec3& rayDir,
    int carIdx,
    RayHit best[NUM_WHEELS])
{
    const int cellIdx = __ldg(&space->bp.cellIdx[carIdx]);
    const int triStart = __ldg(&arena->prefix[cellIdx]);
    const int triCount = __ldg(&space->bp.numTris[carIdx]);

    for (int i = 0; i < triCount; i++)
    {
        const int triIdx = __ldg(&arena->triIdx[triStart + i]);
        const Int3 vertIdx = Int3::ldg(arena->tris[triIdx]);

        const Vec3 v0 = Vec3::ldg(arena->verts[vertIdx.x]);
        const Vec3 edge1 = Vec3::ldg(arena->verts[vertIdx.y]) - v0;
        const Vec3 edge2 = Vec3::ldg(arena->verts[vertIdx.z]) - v0;

        testTriangleWheels(pose, rayDir, v0, edge1, edge2, best);
    }
}

CARL_D CARL_FI void testCarWheels(
    GameState* __restrict__ state,
    const CarPose& pose,
    const Vec3& rayDir,
    int carIdx,
    int simIdx,
    RayHit best[NUM_WHEELS])
{
    const int carBase = simIdx * state->nCars;

    for (int c = 0; c < state->nCars; c++)
    {
        const int otherIdx = carBase + c;
        if (otherIdx == carIdx) continue;
        if (__ldg(&state->cars.isDemoed[otherIdx])) continue;

        const Vec3 otherCen = Vec3::ldg(state->cars.cen[otherIdx]);
        const Quat otherRot = Quat::ldg(state->cars.rot[otherIdx]);

        testOBBWheels(pose, rayDir, otherCen, otherRot, best);
    }
}

CARL_D CARL_FI void testBallWheels(
    GameState* __restrict__ state,
    const CarPose& pose,
    const Vec3& rayDir,
    int simIdx,
    RayHit best[NUM_WHEELS])
{
    const Vec3 ballPos = Vec3::ldg(state->ball.pos[simIdx]);
    testSphereWheels(pose, rayDir, ballPos, BALL_COLLISION_RADIUS, best);
}

// --- Entry point ---

CARL_D CARL_FI void raycastCarSuspension(
    GameState* __restrict__ state,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena,
    int carIdx)
{
    const CarPose pose = loadCarPose(state, carIdx);
    const Vec3 carUp = pose.rot.toWorld(WORLD_Z);
    const Vec3 rayDir = carUp.neg();
    const int simIdx = carIdx / state->nCars;

    RayHit best[NUM_WHEELS];
    initWheelHits(best, carUp);

    testArenaWheels(state, space, arena, pose, rayDir, carIdx, best);
    testCarWheels(state, pose, rayDir, carIdx, simIdx, best);
    testBallWheels(state, pose, rayDir, simIdx, best);

    storeWheelHits(space->susp, carIdx, best);
}
