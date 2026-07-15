#pragma once

#include <cfloat>

#include "../../DataUtils.cuh"
#include "../../RLConstants.cuh"
#include "../../State/Workspace.cuh"
#include "Clip.cuh"

constexpr float CAR_MANIFOLD_BREAK_SQ = CAR_CONTACT_BREAK * CAR_CONTACT_BREAK;

struct ManifoldPoint
{
    Vec3 carPos;
    Vec3 worldPos;
    Vec3 worldNormal;
    float dist;
    float friction;
    float restitution;
    float impulse;
    float impulseTan;
    int lifetime;
};

CARL_D CARL_FI ManifoldPoint readManifoldPoint(
    const CarTriManifold& manifold,
    int idx)
{
    return {
        Vec3::ldg(manifold.carPos[idx]),
        Vec3::ldg(manifold.worldPos[idx]),
        Vec3::ldg(manifold.worldNormal[idx]),
        __ldg(&manifold.dist[idx]),
        __ldg(&manifold.friction[idx]),
        __ldg(&manifold.restitution[idx]),
        __ldg(&manifold.impulse[idx]),
        __ldg(&manifold.impulseTan[idx]),
        __ldg(&manifold.lifetime[idx])
    };
}

CARL_D CARL_FI void writeManifoldPoint(
    CarTriManifold& manifold,
    const ManifoldPoint& point,
    int idx)
{
    manifold.carPos[idx] = point.carPos;
    manifold.worldPos[idx] = point.worldPos;
    manifold.worldNormal[idx] = point.worldNormal;
    manifold.dist[idx] = point.dist;
    manifold.friction[idx] = point.friction;
    manifold.restitution[idx] = point.restitution;
    manifold.impulse[idx] = point.impulse;
    manifold.impulseTan[idx] = point.impulseTan;
    manifold.lifetime[idx] = point.lifetime;
}

CARL_D CARL_FI ManifoldPoint makeManifoldPoint(
    const Contact& contact,
    const CarPose& pose)
{
    return {
        contact.carPos,
        pose.cen + pose.rot.toWorld(contact.arenaPos),
        pose.rot.toWorld(contact.normal),
        -contact.depth,
        CAR_WORLD_FRICTION,
        CAR_WORLD_RESTITUTION,
        0.f,
        0.f,
        0
    };
}

CARL_D CARL_FI int manifoldReplacementIndex(
    const ManifoldPoint* points,
    const ManifoldPoint& candidate)
{
    const Vec3 p0 = points[0].carPos;
    const Vec3 p1 = points[1].carPos;
    const Vec3 p2 = points[2].carPos;
    const Vec3 p3 = points[3].carPos;

    int deepest = -1;
    float deepestPenetration = -candidate.dist;
    for (int i = 0; i < MAX_CAR_MANIFOLD_POINTS; i++)
    {
        const float penetration = -points[i].dist;
        if (penetration > deepestPenetration)
        {
            deepest = i;
            deepestPenetration = penetration;
        }
    }

    float area[4]{};
    if (deepest != 0)
        area[0] = (candidate.carPos - p1).cross(p3 - p2).lenSq();
    if (deepest != 1)
        area[1] = (candidate.carPos - p0).cross(p3 - p2).lenSq();
    if (deepest != 2)
        area[2] = (candidate.carPos - p0).cross(p3 - p1).lenSq();
    if (deepest != 3)
        area[3] = (candidate.carPos - p0).cross(p2 - p1).lenSq();

    int best = 0;
    for (int i = 1; i < MAX_CAR_MANIFOLD_POINTS; i++)
    {
        if (area[i] > area[best]) best = i;
    }

    return best;
}

CARL_D CARL_FI void removeManifoldPoint(
    ManifoldPoint* points,
    int& count,
    int removeIdx)
{
    const int last = --count;
    if (removeIdx != last) points[removeIdx] = points[last];
    points[last] = {};
}

CARL_D CARL_FI void refreshManifold(
    ManifoldPoint* points,
    int& count,
    const CarPose& pose)
{
    for (int i = count - 1; i >= 0; i--)
    {
        const Vec3 worldCarPos = pose.cen + pose.rot.toWorld(points[i].carPos);
        const float dist = (worldCarPos - points[i].worldPos)
            .dot(points[i].worldNormal);

        if (dist > CAR_CONTACT_BREAK)
        {
            removeManifoldPoint(points, count, i);
            continue;
        }

        const Vec3 proj = worldCarPos - points[i].worldNormal * dist;
        if ((points[i].worldPos - proj).lenSq() > CAR_MANIFOLD_BREAK_SQ)
        {
            removeManifoldPoint(points, count, i);
            continue;
        }

        points[i].dist = dist;
        points[i].lifetime++;
    }
}

CARL_D CARL_FI void addManifoldPoint(
    ManifoldPoint* points,
    int& count,
    const ManifoldPoint& point)
{
    int idx = count;
    if (idx == MAX_CAR_MANIFOLD_POINTS)
        idx = manifoldReplacementIndex(points, point);
    else
        count++;

    points[idx] = point;
}

CARL_D CARL_FI bool hasNearbyManifoldPoint(
    const ManifoldPoint* points,
    int count,
    const Vec3& carPos)
{
    for (int i = 0; i < count; i++)
    {
        if ((points[i].carPos - carPos).lenSq() < CAR_MANIFOLD_BREAK_SQ)
            return true;
    }

    return false;
}

CARL_D CARL_FI void insertBestContact(
    ManifoldPoint* points,
    int& pointCount,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena,
    int carIdx,
    const CarPose& pose)
{
    const int pairBase = __ldg(&space->ctHit.carHitStart[carIdx]);
    const int pairCount = carHitPairCount(space, carIdx);
    int bestPair = -1;
    Contact best{};
    float bestDist = FLT_MAX;

    for (int pairOffset = 0; pairOffset < pairCount; pairOffset++)
    {
        const int pairIdx = pairBase + pairOffset;
        const int contactBase = pairIdx * MAX_PAIR_CONTACTS;
        const int contactCount = __ldg(&space->ctNrw.conPairCount[pairIdx]);

        for (int contactOffset = 0; contactOffset < contactCount; contactOffset++)
        {
            const Contact contact = readCarTriContact(
                space->ctCon, contactBase + contactOffset);

            if (hasNearbyManifoldPoint(points, pointCount, contact.carPos))
                continue;

            const float dist = -contact.depth;
            if (dist < bestDist)
            {
                bestDist = dist;
                best = contact;
                bestPair = pairIdx;
            }
        }
    }

    if (bestPair < 0) return;

    adjustInternalEdgeContact(loadTriangleClip(space, arena, bestPair), best);
    addManifoldPoint(points, pointCount, makeManifoldPoint(best, pose));
}

CARL_D CARL_FI void writeLocalManifold(
    CarTriManifold& manifold,
    int carIdx,
    const ManifoldPoint* points,
    int count)
{
    const int base = carIdx * MAX_CAR_MANIFOLD_POINTS;
    manifold.count[carIdx] = count;

    for (int i = 0; i < count; i++)
        writeManifoldPoint(manifold, points[i], base + i);

    for (int i = count; i < MAX_CAR_MANIFOLD_POINTS; i++)
        writeManifoldPoint(manifold, {}, base + i);
}

CARL_D __noinline__ int buildLocalManifold(
    ManifoldPoint* points,
    Workspace* __restrict__ space,
    ArenaMesh* __restrict__ arena,
    int carIdx,
    const CarPose& pose)
{
    const int base = carIdx * MAX_CAR_MANIFOLD_POINTS;
    int pointCount = __ldg(&space->ctMan.count[carIdx]);

    for (int i = 0; i < pointCount; i++)
        points[i] = readManifoldPoint(space->ctMan, base + i);

    refreshManifold(points, pointCount, pose);
    insertBestContact(
        points, pointCount, space, arena, carIdx, pose);

    return pointCount;
}
