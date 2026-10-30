#pragma once

#include "Physics/Collision/SAT.cuh"
#include "Types.cuh"

CARL_D CARL_FI AxisProj projectBox(
    const CarBox& box,
    const Vec3& axis)
{
    const Vec3 localAxis = box.rot.toLocal(axis);
    const float radius = localAxis.abs().dot(CAR_HALF_EX);
    const float center = box.cen.dot(axis);

    return { center - radius, center + radius };
}

CARL_D CARL_FI void testCarCarAxis(
    Vec3 axis,
    const CarBox& boxA,
    const CarBox& boxB,
    int axisIdx,
    SATResult& result)
{
    const float lenSq = axis.lenSq();
    if (lenSq < 1e-12f) return;

    axis = axis * rsqrtf(lenSq);

    const AxisProj projA = projectBox(boxA, axis);
    const AxisProj projB = projectBox(boxB, axis);

    if (projA.max < projB.min || projB.max < projA.min)
    {
        result.overlap = false;
        return;
    }

    const float depthAB = projA.max - projB.min;
    const float depthBA = projB.max - projA.min;
    const float depth = fminf(depthAB, depthBA);
    if (depth * (axisIdx >= 6 ? 1.05f : 1.f) >= result.minPen)
        return;

    result.minPen = depth;
    result.minAxis = depthAB < depthBA ? axis.neg() : axis;
    result.axisIdx = axisIdx;
}

CARL_D CARL_FI SATResult carCarSAT(
    const CarBox& boxA,
    const CarBox& boxB)
{
    SATResult result;
    Vec3 axesA[3];
    Vec3 axesB[3];

    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        axesA[i] = boxA.axis(i);
        axesB[i] = boxB.axis(i);
    }

    const Vec3 delta = boxB.cen * .02f - boxA.cen * .02f;
    const Vec3 half = CAR_HALF_EX * .02f;
    float alignment[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            alignment[i][j] = fabsf(axesA[i].dot(axesB[j]));

#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        const float distance = delta.dot(axesA[i]);
        const float radius = half[i] + half.x * alignment[i][0] + half.y * alignment[i][1] +
                             half.z * alignment[i][2];
        const float depth = (radius - fabsf(distance)) * 50.f;
        if (depth < 0.f)
        {
            result.overlap = false;
            return result;
        }
        if (depth < result.minPen)
        {
            result.minPen = depth;
            result.minAxis = distance > 0 ? axesA[i].neg() : axesA[i];
            result.axisIdx = i;
        }
    }

    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        const float distance = delta.dot(axesB[i]);
        const float radius = half.x * alignment[0][i] + half.y * alignment[1][i] +
                             half.z * alignment[2][i] + half[i];
        const float depth = (radius - fabsf(distance)) * 50.f;
        if (depth < 0.f)
        {
            result.overlap = false;
            return result;
        }
        if (depth < result.minPen)
        {
            result.minPen = depth;
            result.minAxis = distance > 0 ? axesB[i].neg() : axesB[i];
            result.axisIdx = 3 + i;
        }
    }

    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        #pragma unroll
        for (int j = 0; j < 3; j++)
        {
            const int axisIdx = 6 + i * 3 + j;
            testCarCarAxis(axesA[i].cross(axesB[j]), boxA, boxB, axisIdx, result);
            if (!result.overlap) return result;
        }
    }

    return result;
}
