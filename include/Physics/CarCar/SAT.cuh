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
    if (depth >= result.minPen) return;

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

    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        testCarCarAxis(axesA[i], boxA, boxB, i, result);
        if (!result.overlap) return result;
    }

    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        testCarCarAxis(axesB[i], boxA, boxB, 3 + i, result);
        if (!result.overlap) return result;
    }

    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        #pragma unroll
        for (int j = 0; j < 3; j++)
        {
            const int axisIdx = 6 + i * 3 + j;
            testCarCarAxis(
                axesA[i].cross(axesB[j]), boxA, boxB, axisIdx, result);
            if (!result.overlap) return result;
        }
    }

    return result;
}
