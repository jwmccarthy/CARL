#pragma once

#include "Cuda/Math.cuh"

CARL_D CARL_FI int clipPlane(
    const Vec3* input,
    int inputCount,
    Vec3* output,
    const Vec3& planeNormal,
    float planeDist)
{
    if (inputCount == 0) return 0;

    int outputCount = 0;
    for (int i = 0; i < inputCount; i++)
    {
        const Vec3 current = input[i];
        const Vec3 next = input[(i + 1) % inputCount];

        const float currentDist = planeNormal.dot(current) - planeDist;
        const float nextDist = planeNormal.dot(next) - planeDist;
        const bool currentInside = currentDist <= 0.f;
        const bool nextInside = nextDist <= 0.f;

        if (currentInside) output[outputCount++] = current;

        if (currentInside != nextInside)
        {
            const float t = currentDist / (currentDist - nextDist);
            output[outputCount++] = current + (next - current) * t;
        }
    }

    return outputCount;
}
