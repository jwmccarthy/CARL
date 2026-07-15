#pragma once

#include <cfloat>

#include "Cuda/Math.cuh"

struct AxisProj
{
    float min;
    float max;
};

struct SATResult
{
    bool overlap = true;
    int axisIdx = 0;
    float minPen = FLT_MAX;
    Vec3 minAxis = {};
};
