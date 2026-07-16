#pragma once

#include <cstdint>

#include "Cuda/Common.cuh"

struct CarControls
{
    float throttle, steer;
    float yaw, pitch, roll;
    int jump, boost, slide;
};

enum ActionField
{
    ACT_HORIZONTAL,
    ACT_VERTICAL,
    ACT_THROTTLE,
    ACT_POWERSLIDE,
    ACT_BOOST,
    ACT_AIR_ROLL,
    ACT_JUMP,
    ACT_PER_CAR
};

constexpr int ACTION_NVECS[ACT_PER_CAR] = { 3, 3, 3, 2, 2, 3, 2 };

struct DiscreteControls
{
    int32_t horizontal;
    int32_t vertical;
    int32_t throttle;
    int32_t powerslide;
    int32_t boost;
    int32_t airRoll;
    int32_t jump;

    CARL_D CARL_FI static float axis(int32_t action)
    {
        return action == 1 ? -1.f : action == 2 ? 1.f : 0.f;
    }

    CARL_D CARL_FI CarControls decode() const
    {
        const float horizontalAxis = axis(horizontal);

        return {
            axis(throttle),
            horizontalAxis,
            horizontalAxis,
            axis(vertical),
            axis(airRoll),
            jump == 1,
            boost == 1,
            powerslide == 1
        };
    }
};

static_assert(sizeof(DiscreteControls) == ACT_PER_CAR * sizeof(int32_t));
