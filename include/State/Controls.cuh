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
constexpr int ACT_NON_NEUTRAL = 1;

constexpr int actionLogitOffset(ActionField field)
{
    int offset = 0;
    for (int i = 0; i < field; i++) offset += ACTION_NVECS[i];
    return offset;
}

constexpr int ACTION_LOGITS = actionLogitOffset(ACT_PER_CAR);

constexpr float CONTINUOUS_AXIS_MIN = -1.f;
constexpr float CONTINUOUS_BUTTON_MIN = 0.f;
constexpr float CONTINUOUS_ACTION_MAX = 1.f;
constexpr float CONTINUOUS_ACTION_LOW[ACT_PER_CAR] = {
    CONTINUOUS_AXIS_MIN, CONTINUOUS_AXIS_MIN, CONTINUOUS_AXIS_MIN,
    CONTINUOUS_BUTTON_MIN, CONTINUOUS_BUTTON_MIN,
    CONTINUOUS_AXIS_MIN, CONTINUOUS_BUTTON_MIN
};

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
        return action == ACT_NON_NEUTRAL ? CONTINUOUS_AXIS_MIN
            : action == ACT_NON_NEUTRAL + 1 ? CONTINUOUS_ACTION_MAX : 0.f;
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
            jump == ACT_NON_NEUTRAL,
            boost == ACT_NON_NEUTRAL,
            powerslide == ACT_NON_NEUTRAL
        };
    }
};

static_assert(sizeof(DiscreteControls) == ACT_PER_CAR * sizeof(int32_t));

struct ContinuousControls
{
    float horizontal;
    float vertical;
    float throttle;
    float powerslide;
    float boost;
    float airRoll;
    float jump;

    CARL_D CARL_FI static float axis(float action)
    {
        return action < CONTINUOUS_AXIS_MIN ? CONTINUOUS_AXIS_MIN
            : action > CONTINUOUS_ACTION_MAX ? CONTINUOUS_ACTION_MAX : action;
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
            jump >= .5f,
            boost >= .5f,
            powerslide >= .5f
        };
    }
};

static_assert(sizeof(ContinuousControls) == ACT_PER_CAR * sizeof(float));
