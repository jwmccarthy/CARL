#pragma once

#include <cstdint>

#include "../RLConstants.cuh"

// --- DLPack types (minimal, no external dependency) ---

enum DLDeviceType { kDLCUDA = 2 };

enum DLDataTypeCode
{
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLBool = 6
};

struct DLDevice
{
    int device_type;
    int device_id;
};

struct DLDataType
{
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};

struct DLTensor
{
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
};

struct DLManagedTensor
{
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(DLManagedTensor*);
};

// --- Observation and action layout ---

// Ball and car vector fields have the same offsets. Canonical state adds a
// ball-touch flag after the observation's per-car fields.
constexpr int OBS_VECTOR_SIZE = 3;
enum ObservedField
{
    OBS_POS = 0,
    OBS_VEL = OBS_POS + OBS_VECTOR_SIZE,
    OBS_ANG = OBS_VEL + OBS_VECTOR_SIZE,
    OBS_FORWARD = OBS_ANG + OBS_VECTOR_SIZE,
    OBS_UP = OBS_FORWARD + OBS_VECTOR_SIZE,
    OBS_BOOST = OBS_UP + OBS_VECTOR_SIZE,
    OBS_ON_GROUND,
    OBS_DEMOED,
    OBS_HAS_FLIPPED,
    OBS_HAS_DOUBLE_JUMPED,
    OBS_IS_BOOSTING,
    STATE_BALL_TOUCH
};

constexpr int OBS_BALL = OBS_FORWARD;
constexpr int OBS_PER_CAR = STATE_BALL_TOUCH;
constexpr int STATE_PER_CAR = STATE_BALL_TOUCH + 1;
constexpr int OBS_BOOST_PADS = 2 * NUM_BOOST_PADS;  // flags + distances
constexpr int OBS_RELATIVE_EGO_BALL = 2 * OBS_VECTOR_SIZE;  // pos + vel
constexpr int OBS_RELATIVE_PER_OTHER_CAR = OBS_RELATIVE_EGO_BALL;
constexpr int OBS_RELATIVE_GOALS = 2 * OBS_VECTOR_SIZE;
constexpr int OBS_EGO_DODGE = 2;  // hasFlipOrJump + seconds remaining
constexpr int OBS_EGO_FLIP_INDEX = -OBS_EGO_DODGE;
constexpr int OBS_EGO_DODGE_TIME_INDEX = OBS_EGO_FLIP_INDEX + 1;

constexpr CARL_HD int observationDim(int nCars)
{
    return OBS_BALL + nCars * OBS_PER_CAR + OBS_BOOST_PADS
        + OBS_RELATIVE_EGO_BALL
        + (nCars - 1) * OBS_RELATIVE_PER_OTHER_CAR
        + OBS_RELATIVE_GOALS + OBS_EGO_DODGE;
}

constexpr CARL_HD int packedStateDim(int nCars)
{
    return OBS_BALL + nCars * STATE_PER_CAR + NUM_BOOST_PADS;
}
