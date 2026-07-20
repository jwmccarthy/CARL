#pragma once

#include <cstdint>

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

// Per car: 
// pos(3) + vel(3) + ang(3) + forward(3) + up(3) + boost(1)
// + isOnGround + isDemoed + hasFlipped + hasDoubleJumped + isBoosting(5)
constexpr int OBS_PER_CAR = 21;
constexpr int OBS_BALL = 9;  // pos + vel + ang
constexpr int OBS_BOOST_PADS = 2 * 34;  // active flags + distances
constexpr int STATE_PER_CAR = OBS_PER_CAR + 1;  // observation fields + ball touch
