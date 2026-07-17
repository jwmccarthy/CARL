#pragma once

#include <cstdint>

#include "State/Controls.cuh"

// --- DLPack types (minimal, no external dependency) ---

enum DLDeviceType   { kDLCUDA = 2 };
enum DLDataTypeCode { kDLFloat = 2, kDLUInt = 1, kDLInt = 0 };

struct DLDevice { int device_type; int device_id; };

struct DLDataType { uint8_t code; uint8_t bits; uint16_t lanes; };

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
