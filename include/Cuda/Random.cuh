#pragma once

#include <cstdint>

#include "Common.cuh"

CARL_HD CARL_FI uint32_t hash32(uint32_t value)
{
    value = ((value >> 16) ^ value) * 0x45d9f3bu;
    value = ((value >> 16) ^ value) * 0x45d9f3bu;
    return (value >> 16) ^ value;
}

CARL_HD CARL_FI int randomIndex(uint32_t seed, int count)
{
    return (int)(hash32(seed) % (uint32_t)count);
}
