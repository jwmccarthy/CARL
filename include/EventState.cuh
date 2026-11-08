#pragma once

#include "Cuda/DeviceArray.cuh"
#include "State/Events.cuh"

#include <vector>
#include <cstdint>

class EventState
{
    cudaStream_t stream;

    bool captureEvents;
    bool captureTouches;
    
    int nSim;
    int nCars;
    int capacity = 0;

    std::vector<cudaStream_t> consumers;

    void waitForConsumers();

public:
    DeviceArray<int> carFlags;
    DeviceArray<int> ballFlags;
    DeviceArray<int> carSurfaces;
    DeviceArray<int> carSurfaceContacted;
    DeviceArray<int> carSurfaceLeft;
    DeviceArray<int> ballSurfaces;
    DeviceArray<int> ballSurfaceEntered;
    DeviceArray<int> ballSurfaceLeft;
    DeviceArray<int> goalSide;

    DeviceArray<float> ballSpeedChange;
    DeviceArray<float> ballSpeedBefore;

    DeviceArray<int> touchSurfaces;
    DeviceArray<int> touchTickSurfaces;

    DeviceArray<float> touchTickSpeed;
    DeviceArray<float> touchTickSpeedBefore;

    int64_t carShape[2] = {};
    int64_t ballShape[1] = {};
    int64_t touchTickShape[3] = {};

    EventState(
        int nSim, int nCars, int frameskip, cudaStream_t stream,
        bool captureEvents, bool captureTouches);

    void resize(int frameskip);
    void clear();
    void reset();
    void acquire(intptr_t consumerStream);

    EventBuffer buffer() const;
    EventBuffer tickBuffer(int tick) const;

    bool enabled() const { return captureEvents; }
    bool touchesEnabled() const { return captureTouches; }
};
