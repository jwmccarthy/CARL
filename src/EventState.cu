#include "EventState.cuh"

#include <stdexcept>
#include <algorithm>

static void orderStreams(cudaStream_t consumer, cudaStream_t producer)
{
    if (consumer == producer) return;

    cudaEvent_t ready;
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ready, producer));
    CUDA_CHECK(cudaStreamWaitEvent(consumer, ready, 0));
    CUDA_CHECK(cudaEventDestroy(ready));
}

void EventState::acquire(intptr_t consumerStream)
{
    const auto consumer = reinterpret_cast<cudaStream_t>(consumerStream);
    orderStreams(consumer, stream);

    if (std::find(consumers.begin(), consumers.end(), consumer) == consumers.end())
    {
        consumers.push_back(consumer);
    }
}

void EventState::waitForConsumers()
{
    for (cudaStream_t consumer : consumers)
    {
        orderStreams(stream, consumer);
    }

    consumers.clear();
}

EventState::EventState(
    int nSim, int nCars, int frameskip, cudaStream_t stream,
    bool captureEvents, bool captureTouches)
    : stream(stream)
    , captureEvents(captureEvents)
    , captureTouches(captureTouches)
    , nSim(nSim)
    , nCars(nCars)
{
    resize(frameskip);
}

void EventState::resize(int frameskip)
{
    if (frameskip < 1 || nCars < 1 || nSim < 1)
    {
        throw std::invalid_argument("invalid event frameskip capacity");
    }

    if (frameskip == capacity) return;

    waitForConsumers();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int totalCars = nSim * nCars;

    if (captureEvents && !carFlags.ptr)
    {
        carFlags = DeviceArray<int>(totalCars);
        carSurfaceContacted = DeviceArray<int>(totalCars);
        carSurfaceLeft = DeviceArray<int>(totalCars);
        ballFlags = DeviceArray<int>(nSim);
        ballSurfaceEntered = DeviceArray<int>(nSim);
        ballSurfaceLeft = DeviceArray<int>(nSim);
        goalSide = DeviceArray<int>(nSim);
        ballSpeedChange = DeviceArray<float>(nSim);
        ballSpeedBefore = DeviceArray<float>(nSim);

        if (!carSurfaces.ptr)
        {
            carSurfaces = DeviceArray<int>(totalCars);
            ballSurfaces = DeviceArray<int>(nSim);

            const size_t carBytes = totalCars * sizeof(int);
            const size_t ballBytes = nSim * sizeof(int);

            CUDA_CHECK(cudaMemsetAsync(carSurfaces.ptr, 0xFF, carBytes, stream));
            CUDA_CHECK(cudaMemsetAsync(ballSurfaces.ptr, 0xFF, ballBytes, stream));
        }
    }

    if (captureTouches && !touchSurfaces.ptr)
    {
        touchSurfaces = DeviceArray<int>(totalCars);
    }

    if (captureTouches)
    {
        touchTickSurfaces = DeviceArray<int>(totalCars * frameskip);
        touchTickSpeed = DeviceArray<float>(totalCars * frameskip);
        touchTickSpeedBefore = DeviceArray<float>(totalCars * frameskip);
    }

    capacity = frameskip;

    carShape[0] = nSim;
    carShape[1] = nCars;

    ballShape[0] = nSim;
    touchTickShape[0] = frameskip;
    touchTickShape[1] = nSim;
    touchTickShape[2] = nCars;
    clear();
}

void EventState::clear()
{
    waitForConsumers();

    if (captureEvents)
    {
        const size_t carBytes = carFlags.count * sizeof(int);
        const size_t ballBytes = ballFlags.count * sizeof(int);
        const size_t speedBytes = ballSpeedChange.count * sizeof(float);

        CUDA_CHECK(cudaMemsetAsync(carFlags.ptr, 0, carBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(carSurfaceContacted.ptr, 0, carBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(carSurfaceLeft.ptr, 0, carBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ballFlags.ptr, 0, ballBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ballSurfaceEntered.ptr, 0, ballBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ballSurfaceLeft.ptr, 0, ballBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(goalSide.ptr, 0, ballBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ballSpeedChange.ptr, 0, speedBytes, stream));
    }

    if (captureTouches)
    {
        CUDA_CHECK(cudaMemsetAsync(touchSurfaces.ptr, 0, nSim * nCars * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(touchTickSurfaces.ptr, 0, touchTickSurfaces.count * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(touchTickSpeed.ptr, 0, touchTickSpeed.count * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(touchTickSpeedBefore.ptr, 0, touchTickSpeedBefore.count * sizeof(float), stream));
    }
}

void EventState::reset()
{
    clear();
    if (!captureEvents) return;

    const size_t carBytes = carSurfaces.count * sizeof(int);
    const size_t ballBytes = ballSurfaces.count * sizeof(int);

    CUDA_CHECK(cudaMemsetAsync(carSurfaces.ptr, 0xFF, carBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(ballSurfaces.ptr, 0xFF, ballBytes, stream));
}

EventBuffer EventState::buffer() const
{
    EventBuffer result;
    result.carFlags = carFlags.ptr;
    result.ballFlags = ballFlags.ptr;
    result.carSurfaces = carSurfaces.ptr;
    result.carSurfaceContacted = carSurfaceContacted.ptr;
    result.carSurfaceLeft = carSurfaceLeft.ptr;
    result.ballSurfaces = ballSurfaces.ptr;
    result.ballSurfaceEntered = ballSurfaceEntered.ptr;
    result.ballSurfaceLeft = ballSurfaceLeft.ptr;
    result.goalSide = goalSide.ptr;
    result.ballSpeedChange = ballSpeedChange.ptr;
    result.ballSpeedBefore = ballSpeedBefore.ptr;

    result.touches = {touchSurfaces.ptr};

    result.nSim = nSim;
    result.nCars = nCars;
    
    return result;
}

EventBuffer EventState::tickBuffer(int tick) const
{
    EventBuffer result = buffer();
    if (captureTouches)
    {
        const int offset = tick * nSim * nCars;
        result.touches.tickSurfaces = touchTickSurfaces.ptr + offset;
        result.touches.tickSpeedBefore = touchTickSpeedBefore.ptr + offset;
        result.touchSpeed = touchTickSpeed.ptr + offset;
    }
    return result;
}
