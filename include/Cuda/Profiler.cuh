#pragma once

#include <cstdio>
#include <cstring>

#include "Common.cuh"

struct ScopedCudaProfiler
{
    static constexpr int MaxStages = 16;

    struct Stage
    {
        const char* name = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        float       totalMs = 0.f;
        int         count = 0;
    };

    struct Scope
    {
        Stage* stage = nullptr;

        Scope() = default;
        Scope(Stage* stage) : stage(stage) {}
        Scope(Scope&& other) noexcept : stage(other.stage) { other.stage = nullptr; }

        ~Scope()
        {
            if (!stage) return;

            float ms = 0.f;
            CUDA_CHECK(cudaEventRecord(stage->stop));
            CUDA_CHECK(cudaEventSynchronize(stage->stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, stage->start, stage->stop));

            stage->totalMs += ms;
            stage->count++;
        }
    };

    int   ticks = 1;
    int   simScale = 1;
    bool  running = false;
    Stage stages[MaxStages] = {};
    int   stageCount = 0;

    static ScopedCudaProfiler& get()
    {
        static ScopedCudaProfiler profiler;
        return profiler;
    }

    ~ScopedCudaProfiler()
    {
        if (running) stop();

        for (int i = 0; i < stageCount; i++)
        {
            CUDA_CHECK(cudaEventDestroy(stages[i].start));
            CUDA_CHECK(cudaEventDestroy(stages[i].stop));
        }
    }

    void configure(int ticks, int simScale = 1)
    {
        this->ticks = ticks;
        this->simScale = simScale;
    }

    void start()
    {
        running = true;
    }

    void stop()
    {
        running = false;
        print();
    }

    Scope scope(const char* name)
    {
        Stage* stage = running ? findOrAdd(name) : nullptr;
        if (!stage) return {};

        CUDA_CHECK(cudaEventRecord(stage->start));
        return { stage };
    }

private:
    Stage* findOrAdd(const char* name)
    {
        for (int i = 0; i < stageCount; i++)
        {
            if (std::strcmp(stages[i].name, name) == 0) return &stages[i];
        }

        if (stageCount == MaxStages) return nullptr;

        Stage& stage = stages[stageCount++];
        stage.name = name;

        CUDA_CHECK(cudaEventCreate(&stage.start));
        CUDA_CHECK(cudaEventCreate(&stage.stop));

        return &stage;
    }

    void print() const
    {
        float aggregateMs = 0.f;

        std::printf("%-14s %10s\n", "stage", "us/tick");
        std::printf("-------------------------\n");

        for (int i = 0; i < stageCount; i++)
        {
            const Stage& stage = stages[i];
            aggregateMs += stage.totalMs;

            const float usPerTick = stage.count > 0
                ? stage.totalMs * 1000.f / stage.count
                : 0.f;
            std::printf("%-14s %10.3f\n", stage.name, usPerTick);
        }

        std::printf("-------------------------\n");
        std::printf(
            "%-14s %10.3f\n",
            "aggregate",
            aggregateMs * 1000.f / (float)ticks);

        double rate = aggregateMs > 0.f
            ? (double)ticks * simScale * 1000.0 / aggregateMs
            : 0.0;
        const char* unit = "";

        if (rate >= 1e6)
        {
            rate /= 1e6;
            unit = "M";
        }
        else if (rate >= 1e3)
        {
            rate /= 1e3;
            unit = "K";
        }

        char rateBuf[16];
        std::snprintf(rateBuf, sizeof(rateBuf), "%.1f%s", rate, unit);
        std::printf("%-14s %10s\n", "sim ticks/s", rateBuf);
    }
};

#define PROFILE(name, code) do { \
    auto _cudaProfileScope = ScopedCudaProfiler::get().scope(name); \
    code; \
} while (0)
