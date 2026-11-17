#pragma once

#include <stdexcept>
#include <string>

#include "Common.cuh"

// The owner must finish outstanding launches before releasing captured buffers.
class CudaGraph
{
#ifndef CARL_USE_HIP
    cudaGraphExec_t executable = nullptr;

    static void require(cudaError_t status, const char* operation)
    {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
#endif
    size_t nodes = 0;

public:
    CudaGraph() = default;
    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    ~CudaGraph() { clear(); }

    size_t nodeCount() const { return nodes; }

    void clear()
    {
#ifndef CARL_USE_HIP
        if (executable) cudaGraphExecDestroy(executable);
        executable = nullptr;
#endif
        nodes = 0;
    }

    template<typename Submit>
    void replay(cudaStream_t stream, Submit submit)
    {
#ifdef CARL_USE_HIP
        throw std::runtime_error("native_cuda_graphs requires the CUDA backend");
#else
        if (!executable)
        {
            cudaGraph_t graph = nullptr;
            require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                "begin native physics capture");
            try
            {
                submit();
            }
            catch (...)
            {
                cudaStreamEndCapture(stream, &graph);
                if (graph) cudaGraphDestroy(graph);
                throw;
            }

            const cudaError_t captured = cudaStreamEndCapture(stream, &graph);
            if (captured != cudaSuccess)
            {
                if (graph) cudaGraphDestroy(graph);
                require(captured, "end native physics capture");
            }

            const cudaError_t counted = cudaGraphGetNodes(graph, nullptr, &nodes);
            const cudaError_t instantiated = counted == cudaSuccess
                ? cudaGraphInstantiate(&executable, graph, 0)
                : counted;
            cudaGraphDestroy(graph);
            if (instantiated != cudaSuccess)
            {
                clear();
                require(instantiated, "instantiate native physics graph");
            }
        }

        require(cudaGraphLaunch(executable, stream), "launch native physics graph");
#endif
    }
};
