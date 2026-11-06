#include <fstream>
#include <iomanip>
#include <vector>

#include <RocketSim.h>

int main(int argc, char** argv)
{
    if (argc != 3) return 2;

    RocketSim::Init(argv[1], true);
    auto* arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
    std::ofstream out(argv[2]);
    out << std::setprecision(9);

    auto meshes = RocketSim::GetArenaCollisionShapes(RocketSim::GameMode::SOCCAR);

    for (int mesh = 0; mesh < meshes.size(); mesh++)
    {
        auto* shape = meshes[mesh];
        const unsigned char* vertices;
        const unsigned char* indices;
        int nVerts;
        int vertexStride;
        int indexStride;
        int nFaces;

        shape->getMeshInterface()->getLockedReadOnlyVertexIndexBase(
            &vertices, nVerts, vertexStride, &indices, indexStride, nFaces);

        auto& nodes = shape->getOptimizedBvh()->getQuantizedNodeArray();
        auto& subtrees = shape->getOptimizedBvh()->getSubtreeInfoArray();
        std::vector<int> traversal;

        // Record leaf nodes in native BVH traversal order
        for (int i = 0; i < subtrees.size(); i++)
        {
            auto& subtree = subtrees[i];
            const int end = subtree.m_rootNodeIndex + subtree.m_subtreeSize;

            for (int node = subtree.m_rootNodeIndex; node < end; node++)
            {
                if (nodes[node].isLeafNode()) traversal.push_back(node);
            }
        }

        if (traversal.size() != nFaces) return 3;

        auto* info = shape->getTriangleInfoMap();

        for (int rank = 0; rank < traversal.size(); rank++)
        {
            const auto& node = nodes[traversal[rank]];
            const int triangle = node.getTriangleIndex();
            const auto* vertexIndices = reinterpret_cast<const int*>(indices + triangle * indexStride);
            out << mesh << ' ' << rank;

            // Write the triangle in Unreal units for matching against the OBJ
            for (int vertex = 0; vertex < 3; vertex++)
            {
                const auto* xyz = reinterpret_cast<const float*>(
                    vertices + vertexIndices[vertex] * vertexStride);

                for (int axis = 0; axis < 3; axis++) out << ' ' << xyz[axis] * BT_TO_UU;
            }

            auto* edge = info ? info->find(triangle) : nullptr;
            out << ' ' << (edge ? edge->m_edgeV0V1Angle : SIMD_2_PI)
                << ' ' << (edge ? edge->m_edgeV1V2Angle : SIMD_2_PI)
                << ' ' << (edge ? edge->m_edgeV2V0Angle : SIMD_2_PI)
                << ' ' << (edge ? edge->m_flags : 0);

            // Preserve native Bullet-space vertices for collision queries
            for (int vertex = 0; vertex < 3; vertex++)
            {
                const auto* xyz = reinterpret_cast<const float*>(
                    vertices + vertexIndices[vertex] * vertexStride);

                for (int axis = 0; axis < 3; axis++) out << ' ' << xyz[axis];
            }

            out << '\n';
        }

        shape->getMeshInterface()->unLockReadOnlyVertexBase(0);
    }

    delete arena;
    return out ? 0 : 1;
}
