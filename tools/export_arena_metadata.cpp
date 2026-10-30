#include <RocketSim.h>
#include <fstream>
#include <iomanip>

int main(int argc,char** argv){
    if(argc!=3)return 2;
    RocketSim::Init(argv[1],true);
    auto* arena=RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
    std::ofstream out(argv[2]);out<<std::setprecision(9);
    auto meshes=RocketSim::GetArenaCollisionShapes(RocketSim::GameMode::SOCCAR);
    for(int mesh=0;mesh<meshes.size();mesh++){
        auto* shape=meshes[mesh];
        const unsigned char *vertices,*indices;int nverts,stride,indexStride,nfaces;
        shape->getMeshInterface()->getLockedReadOnlyVertexIndexBase(&vertices,nverts,stride,&indices,indexStride,nfaces);
        auto& nodes=shape->getOptimizedBvh()->getQuantizedNodeArray();
        auto& subtrees=shape->getOptimizedBvh()->getSubtreeInfoArray();
        std::vector<int> traversal;
        for(int s=0;s<subtrees.size();s++){
            auto& subtree=subtrees[s];
            for(int n=subtree.m_rootNodeIndex;n<subtree.m_rootNodeIndex+subtree.m_subtreeSize;n++)
                if(nodes[n].isLeafNode())traversal.push_back(n);
        }
        if(traversal.size()!=nfaces)return 3;
        auto* info=shape->getTriangleInfoMap();
        for(int rank=0;rank<traversal.size();rank++){
            const auto& node=nodes[traversal[rank]];
            int tri=node.getTriangleIndex();out<<mesh<<' '<<rank;
            auto* ids=reinterpret_cast<const int*>(indices+tri*indexStride);
            for(int v=0;v<3;v++){
                auto* xyz=reinterpret_cast<const float*>(vertices+ids[v]*stride);
                for(int k=0;k<3;k++)out<<' '<<xyz[k]*BT_TO_UU;
            }
            auto* e=info?info->find(tri):nullptr;
            out<<' '<<(e?e->m_edgeV0V1Angle:SIMD_2_PI)<<' '<<(e?e->m_edgeV1V2Angle:SIMD_2_PI)
               <<' '<<(e?e->m_edgeV2V0Angle:SIMD_2_PI)<<' '<<(e?e->m_flags:0);
            for(int v=0;v<3;v++){
                auto* xyz=reinterpret_cast<const float*>(vertices+ids[v]*stride);
                for(int k=0;k<3;k++)out<<' '<<xyz[k];
            }
            out<<'\n';
        }
        shape->getMeshInterface()->unLockReadOnlyVertexBase(0);
    }
    delete arena;
    return out?0:1;
}
