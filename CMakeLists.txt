cmake_minimum_required(VERSION 3.20)

project(CARL LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

file(GLOB CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(carl STATIC ${CORE_SOURCES})

set_target_properties(carl PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(carl PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(carl PUBLIC
    CUDA::cudart_static
)

target_compile_options(carl PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

# --- Python module ---

option(CARL_BUILD_PYTHON "Build the carl Python module" ON)

if(CARL_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(carl_module python/carl.cu)

    set_target_properties(carl_module PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
        CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME _carl
    )

    target_link_libraries(carl_module PRIVATE carl)
endif()

# --- Physics parity harness (optional) ---

option(CARL_BUILD_PARITY_HARNESS "Build the CARL vs RocketSim parity harness" OFF)

if(CARL_BUILD_PARITY_HARNESS)
    if(NOT DEFINED ROCKETSIM_COLLISION_MESH_ROOT)
        message(FATAL_ERROR
            "CARL_BUILD_PARITY_HARNESS requires ROCKETSIM_COLLISION_MESH_ROOT")
    endif()

    add_subdirectory(third_party/RocketSim)

    add_executable(carl_parity_harness tools/parity_harness.cu)

    set_target_properties(carl_parity_harness PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_include_directories(carl_parity_harness PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RocketSim/src"
    )

    target_link_libraries(carl_parity_harness PRIVATE
        carl
        RocketSim
    )

    target_compile_definitions(carl_parity_harness PRIVATE
        ROCKETSIM_COLLISION_MESH_ROOT="${ROCKETSIM_COLLISION_MESH_ROOT}"
    )

    target_compile_options(carl_parity_harness PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-w>
        $<$<COMPILE_LANGUAGE:CUDA>:-w>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
    )
endif()
