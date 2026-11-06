cmake_minimum_required(VERSION 3.24)

set(CARL_GPU_BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE CARL_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
if(CARL_GPU_BACKEND STREQUAL "HIP")
    project(CARL LANGUAGES CXX HIP)
    find_package(hip REQUIRED CONFIG HINTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/cmake/hip")
    find_package(hipcub REQUIRED CONFIG)
elseif(CARL_GPU_BACKEND STREQUAL "CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    project(CARL LANGUAGES CXX CUDA)
    message(STATUS "[carl] CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    find_package(CUDAToolkit 12.8 REQUIRED)
    if(NOT CUDAToolkit_VERSION VERSION_LESS 12.9)
        message(FATAL_ERROR "CARL requires CUDA 12.8, found ${CUDAToolkit_VERSION}")
    endif()
    message(STATUS "[carl] CUDAToolkit: ${CUDAToolkit_VERSION} (${CUDAToolkit_BIN_DIR})")
else()
    message(FATAL_ERROR "CARL_GPU_BACKEND must be CUDA or HIP")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(carl STATIC ${CORE_SOURCES})

if(CARL_GPU_BACKEND STREQUAL "HIP")
    set(HIP_SOURCES
        ${CORE_SOURCES}
        "${CMAKE_CURRENT_SOURCE_DIR}/python/carl.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/parity_harness.cu")
    list(FILTER HIP_SOURCES INCLUDE REGEX "\\.cu$")
    set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

set_target_properties(carl PROPERTIES
    CXX_STANDARD              20
    CXX_STANDARD_REQUIRED     YES
    CUDA_STANDARD             20
    CUDA_STANDARD_REQUIRED    YES
    CUDA_SEPARABLE_COMPILATION ON
    HIP_STANDARD              20
    HIP_STANDARD_REQUIRED     YES
    POSITION_INDEPENDENT_CODE  ON
)

target_include_directories(carl PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(CARL_GPU_BACKEND STREQUAL "HIP")
    target_compile_definitions(carl PUBLIC CARL_USE_HIP=1)
    target_link_libraries(carl PUBLIC hip::host hip::hipcub)
    target_compile_options(carl PUBLIC $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
    target_link_options(carl PUBLIC $<$<LINK_LANGUAGE:HIP>:-fgpu-rdc>)
else()
    target_link_libraries(carl PUBLIC CUDA::cudart_static)
endif()

target_compile_options(carl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CXX>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# --- Python module ---

option(CARL_BUILD_PYTHON "Build the carl Python module" ON)

if(CARL_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(carl_module python/carl.cu)

    set_target_properties(carl_module PROPERTIES
        CUDA_STANDARD             20
        CUDA_STANDARD_REQUIRED    YES
        CUDA_SEPARABLE_COMPILATION ON
        HIP_STANDARD              20
        HIP_STANDARD_REQUIRED     YES
        OUTPUT_NAME               _carl
    )

    target_link_libraries(carl_module PRIVATE carl)
endif()

# --- Optional RocketSim tools ---

option(CARL_BUILD_PARITY_HARNESS "Build the CARL vs RocketSim parity harness" OFF)
option(CARL_BUILD_ARENA_METADATA "Build the RocketSim arena metadata exporter" OFF)

set(CARL_HAS_RLGYM_MESHES OFF)
if(CARL_BUILD_ARENA_METADATA)
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c
                "from importlib.metadata import distribution; p = distribution('rlgym-rocket-league').locate_file('rlgym/rocket_league/sim/collision_meshes/soccar'); assert p.is_dir() and any(p.glob('*.cmf'))"
            RESULT_VARIABLE CARL_RLGYM_RESULT
            ERROR_QUIET
        )
        if(CARL_RLGYM_RESULT STREQUAL "0")
            set(CARL_HAS_RLGYM_MESHES ON)
        endif()
    endif()

    if(NOT CARL_HAS_RLGYM_MESHES)
        message(STATUS "[carl] RLGym meshes unavailable; skipping metadata exporter and using default arena setup")
    endif()
endif()

if(CARL_BUILD_PARITY_HARNESS OR CARL_HAS_RLGYM_MESHES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RocketSim/CMakeLists.txt")
        message(FATAL_ERROR "RocketSim is missing. Run git submodule update --init --recursive")
    endif()

    add_subdirectory(third_party/RocketSim)
endif()

if(CARL_HAS_RLGYM_MESHES)
    add_executable(carl_export_arena_metadata tools/export_arena_metadata.cpp)
    set_target_properties(carl_export_arena_metadata PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
    )
    target_include_directories(carl_export_arena_metadata PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RocketSim/src"
    )
    target_link_libraries(carl_export_arena_metadata PRIVATE RocketSim)
endif()

if(CARL_BUILD_PARITY_HARNESS)
    if(NOT DEFINED ROCKETSIM_COLLISION_MESH_ROOT)
        message(FATAL_ERROR
            "CARL_BUILD_PARITY_HARNESS requires ROCKETSIM_COLLISION_MESH_ROOT")
    endif()

    add_executable(carl_parity_harness tools/parity_harness.cu)

    set_target_properties(carl_parity_harness PROPERTIES
        CXX_STANDARD              20
        CXX_STANDARD_REQUIRED     YES
        CUDA_STANDARD             20
        CUDA_STANDARD_REQUIRED    YES
        CUDA_SEPARABLE_COMPILATION ON
        HIP_STANDARD              20
        HIP_STANDARD_REQUIRED     YES
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
    )
endif()
