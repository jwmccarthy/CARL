cmake_minimum_required(VERSION 3.20)

project(CARL LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

file(GLOB CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

list(REMOVE_ITEM CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu"
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
    CUDA::cudart
)

target_compile_options(carl PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

# --- Benchmark ---

add_executable(main src/main.cu)

set_target_properties(main PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)

target_link_libraries(main PRIVATE carl)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-attributes>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=174,20044>
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)

# --- Python module ---

option(CARL_BUILD_PYTHON "Build the carl Python module" ON)

if(CARL_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(carl_module python/carl.cu)

    set_target_properties(carl_module PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
        CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME carl
    )

    target_link_libraries(carl_module PRIVATE carl)
endif()
