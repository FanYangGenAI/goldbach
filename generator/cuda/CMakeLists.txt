cmake_minimum_required(VERSION 3.18)
project(goldbach_generate LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(goldbach_generate
    src/main.cpp
    src/sample.cpp
    src/primes.cpp
    src/buckets.cpp
    src/pipeline.cpp
    src/gpu_tester.cpp
    cuda/primality.cu
)

target_include_directories(goldbach_generate
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

set_target_properties(goldbach_generate PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

if(MSVC)
    target_compile_options(goldbach_generate PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
else()
    target_compile_options(goldbach_generate PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

target_link_libraries(goldbach_generate PRIVATE CUDA::cudart)
