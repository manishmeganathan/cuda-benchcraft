cmake_minimum_required(VERSION 3.20)
project(cuda_benchcraft LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "86;90")
set(TORCH_CUDA_ARCH_LIST "8.6;9.0")

if(NOT DEFINED ENV{LIBTORCH_DIR} OR "$ENV{LIBTORCH_DIR}" STREQUAL "")
  message(FATAL_ERROR
    "Environment variable LIBTORCH_DIR is not defined.\n"
    "Please set it to your LibTorch installation path, e.g.:\n"
    "  export LIBTORCH_DIR=/path/to/libtorch"
  )
endif()

message(STATUS "Expecting LibTorch at: ${LIBTORCH_DIR}")
set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_DIR})
find_package(Torch REQUIRED) 

find_program(NSYS nsys)
if(NSYS)
  message(STATUS "NSight Systems (nsys) found: ${NSYS}")
else()
  message(STATUS "NSight Systems (nsys) NOT found; profiling will not be available on benchcraft")
endif()

# -- GEMM Bench --------

add_executable(bench_gemm
  gemm/main.cu
  gemm/kernels_naive.cu
  gemm/kernels_cublas.cu
)

target_include_directories(bench_gemm PUBLIC common)
target_include_directories(bench_gemm PRIVATE gemm/include)
target_include_directories(bench_gemm PRIVATE ${CMAKE_SOURCE_DIR})

target_compile_options(bench_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    -Xptxas=-v
  >
)

target_link_libraries(bench_gemm PRIVATE CUDA::cudart CUDA::cublas)

# -- Softmax Bench --------

add_executable(bench_softmax
  softmax/main.cu
  softmax/kernels_naive.cu
  softmax/kernels_reduce.cu
  softmax/kernels_torch.cpp
)

target_include_directories(bench_softmax PUBLIC common)
target_include_directories(bench_softmax PRIVATE softmax/include)
target_include_directories(bench_softmax PRIVATE ${CMAKE_SOURCE_DIR})

target_compile_options(bench_softmax PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -lineinfo
    -Xptxas=-v
  >
)

target_link_libraries(bench_softmax PRIVATE CUDA::cudart ${TORCH_LIBRARIES})
