cmake_minimum_required(VERSION 3.24)
project(cuda_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET PROPERTY CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_bench
  src/main.cu
  src/utils.cu
  src/kernels.cu
  src/kernels_naive.cu
  src/kernels_cublas.cu
)

target_include_directories(gemm_bench PRIVATE include)

# Handy CUDA flags: fast math, line info, and show ptxas register usage.
target_compile_options(gemm_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -lineinfo
    -Xptxas=-v
  >
)

target_link_libraries(gemm_bench PRIVATE CUDA::cudart CUDA::cublas)