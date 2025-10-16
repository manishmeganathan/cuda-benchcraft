cmake_minimum_required(VERSION 3.20)
project(cuda_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

find_program(NCU_EXE ncu)
if(NCU_EXE)
  message(STATUS "Nsight Compute found: ${NCU_EXE}")
else()
  message(STATUS "Nsight Compute (ncu) NOT found; profiling disabled until installed.")
endif()

add_executable(gemm_bench
  src/main.cu
  src/utils.cu
  src/kernels.cu
  src/kernels_naive.cu
  src/kernels_cublas.cu
)

target_include_directories(gemm_bench PRIVATE include)

target_compile_options(gemm_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -lineinfo
    -Xptxas=-v
  >
)

target_link_libraries(gemm_bench PRIVATE CUDA::cudart CUDA::cublas)