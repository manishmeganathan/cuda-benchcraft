cmake_minimum_required(VERSION 3.20)
project(cuda_benchcraft LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

find_program(NSYS nsys)
if(NSYS)
  message(STATUS "Nsight Systems (nsys) found: ${NSYS}")
else()
  message(STATUS "Nsight Systems (nsys) NOT found; profiling will not be available on benchcraft")
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
    --use_fast_math
    -lineinfo
    -Xptxas=-v
  >
)

target_link_libraries(bench_gemm PRIVATE CUDA::cudart CUDA::cublas)

# -- Softmax Bench --------

add_executable(bench_softmax
  softmax/main.cu
  softmax/kernels_naive.cu
)

target_include_directories(bench_softmax PUBLIC common)
target_include_directories(bench_softmax PRIVATE softmax/include)
target_include_directories(bench_softmax PRIVATE ${CMAKE_SOURCE_DIR})

target_compile_options(bench_softmax PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -lineinfo
    -Xptxas=-v
  >
)

target_link_libraries(bench_softmax PRIVATE CUDA::cudart)
