// include/kernels.hpp
// Kernel taxonomy + helpers (names, parse, dispatch).

#pragma once
#include <cuda_runtime.h>

enum class KernelKind : int {
  Naive1D,
  Naive2D,
//   Coalesced,
//   Unrolled,
//   TiledShared,
//   TiledSharedPadded,
//   RegTile,
//   WarpTile,
//   DoubleBuffer,
//   CpAsync,
//   SplitK,
//   WMMA,
  CuBLAS,
  Count
};

// Returns canonical name for a kernel kind.
const char* kernel_name(KernelKind k);

// Returns canonical kernel names in enum order.
void list_kernels();

// Maps string to KernelKind; defaults to Naive1D if no match.
KernelKind parse_kind(const char* s);

// Unified Kernel Launcher
void launch_kernel(
    KernelKind kind,            // kind of matmul kernel
    const float* A,             // input matrix A [M * K]
    const float* B,             // input matrix B [K * N]
    float* C,                   // output matrix C [M * N]
    int M, int N, int K,        // matrix size params
    cudaStream_t stream = 0  
);