// gemm/include/kernels.hpp
// Kernel taxonomy + helpers (names, parse, dispatch).

#pragma once
#include <string>
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
inline static const char* kernel_name(KernelKind k) {
  switch (k) {
    case KernelKind::Naive1D: return "Naive1D";
    case KernelKind::Naive2D: return "Naive2D";
    case KernelKind::CuBLAS: return "CuBLAS";
    default: return "Unknown";
  }
}

// Maps string to KernelKind; defaults to Naive1D if no match.
inline static KernelKind parse_kernel(const char* s) {
  for (int i = 0; i < (int)KernelKind::Count; ++i) {
    KernelKind k = static_cast<KernelKind>(i);
    if (std::string(s) == kernel_name(k)) return k;
  }
  return KernelKind::Naive1D;
}

// Returns canonical kernel names in enum order.
inline static void list_kernels() {
  for (int i = 0; i < (int)KernelKind::Count; ++i)
    std::printf("%s\n", kernel_name(static_cast<KernelKind>(i)));
}

// Forward declarations for variant-specific launchers
void launch_naive1d(const float*, const float*, float*, int, int, int, cudaStream_t);
void launch_naive2d(const float*, const float*, float*, int, int, int, cudaStream_t);
void launch_cublas(const float*, const float*, float*, int, int, int, cudaStream_t);

inline static void not_impl(KernelKind k) {
  std::fprintf(stderr, "Kernel '%s' not implemented yet.\n", kernel_name(k));
}

// Unified GEMM Kernel Launcher
inline static void launch_kernel(
    KernelKind kind,            // kind of matmul kernel
    const float* A,             // input matrix A [M * K]
    const float* B,             // input matrix B [K * N]
    float* C,                   // output matrix C [M * N]
    int M, int N, int K,        // matrix size params
    cudaStream_t stream = 0  
) {
  switch (kind) {
    case KernelKind::Naive1D: launch_naive1d(A,B,C,M,N,K,stream); break;
    case KernelKind::Naive2D: launch_naive2d(A,B,C,M,N,K,stream); break;
    case KernelKind::CuBLAS: return launch_cublas(A,B,C,M,N,K,stream);
    default: return not_impl(kind);
  }
}