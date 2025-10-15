// src/kernel.cu
// Kernel dispatcher + helpers (names, parse, dispatch)

#include <cstdio>
#include <string>

#include "kernels.hpp"

// Forward declarations for variant-specific launchers
void launch_naive1d(const float*, const float*, float*, int, int, int, cudaStream_t);
void launch_naive2d(const float*, const float*, float*, int, int, int, cudaStream_t);
void launch_cublas(const float*, const float*, float*, int, int, int, cudaStream_t);

// Returns canonical name for a kernel kind.
const char* kernel_name(KernelKind k) {
  switch (k) {
    case KernelKind::Naive1D: return "Naive1D";
    case KernelKind::Naive2D: return "Naive2D";
    //case KernelKind::Coalesced: return "Coalesced";
    // case KernelKind::Unrolled: return "Unrolled";
    // case KernelKind::TiledShared: return "TiledShared";
    // case KernelKind::TiledSharedPadded: return "TiledSharedPadded";
    // case KernelKind::RegTile: return "RegTile";
    // case KernelKind::WarpTile: return "WarpTile";
    // case KernelKind::DoubleBuffer: return "DoubleBuffer";
    // case KernelKind::CpAsync: return "CpAsync";
    // case KernelKind::SplitK: return "SplitK";
    // case KernelKind::WMMA: return "WMMA";
    case KernelKind::CuBLAS: return "CuBLAS";
    default: return "Unknown";
  }
}

// Returns canonical kernel names in enum order.
void list_kernels() {
  for (int i = 0; i < (int)KernelKind::Count; ++i)
    std::printf("%s\n", kernel_name(static_cast<KernelKind>(i)));
}

// Maps string to KernelKind; defaults to Naive1D if no match.
KernelKind parse_kind(const char* s) {
  for (int i = 0; i < (int)KernelKind::Count; ++i) {
    KernelKind k = static_cast<KernelKind>(i);
    if (std::string(s) == kernel_name(k)) return k;
  }
  return KernelKind::Naive1D;
}

inline static void not_impl(KernelKind k) {
  std::fprintf(stderr, "Kernel '%s' not implemented yet.\n", kernel_name(k));
}

// Launch switchboard; wires kinds to per-variant launchers.
void launch_kernel(
    KernelKind kind,
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
  switch (kind) {
    case KernelKind::Naive1D: launch_naive1d(A,B,C,M,N,K,stream); break;
    case KernelKind::Naive2D: launch_naive2d(A,B,C,M,N,K,stream); break;
    case KernelKind::CuBLAS: return launch_cublas(A,B,C,M,N,K,stream);
    default: return not_impl(kind);
  }
}

