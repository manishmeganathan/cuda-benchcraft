// softmax/include/kernels.hpp
// Kernel taxonomy + helpers (names, parse, dispatch).

#pragma once
#include <string>
#include <cuda_runtime.h>

enum class KernelKind : int {
    Brute,
    Naive,
    AtomicReduce,
    TreeReduce,
    // Online
    // LoopUnroll
    // WideAccess
    // ShuffleSync
    Torch,
    Count
};

// Returns canonical name for a kernel kind
inline static const char* kernel_name(KernelKind k) {
    switch (k) {
        case KernelKind::Brute: return "Brute";
        case KernelKind::Naive: return "Naive";
        case KernelKind::AtomicReduce: return "AtomicReduce";
        case KernelKind::TreeReduce: return "TreeReduce";
        case KernelKind::Torch: return "Torch";
        default: return "Unknown";
    }
}

// Maps string to KernelKind; defaults to Naive if no match.
inline static KernelKind parse_kernel(const char* s) {
    std::string name = std::string(s);

    if (name == "Brute") return KernelKind::Brute;
    if (name == "Naive") return KernelKind::Naive;
    if (name == "AtomicReduce") return KernelKind::AtomicReduce;
    if (name == "TreeReduce") return KernelKind::TreeReduce;
    if (name == "Torch") return KernelKind::Torch;

    return KernelKind::Naive;
}

// Returns canonical kernel names in enum order.
inline static void list_kernels() {
  for (int i = 0; i < (int)KernelKind::Count; ++i)
    std::printf("%s\n", kernel_name(static_cast<KernelKind>(i)));
}

// Forward declarations for variant-specific launchers
void launch_brute(const float*, float*, int, int, cudaStream_t);
void launch_naive(const float*, float*, int, int, cudaStream_t);
void launch_atomic_reduce(const float*, float*, int, int, cudaStream_t);
void launch_tree_reduce(const float*, float*, int, int, cudaStream_t);

// Softmax Torch Kernel Launcher
// Return the average execution time in millisecods
float launch_torch(
  int iters,
  const float* X,     // input matrix X [M * N]
  float* Y,           // output matrix Y [M * N]
  int M, int N        // matrix size params
);

// Unified Softmax Kernel Launcher
// Can be used for all kernels except Torch
inline static void launch_kernel(
    KernelKind kind,    // kind of softmax kernel
    const float* X,     // input matrix X [M * N]
    float* Y,           // output matrix Y [M * N]
    int M, int N,       // matrix size params
    cudaStream_t stream = 0  
) {
  switch (kind) {
    case KernelKind::Brute: launch_brute(X,Y,M,N,stream); break;
    case KernelKind::Naive: launch_naive(X,Y,M,N,stream); break;
    case KernelKind::AtomicReduce: launch_atomic_reduce(X,Y,M,N,stream); break;
    case KernelKind::TreeReduce: launch_tree_reduce(X,Y,M,N,stream); break;

    case KernelKind::Torch: 
      std::fprintf(stderr, "Kernel 'Torch' cannot be called from unified launcher.\n");
      return;

    default: 
      std::fprintf(stderr, "Kernel '%s' not implemented yet.\n", kernel_name(kind));
      return;
  }
}