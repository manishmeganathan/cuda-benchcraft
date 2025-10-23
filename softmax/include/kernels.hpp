// softmax/include/kernels.hpp
// Kernel taxonomy + helpers (names, parse, dispatch).

#pragma once
#include <string>
#include <cuda_runtime.h>

enum class KernelKind : int {
    NaiveElem,
    NaiveRow,
    Torch,
    Count
};

// Returns canonical name for a kernel kind
inline static const char* kernel_name(KernelKind k) {
    switch (k) {
        case KernelKind::NaiveElem: return "NaiveElem";
        case KernelKind::NaiveRow: return "NaiveRow";
        case KernelKind::Torch: return "Torch";
        default: return "Unknown";
    }
}

// Maps string to KernelKind; defaults to Naive1D if no match.
inline static KernelKind parse_kernel(const char* s) {
    std::string name = std::string(s);

    if (name == "NaiveElem") return KernelKind::NaiveElem;
    if (name == "NaiveRow") return KernelKind::NaiveRow;
    if (name == "Torch") return KernelKind::Torch;

    return KernelKind::NaiveElem;
}

// Returns canonical kernel names in enum order.
inline static void list_kernels() {
  for (int i = 0; i < (int)KernelKind::Count; ++i)
    std::printf("%s\n", kernel_name(static_cast<KernelKind>(i)));
}


// Forward declarations for variant-specific launchers
void launch_naive_element(const float*, float*, int, int, cudaStream_t);
void launch_naive_row(const float*, float*, int, int, cudaStream_t);
void launch_torch(const float*, float*, int, int, cudaStream_t);

// Unified Softmax Kernel Launcher
inline static void launch_kernel(
    KernelKind kind,    // kind of softmax kernel
    const float* X,     // input matrix X [M * N]
    float* Y,           // output matrix Y [M * N]
    int M, int N,       // matrix size params
    cudaStream_t stream = 0  
) {
  switch (kind) {
    case KernelKind::NaiveElem: launch_naive_element(X,Y,M,N,stream); break;
    case KernelKind::NaiveRow: launch_naive_row(X,Y,M,N,stream); break;
    case KernelKind::Torch: launch_torch(X,Y,M,N,stream); break;

    default: 
      std::fprintf(stderr, "Kernel '%s' not implemented yet.\n", kernel_name(kind));
      return;
  }
}