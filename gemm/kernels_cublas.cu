// gemm/kernels_cublas.cu
// cuBLAS SGEMM reference (row-major via swapped args): C = A*B

#include "kernels.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

void launch_cublas(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
  static cublasHandle_t handle = nullptr;
  if (!handle) cublasCreate(&handle);
  cublasSetStream(handle, s);

  // Simple GEMM with scalar transforms
  const float alpha = 1.f, beta = 0.f;

  // cuBLAS expects matrices in col-major form but our code uses row-major.
  // We can simply pass the same matrices but with column leading dimensions to get desired results
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K,
              &alpha,
              B, N,
              A, K,
              &beta,
              C, N);
}