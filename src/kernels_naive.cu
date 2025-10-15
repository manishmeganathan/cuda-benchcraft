// src/kernels_naive.cu
// Naive matmul kernels (row-major): C = A(M x K) * B(K x N)

#include "kernels.hpp"

// Kernel function to compute Cij with a 1D grid.
// Rows and columns are calculated with stride linearization
__global__ void matmul_naive1d(const float* A, const float* B, float* C, int M, int N, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Overlaunch control
  if (idx >= M * N) return;

  // Determine output element row and column
  int row = idx / N; 
  int col = idx % N;

  // Accumulate the dot product of Ai and Bj
  float dot = 0.f;
  for (int k = 0; k < K; ++k) {
    dot += A[row * K + k] * B[k * N + col];
  }

  // Set the dot product to Cij
  C[row * N + col] = dot;
}

void launch_naive1d(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s){
  dim3 block(256);
  dim3 grid((M * N + block.x - 1)/block.x);

  matmul_naive1d<<<grid, block, 0, s>>>(A,B,C,M,N,K);
}

// Kernel function to compute Cij with a 2D grid.
// Rows and columns are directly mapped to thread blocks
__global__ void matmul_naive2d(const float* A, const float* B, float* C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Overlaunch control
  if (row >= M || col >= N) return;

  // Accumulate the dot product of Ai and Bj
  float dot = 0.f;
  for (int k = 0; k < K; ++k) {
    dot += A[row * K + k] * B[k * N + col];
  }

  // Set the dot product to Cij
  C[row * N + col] = dot;
}

void launch_naive2d(const float* A, const float* B,float* C, int M, int N, int K, cudaStream_t s){
  dim3 block(16,16); // 256 blocks
  dim3 grid((N + 15)/16, (M + 15)/16);
  
  matmul_naive2d<<<grid, block, 0, s>>>(A,B,C,M,N,K);
}
