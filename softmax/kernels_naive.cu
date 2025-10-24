// softmax/kernels_naive.cu
// Naive softmax kernels

#include "kernels.hpp"
#include <math_constants.h> 

// Kernel function to compute the softmax activation for a matrix [M * N]
// Each threads computes 1 element of the output matrix but independently 
// computes the max and sum for the softmax function for each element
__global__ void softmax_naive(const float* X, float* Y, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Overlaunch control
    if (row >= M || col >= N) return;

    // Define max and sum for the row
    float maxval = -CUDART_INF_F;
    float sumval = 0.f;

    // Find the maximum value in the row
    for (int i = 0; i < N; i++) {
        maxval = fmaxf(maxval, X[row * N + i]);
    }

    // Aggregate the sum of exponential values in the row
    for (int i = 0; i < N; i++) {
        // Deduct the max value from each value to
        // force the exponent value between 0 and 1
        sumval += expf(X[row * N + i] - maxval); 
    }

    // Determine output index, used twice in next line
    int idx = row * N + col;
    // Divide the value for index with the aggregated sum after
    // deducting the max value for the same reason as above
    Y[idx] = expf(X[idx] - maxval) / sumval;
}

void launch_naive(const float* X, float* Y, int M, int N, cudaStream_t s){
  dim3 block(16,16); // 256 threads
  dim3 grid((M + 15)/16, (N + 15)/16);

  softmax_naive<<<grid, block, 0, s>>>(X,Y,M,N);
}

// NaiveRow 
//
// Kernel function to compute the softmax activation for a matrix [M * N]
// Each thread computes all elements in a row of output matrix. 
//
// Optimizations (Vs Naive):
// - Reduces overhead of computing the sum and max for each element and instead
//   shares the computed values for all elements in the row
// - Improves performance but at the cost of reduced row-level parallelism for wide matrices
__global__ void softmax_naive_row(const float* X, float* Y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Overlaunch control
    if (row >= M) return;

    // Define max and sum for the row
    float maxval = -CUDART_INF_F;
    float sumval = 0.f;

    // Find the maximum value in the row
    for (int i = 0; i < N; i++) {
        maxval = fmaxf(maxval, X[row * N + i]);
    }

    // Aggregate the sum of exponential values
    for (int i = 0; i < N; i++) {
        // Deduct the max value from each value to
        // force the exponent value between 0 and 1
        sumval += expf(X[row * N + i] - maxval); 
    }

    // Set output for all elements in the row
    for (int i = 0; i < N; i++) {
        // Determine output index, used twice in next line
        int idx = row * N + i;

        // Divide the value for index with the aggregated sum after
        // deducting the max value for the same reason as above
        Y[idx] = expf(X[idx] - maxval) / sumval;
    } 
}

void launch_naive_row(const float* X, float* Y, int M, int N, cudaStream_t s){
  dim3 block(256);
  dim3 grid((M * block.x - 1)/block.x);

  softmax_naive_row<<<grid, block, 0, s>>>(X,Y,M,N);
}