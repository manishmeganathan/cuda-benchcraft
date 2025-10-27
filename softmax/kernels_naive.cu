// softmax/kernels_naive.cu
// Naive & brute-force softmax kernels

#include "kernels.hpp"
#include <math_constants.h> 

// Device function to compute max value in a row of floats
__device__ float naive_row_max(const float* X, float* Y, int row, int width) {
    float maxval = -CUDART_INF_F;

    // Find the maximum value in the row
    for (int i = 0; i < width; i++) {
        maxval = fmaxf(maxval, X[row * width + i]);
    }

    return maxval;
}

// Device function to compute the sum(exp) of a row floats
// Requires the max value in the row to correct against numeric overflow
__device__ float naive_row_sum(const float* X, float* Y, int row, int width, float max) {
    float sumval = 0.f;

    // Aggregate the sum of exponential values in the row
    for (int i = 0; i < width; i++) {
        // Deduct the max value from each value to
        // force the exponent value between 0 and 1
        sumval += expf(X[row * width + i] - max); 
    }

    return sumval;
}

// Brute
//
// Kernel function to compute the softmax activation for a matrix [M * N]
// Each threads computes 1 element of the output matrix but independently 
// computes the max and sum for the softmax function for each element
//
// This is a brute-force approach to parallelising softmax
__global__ void softmax_brute(const float* X, float* Y, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Overlaunch control
    if (row >= M || col >= N) return;

    // Determine max and sum for the row
    float maxval = naive_row_max(X, Y, row, N);
    float sumval = naive_row_sum(X, Y, row, N, maxval);

    // Determine output index, used twice in next line
    int idx = row * N + col;
    // Divide the value for index with the aggregated sum after
    // deducting the max value for the same reason as above
    Y[idx] = expf(X[idx] - maxval) / sumval;
}

// Brute Kernel Launcher
void launch_brute(const float* X, float* Y, int M, int N, cudaStream_t s){
  dim3 block(16,16); // 256 threads
  dim3 grid((M + 15)/16, (N + 15)/16);

  softmax_brute<<<grid, block, 0, s>>>(X,Y,M,N);
}

// Naive
//
// Kernel function to compute the softmax activation for a matrix [M * N]
// Each thread computes all elements in a row of output matrix. 
//
// Optimizations (vs Brute):
// - Reduces overhead of computing the sum and max for each element and
//   instead shares the computed values for all elements in the row
// - Slightly better performance for narrow matrices but at the   
//   cost of reduced row-level parallelism for wide matrices
__global__ void softmax_naive(const float* X, float* Y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Overlaunch control
    if (row >= M) return;

    // Determine max and sum for the row
    float maxval = naive_row_max(X, Y, row, N);
    float sumval = naive_row_sum(X, Y, row, N, maxval);

    // Set output for all elements in the row
    for (int i = 0; i < N; i++) {
        // Determine output index, used twice in next line
        int idx = row * N + i;

        // Divide the value for index with the aggregated sum after
        // deducting the max value for the same reason as above
        Y[idx] = expf(X[idx] - maxval) / sumval;
    } 
}

// Naive Kernel Launcher
void launch_naive(const float* X, float* Y, int M, int N, cudaStream_t s){
  dim3 block(256);
  dim3 grid((M * block.x - 1)/block.x);

  softmax_naive<<<grid, block, 0, s>>>(X,Y,M,N);
}
