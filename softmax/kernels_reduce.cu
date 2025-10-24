// softmax/kernels_reduce.cu
// Softmax kernels that parallel reduce across shared memory

#include "kernels.hpp"
#include <math_constants.h> 

// Kernel function to compute the softmax activation for a matrix [M * N]
// Each thread participates in a parallel reduction with other threads in the block to converge
// on a max and sum for the row by synchronizing and communicating over the block's shared memory
//
// Optimizations (vs NaiveRow):
// - Using a shared memory buffer for parallel reduction operations like max and sum
// - Using coalesced memory access by striding the matrix row by the width of the thread block
// - Minimizing division operations when normalizing by computing it once per thread instead once per element 
template<int TPB>
__global__ void softmax_reduce(const float* X, float* Y, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return; // Overlaunch control

    int tx = threadIdx.x;

    // Shared memory buffer for reductions
    __shared__ float reduce[TPB]; 

    // Define max and sum for the row
    float maxval = -CUDART_INF_F;
    float sumval = 0.f;

    // Find the local maximum among N / TPB elements in the row
    // Each thread will do the same with coalesced memory access
    for (int i = tx; i < N; i += TPB) 
        maxval = fmaxf(maxval, X[row * N + i]);

    // Share the local maximum from this thread into shared 
    // memory and wait for all other threads to do the same
    reduce[tx] = maxval;
    __syncthreads();

    // Reduce the thread level maximums into a single maximum for the row
    // Thread participation is halved per iteration 
    for (int stride = TPB / 2; stride > 0; stride /= 2) {
        if (tx < stride) 
            reduce[tx] = fmaxf(reduce[tx], reduce[tx + stride]);
        __syncthreads();
    }

    // Wait for all threads to finish reduction and  
    // collect the reduced maximum value for the row
    __syncthreads();
    maxval = reduce[0];

    // Find the local sum among N / TPB elements in the row
    // Each thread will do the same with coalesced memory access
    for (int i = tx; i < N; i += TPB) 
        sumval += expf(X[row * N + i] - maxval);

    // Share the local sum from this thread into shared 
    // memory and wait for all other threads to do the same
    reduce[tx] = sumval;
    __syncthreads();
   
    // Reduce the thread level sums into a single sum for the row
    // Thread participation is halved per iteration 
    for (int stride = TPB / 2; stride > 0; stride /= 2) {
        if (tx < stride) 
            reduce[tx] += reduce[tx + stride];
        __syncthreads();
    }

    // Wait for all threads to finish reduction and  
    // collect the reduced sum value for the row
    __syncthreads();
    sumval = reduce[0];

    // Calculate the normalization factor
    // We reduce the number of divisions the GPU has to perform by
    // precomputing this once per thread instead of once per loop
    float norm = 1.0f / sumval;

    // Normalize all element in the row and set to output matrix
    for (int i = tx; i < N; i += TPB) 
        Y[row * N + i] = norm * expf(X[row * N + i] - maxval);
}

void launch_reduce(const float* X, float* Y, int M, int N, cudaStream_t s) {
    const int TPB = 256; // Threads-per-block (2^X)
    
    dim3 block(TPB); 
    dim3 grid(M);    // 1 block per row

    softmax_reduce<TPB><<<grid, block, 0, s>>>(X,Y,M,N);
}
