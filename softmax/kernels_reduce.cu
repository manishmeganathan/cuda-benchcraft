// softmax/kernels_reduce.cu
// Softmax kernels that parallel reduce across shared memory

#include "kernels.hpp"
#include "cudatools.hpp"
#include <math_constants.h> 

// Device function to perform an atomic CAS operation 
// for max with floating point values.
__device__ inline float atomicMaxFloat(float* addrf, float value) {
    // Reinterpret the float pointer as an int pointer
    int* addri = reinterpret_cast<int*>(addrf);

    // Reads the bits in the pointer as float
    int old = *addri;
    float oldf = __int_as_float(old);

    if (!(value == value)) return oldf; // ignore NaN


    // Update only if new value is greater than existing
    // Loops until successfully updating our value with the current assumed value (when we dereferenced)
    // If returned value matches, we succeed, otherwise we fetch the new value and check again
    while (oldf < value) {
        // Record the value we currently assume is at the location
        int assumed = old;

        // Perform CAS and capture value currently at location
        // Break if it matches our assumed value
        old = atomicCAS(addri, assumed, __float_as_int(value));
        if (assumed == old) break; 

        // CAS failed, try again with the current value in the location
        // If this value is greater, we exit at the loop check
        oldf = __int_as_float(old);
    }

    return oldf;
}


// AtomicReduce
//
// Kernel function to compute the softmax activation for a matrix [M * N]
// Each thread participates in an atomic reduction with other threads in the block to 
// converge on a max and sum for the row by using shared memory and atomic operations
//
// Optimizations (vs Naive):
// - Using a shared memory value with atomic operations to determine the row max and sum
// - Using coalesced memory access by striding the matrix row by the width of the thread block
// - Using fast-math function equivalents for computing the exponents (__expf instead of expf). 
//   Loss in precision but gains in performance
// - Minimizing division operations when normalizing by computing it once per thread instead once per element 
template<int TPB>
__global__ void softmax_atomic_reduce(const float* X, float* Y, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return; // Overlaunch control

    const int tx = threadIdx.x; // thread index
    const int rx = row * N;     // row offset

    // Shared memory accumulator for reduction
    __shared__ float reduce;

    // Initialize reduce for max 
    // Only performed by 1 thread
    if (tx == 0) reduce = -CUDART_INF_F;
    __syncthreads();

    // Define max and sum for the row
    float maxval = -CUDART_INF_F;
    float sumval = 0.f;

    // Find the local maximum among N / TPB elements in the row
    // Each thread will do the same with coalesced memory access
    for (int i = tx; i < N; i += TPB) 
        maxval = fmaxf(maxval, X[rx + i]);

    // Atomically reduce thread level maximums into single max for the row
    // We do not have to barrier after local max calculation, every thread 
    // will join the queue to perform the atomic max op when it is ready
    atomicMaxFloat(&reduce, maxval);
    __syncthreads(); 
    maxval = reduce;

    // Initialize reduce for exp sum 
    // Only performed by 1 thread
    if (tx == 0) reduce = 0.f;
    __syncthreads();

    // Find the local sum among N / TPB elements in the row
    // Each thread will do the same with coalesced memory access
    for (int i = tx; i < N; i += TPB) 
        sumval += __expf(X[rx + i] - maxval);

    // Atomically reduce thread level exp sum into single sum for the row
    // We do not have to barrier after local dum calculation, every thread 
    // will join the queue to perform the atomic sum op when it is ready
    atomicAdd(&reduce, sumval);
    __syncthreads();
    sumval = reduce;
    
    // Calculate the normalization factor
    // We reduce the number of divisions the GPU has to perform by
    // precomputing this once per thread instead of once per loop
    float norm = 1.0f / sumval;

    // Normalize all element in the row and set to output matrix
    for (int i = tx; i < N; i += TPB) 
        Y[rx+ i] = norm * __expf(X[rx + i] - maxval);
}

// AtomicReduce Kernel Launcher
void launch_atomic_Reduce(const float* X, float* Y, int M, int N, cudaStream_t s) {
    const int TPB = 256; // Threads-per-block (2^X)
    
    dim3 block(TPB); 
    dim3 grid(M);    // 1 block per row

    softmax_atomic_reduce<TPB><<<grid, block, 0, s>>>(X,Y,M,N);
}


// TreeReduce
//
// Kernel function to compute the softmax activation for a matrix [M * N]
// Each thread participates in a parallel reduction with other threads in the block to converge
// on a max and sum for the row by synchronizing and communicating over the block's shared memory
//
// Optimizations (vs Naive):
// - Using a shared memory buffer for parallel reduction operations like max and sum
// - Using coalesced memory access by striding the matrix row by the width of the thread block
// - Using fast-math function equivalents for computing the exponents (__expf instead of expf). 
//   Loss in precision but gains in performance
// - Minimizing division operations when normalizing by computing it once per thread instead once per element 
//
// Optimizations (vs AtomicReduce):
// - Uses a tree-like parallel reduction for convergence.
// - Thread participation wanes as we go further down the reduction but there is
//   no lock contention for atomic access, so no 'work' is halted unnecessarily
template<int TPB>
__global__ void softmax_tree_reduce(const float* X, float* Y, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return; // Overlaunch control

    const int tx = threadIdx.x; // thread index
    const int rx = row * N;     // row offset

    // Shared memory buffer for reductions
    __shared__ float reduce[TPB]; 

    // Define max and sum for the row
    float maxval = -CUDART_INF_F;
    float sumval = 0.f;

    // Find the local maximum among N / TPB elements in the row
    // Each thread will do the same with coalesced memory access
    for (int i = tx; i < N; i += TPB) 
        maxval = fmaxf(maxval, X[rx + i]);

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
        sumval += __expf(X[rx + i] - maxval);

    // Share the local sum from this thread into shared 
    // memory and wait for all other threads to do the same
    reduce[tx] = sumval;
    __syncthreads();
   
    // Reduce the thread level exp sums into a single sum for the row
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
        Y[rx+ i] = norm * __expf(X[rx + i] - maxval);
}

// TreeReduce Kernel Launcher
void launch_tree_reduce(const float* X, float* Y, int M, int N, cudaStream_t s) {
    const int TPB = 256; // Threads-per-block (2^X)
    
    dim3 block(TPB); 
    dim3 grid(M);    // 1 block per row

    softmax_tree_reduce<TPB><<<grid, block, 0, s>>>(X,Y,M,N);
}