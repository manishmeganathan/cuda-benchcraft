// common/cudatools.hpp
// CUDA Error Checking & Event Timer

#pragma once
#include <stdio.h> 
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA Error Checking
// Wraps CUDA calls and exits on failure (prints file/line + error string).
#define CUDA_CHECK(stmt) \
    do { cudaError_t err = (stmt); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error '%s' at %s:%d: %s\n", #stmt, __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(1); \
        } \
    } while(0)

// Measure the time of execution of a callable over N iteration using 
// CUDA events. Returns the average execution time in millisecods. 
// The callable is called once for warmup outside of the timing window
template <class F>
inline float time_cuda_events(int iters, cudaStream_t stream, F&& body) {
    // Create CUDA events (start & end)
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    body(); // warmup once
    CUDA_CHECK(cudaPeekAtLastError()); // check the launch right away (helps catch bad builds)

    // Execute the function N time and record each iteration
    CUDA_CHECK(cudaEventRecord(s, stream));
    for (int i = 0; i < iters; i++) body();
    CUDA_CHECK(cudaEventRecord(e, stream));
    CUDA_CHECK(cudaEventSynchronize(e));

    // Extract elapsed time between start and end events
    // This time represents the total time for N iterations
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));

    return ms / iters;
}