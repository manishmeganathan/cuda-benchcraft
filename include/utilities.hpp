// include/util.hpp
// Utilities: CUDA error check + templated event timer + CLI utility declarations

#pragma once
#include <cstdio>
#include <cstdlib>
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

// Times a callable over N iters using CUDA events; returns average ms.
template <class F>
inline float time_ms_repeat(int iters, cudaStream_t stream, F&& body) {
    // Create CUDA events (start & end)
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    body(); // warmup once

    // Capture start event
    CUDA_CHECK(cudaEventRecord(s, stream));
    // Capture end event (N iterations)
    for (int i = 0; i < iters; ++i) body();
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

// Matrix & Vector utilities
void fill_vector(std::vector<float>& vec, unsigned seed);

// CLI utilities
bool flag_present(int argc, char** argv, const char* flag);
const char* opt_value(int argc, char** argv, const char* key, const char* def);