#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#include "kernels.hpp"
#include "utilities.hpp"

#include <cuda_runtime.h>

// Prints basic CLI help
static void print_help() {
  std::printf("gemm_bench options:\n");
  std::printf("  --M <int> --N <int> --K <int>\n");
  std::printf("  --iters <int>\n");
  std::printf("  --kind <name|all>\n");
  std::printf("  --list\n");
  std::printf("  --check\n");
}

int main(int argc, char** argv) {
  if (flag_present(argc, argv, "--help")) { print_help(); return 0; }
  if (flag_present(argc, argv, "--list")) { list_kernels(); return 0; }

  // Parse matrix sizes, benchmark iteration and kernel kind (defaults are conservative).
  int M = std::atoi(opt_value(argc, argv, "--M", "1024"));
  int N = std::atoi(opt_value(argc, argv, "--N", "1024"));
  int K = std::atoi(opt_value(argc, argv, "--K", "1024"));
  int iters = std::atoi(opt_value(argc, argv, "--iters", "10"));
  std::string kind_s = opt_value(argc, argv, "--kind", "all");

  // Host buffers
  std::vector<float> hA((size_t)M * K);
  std::vector<float> hB((size_t)K * N);
  std::vector<float> hC((size_t)M * N, 0.f);        // pre-fill C with 0 values
  std::vector<float> hC_ref((size_t)M * N, 0.f);    // pre-fill C with 0 values
  
  // Fill A & B deterministically
  fill_vector(hA, 1234); 
  fill_vector(hB, 5678);

  // Device buffers + stream; async copies.
  float *dA=nullptr, *dB=nullptr, *dC=nullptr, *dC_ref=nullptr;
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
  // Determine memory sizing for matrices
  size_t sA = sizeof(float) * (size_t)M * K;
  size_t sB = sizeof(float) * (size_t)K * N;
  size_t sC = sizeof(float) * (size_t)M * N;

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&dA, sA));
  CUDA_CHECK(cudaMalloc(&dB, sB));
  CUDA_CHECK(cudaMalloc(&dC, sC));
  CUDA_CHECK(cudaMalloc(&dC_ref, sC));
  // Copy filled matrices for A & B to device
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sB, cudaMemcpyHostToDevice));
  // Set output C matrix values to 0
  CUDA_CHECK(cudaMemset(dC, 0, sC));
  CUDA_CHECK(cudaMemset(dC_ref, 0, sC));

  // cuBLAS reference once: C_ref = A * B (row-major via dispatcher’s CuBLAS path).
  launch_kernel(KernelKind::CuBLAS, dA, dB, dC_ref, M, N, K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC_ref, sC, cudaMemcpyDeviceToHost));

  // CSV header once; one row per kernel kind.
  printf("name,M,N,K,iters,ms_avg,gflops,max_abs_err,rel_err\n");

  // Runs one kernel kind: measure avg ms, compute GFLOP/s, print CSV row.
  auto run_one = [&](KernelKind k) {
    CUDA_CHECK(cudaMemset(dC, 0, sC));  

    // Time N iterations with CUDA events (GPU-only time, excludes memory operations).
    float ms = time_ms_repeat(iters, stream, [&]{
      launch_kernel(k, dA, dB, dC, M, N, K, stream);
    });

    // GEMM FLOP count: 2 * M * N * K (mul + add per inner product).
    double gflops = (2.0 * (double)M * (double)N * (double)K) / (ms / 1e3) / 1e9;

    // Copy kernel output C back to host for comparison with cuBLAS reference.
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sC, cudaMemcpyDeviceToHost));
   
    // Determine max absolute error and max reference magnitude
    double max_abs = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < hC.size(); ++i) {
      double diff = std::abs((double)hC[i] - (double)hC_ref[i]);
      if (diff > max_abs) max_abs = diff;                    
      
      double elem = std::abs((double)hC_ref[i]);              
      if (elem > max_ref) max_ref = elem;         
    }

    // Compute relative error from max absolute error and max reference magnitude
    // The 1e-7 correction is to avoid the numerical instability of divide-by-zero
    double rel_err = max_abs / (max_ref + 1e-7);           

    // One CSV row; scripts/bench.py will aggregate and plot this.
    std::printf("%s,%d,%d,%d,%d,%.5f,%.3f,%.3e,%.3e\n", kernel_name(k), M, N, K, iters, ms, gflops, max_abs, rel_err);

    // Ensure the kernel fully completes before launching the next variant.
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  if (kind_s == "all") {
    run_one(KernelKind::Naive1D);
    run_one(KernelKind::Naive2D);
    // ...
    run_one(KernelKind::CuBLAS);
  } else {
    run_one(parse_kind(kind_s.c_str()));
  }

  // Release device memory
  CUDA_CHECK(cudaFree(dA)); 
  CUDA_CHECK(cudaFree(dB)); 
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dC_ref));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}