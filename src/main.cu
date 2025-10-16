#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>

#include "kernels.hpp"
#include "utilities.hpp"

#include <cuda_runtime.h>

static void print_help() {
  std::printf("gemm_bench (engine)\n");
  std::printf("  --M <int> --N <int> --K <int>\n");
  std::printf("  --iters <int>\n");
  std::printf("  --kind <KernelName>\n");
  std::printf("  --seedA <uint> --seedB <uint>\n");
  std::printf("  --format <csv|json>\n");
  std::printf("  --output <path>    (required; appends one record)\n");
  std::printf("  --list | --help\n");
}

int main(int argc, char** argv) {
  if (flag_present(argc, argv, "--help")) { print_help(); return 0; }
  if (flag_present(argc, argv, "--list")) { list_kernels(); return 0; }

  // Parse matrix sizes and benchmark iterations
  int M = std::atoi(flag_opt(argc, argv, "--M", "1024"));
  int N = std::atoi(flag_opt(argc, argv, "--N", "1024"));
  int K = std::atoi(flag_opt(argc, argv, "--K", "1024"));
  int iters = std::atoi(flag_opt(argc, argv, "--iters", "10"));
  // Parse matrix fill seeds
  unsigned seedA = (unsigned)std::strtoul(flag_opt(argc, argv, "--seedA", "1234"), nullptr, 10);
  unsigned seedB = (unsigned)std::strtoul(flag_opt(argc, argv, "--seedB", "5678"), nullptr, 10);
  // Parse kernel kind, result formatting and output path
  std::string kind = flag_opt(argc, argv, "--kind", "");
  std::string out = flag_opt(argc, argv, "--output", "");
  std::string fmt = flag_opt(argc, argv, "--format", "csv");

  // --kind and --output are required flags
  if (kind.empty() || out.empty()) {
    std::fprintf(stderr, "error: --kind and --output are required. use --help.\n");
    return 2;
  }
  // Constrain --format to csv or json
  if (fmt != "csv" && fmt != "json") {
    std::fprintf(stderr, "error: --format must be 'csv' or 'json'.\n");
    return 2;
  }

  // Host buffers
  std::vector<float> hA((size_t)M * K);
  std::vector<float> hB((size_t)K * N);
  std::vector<float> hC((size_t)M * N, 0.f);        // pre-fill C with 0 values
  std::vector<float> hC_ref((size_t)M * N, 0.f);    // pre-fill C with 0 values
  
  // Fill A & B deterministically
  fill_vector(hA, seedA); 
  fill_vector(hB, seedB);

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

  // cuBLAS reference: C_ref = A * B (row-major via dispatcher’s CuBLAS path).
  launch_kernel(KernelKind::CuBLAS, dA, dB, dC_ref, M, N, K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC_ref, sC, cudaMemcpyDeviceToHost));

  // Resolve kernel kind
  KernelKind k = parse_kind(kind.c_str());
  
  CUDA_CHECK(cudaMemset(dC, 0, sC));
  // Run the kernel 'iters' times and time with CUDA events
  float ms = time_ms_repeat(iters, stream, [&]{
    launch_kernel(k, dA, dB, dC, M, N, K, stream);
  });
  
  // Copy kernel output C back to host for comparison with cuBLAS reference.
  CUDA_CHECK(cudaStreamSynchronize(stream));
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
  // GEMM FLOP count: 2 * M * N * K (mul + add per inner product).
  double gflops = (2.0 * (double)M * (double)N * (double)K) / (ms / 1e3) / 1e9;

  // Serialize one record and append to --output
  const char* kname = kernel_name(k);
  std::string record = (fmt == "json")
    ? make_json_record(kname, M, N, K, iters, ms, gflops, max_abs, rel_err)
    : make_csv_record (kname, M, N, K, iters, ms, gflops, max_abs, rel_err);

  // Closure to free GPU resources
  auto release_device = [&](){
    if (dA) cudaFree(dA), dA = nullptr;
    if (dB) cudaFree(dB), dB = nullptr;
    if (dC) cudaFree(dC), dC = nullptr;
    if (dC_ref) cudaFree(dC_ref), dC_ref = nullptr;
    if (stream) cudaStreamDestroy(stream), stream = nullptr;
  };

  // Append the benchmark record to the output file
  if (!append_text_line(out, record)) {
    std::fprintf(stderr, "error: failed to append to output file: %s\n", out.c_str());
    
    release_device();
    return 3;
  }

  release_device();
  return 0;
}