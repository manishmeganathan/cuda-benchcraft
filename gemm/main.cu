#include "cudatools.hpp"
#include "filesystem.hpp"
#include "flagparse.hpp"
#include "vectors.hpp"

#include "gemm/include/kernels.hpp"
#include "gemm/include/records.hpp"

static void print_help() {
  std::printf("bench_gemm\n");
  std::printf("  --list | --help\n");
  std::printf("  --iters <int>\n");
  std::printf("  --kind <KernelName> (required) \n");
  std::printf("  --M <int> --N <int> --K <int>\n");
  std::printf("  --seedA <uint> --seedB <uint>\n");
  std::printf("  --format <csv|json>\n");
  std::printf("  --output <path>    (required; appends record)\n");
}

int main(int argc, char** argv) {
  if (flag_present(argc, argv, "--help")) { print_help(); return 0; }
  if (flag_present(argc, argv, "--list")) { list_kernels(); return 0; }

  // Parse matrix sizes and benchmark iterations
  const int M = std::atoi(flag_opt(argc, argv, "--M", "1024"));
  const int N = std::atoi(flag_opt(argc, argv, "--N", "1024"));
  const int K = std::atoi(flag_opt(argc, argv, "--K", "1024"));
  const int iters = std::atoi(flag_opt(argc, argv, "--iters", "10"));
  // Parse matrix fill seeds
  const unsigned seedA = (unsigned)std::strtoul(flag_opt(argc, argv, "--seedA", "1234"), nullptr, 10);
  const unsigned seedB = (unsigned)std::strtoul(flag_opt(argc, argv, "--seedB", "5678"), nullptr, 10);
  // Parse kernel kind, result formatting and output path
  const std::string kind = flag_opt(argc, argv, "--kind", "");
  const std::string out = flag_opt(argc, argv, "--output", "");
  const std::string fmt = flag_opt(argc, argv, "--format", "json");

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
  fill_vector_uniform(hA, seedA); 
  fill_vector_uniform(hB, seedB);

  // Device buffers + stream; async copies.
  float *dA=nullptr, *dB=nullptr, *dC=nullptr, *dC_ref=nullptr;
  cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
  // Determine memory sizing for matrices
  size_t sA = sizeof(float) * (size_t)M * K;
  size_t sB = sizeof(float) * (size_t)K * N;
  size_t sC = sizeof(float) * (size_t)M * N;

  // Closure to free GPU resources
  auto release_device = [&](){
    if (dA) cudaFree(dA), dA = nullptr;
    if (dB) cudaFree(dB), dB = nullptr;
    if (dC) cudaFree(dC), dC = nullptr;
    if (dC_ref) cudaFree(dC_ref), dC_ref = nullptr;
    if (stream) cudaStreamDestroy(stream), stream = nullptr;
  };

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&dA, sA));
  CUDA_CHECK(cudaMalloc(&dB, sB));
  CUDA_CHECK(cudaMalloc(&dC, sC));
  CUDA_CHECK(cudaMalloc(&dC_ref, sC));
  // Copy filled matrices for A & B to device
  CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(), sA, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dB, hB.data(), sB, cudaMemcpyHostToDevice, stream));
  // Set output C matrix values to 0
  CUDA_CHECK(cudaMemsetAsync(dC, 0, sC, stream));
  CUDA_CHECK(cudaMemsetAsync(dC_ref, 0, sC, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // cuBLAS reference: C_ref = A * B (row-major via dispatcher’s CuBLAS path).
  launch_kernel(KernelKind::CuBLAS, dA, dB, dC_ref, M, N, K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC_ref, sC, cudaMemcpyDeviceToHost));
  // Reset dC after CuBLAS run
  CUDA_CHECK(cudaMemset(dC, 0, sC));

  // Resolve kernel kind
  KernelKind k = parse_kernel(kind.c_str());
  
  // Run the kernel 'iters' times and time with CUDA events
  float ms = time_cuda_events(iters, stream, [&]{
    launch_kernel(k, dA, dB, dC, M, N, K, stream);
  });
  
  // Copy kernel output C back to host for comparison with cuBLAS reference.
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpyAsync(hC.data(), dC, sC, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Compare the kernel output with the reference value
  double max_abs_err = 0.0, rel_err = 0.0;
  cmp_vectors(hC, hC_ref, max_abs_err, rel_err);

  // GEMM FLOP count: 2 * M * N * K
  double gflops = (2.0 * (double)M * (double)N * (double)K) / (ms / 1e3) / 1e9;

  // Serialize one record and append to --output
  const char* kname = kernel_name(k);
  std::string record = (fmt == "json")
    ? make_json_record(kname, M, N, K, iters, ms, gflops, max_abs_err, rel_err)
    : make_csv_record (kname, M, N, K, iters, ms, gflops, max_abs_err, rel_err);

  // Append the benchmark record to the output file
  if (!append_file_line(out, record)) {
    std::fprintf(stderr, "error: failed to append to output file: %s\n", out.c_str());
    
    release_device();
    return 3;
  }

  release_device();
  return 0;
}