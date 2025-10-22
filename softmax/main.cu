#include "cudatools.hpp"
#include "filesystem.hpp"
#include "flagparse.hpp"
#include "vectors.hpp"

#include "softmax/include/kernels.hpp"
#include "softmax/include/records.hpp"

// Simple CPU reference (stable softmax) for verification
static void softmax_ref(const std::vector<float>& X, std::vector<float>& Y, int M, int N) {
  for (int i = 0; i < M; ++i) {
    const float* x = &X[i * N];
    float*       y = &Y[i * N];

    float max = -INFINITY;
    double sum = 0.0;

    for (int i = 0; i < N; i++) max = x[i] > max ? x[i] : max;

    for (int i = 0; i < N; i++) {
        y[i] = std::exp(x[i] - max);
        sum += y[i];
    }

    float inv = 1.0 / sum;
    for (int i = 0; i < N; i++) y[i] *= inv;
  }
}

static void print_help() {
  std::printf("bench_softmax\n");
  std::printf("  --list | --help\n");
  std::printf("  --iters <int>\n");
  std::printf("  --kind <KernelName> (required)\n");
  std::printf("  --M <int> --N <int>\n");
  std::printf("  --seedX <uint>\n");
  std::printf("  --format <csv|json>\n");
  std::printf("  --output <path>    (required; appends record)\n");
}

int main(int argc, char** argv) {
    if (flag_present(argc, argv, "--help")) { print_help(); return 0; }
    if (flag_present(argc, argv, "--list"))  { list_kernels(); return 0; }

    // Parse matrix sizes and benchmark iterations
    const int M = std::atoi(flag_opt(argc, argv, "--M", "1024"));
    const int N = std::atoi(flag_opt(argc, argv, "--N", "1024"));
    const int K = std::atoi(flag_opt(argc, argv, "--K", "1024"));
    const int iters = std::atoi(flag_opt(argc, argv, "--iters", "10"));
    // Parse matrix fill seeds
    const unsigned seedX = (unsigned)std::strtoul(flag_opt(argc, argv, "--seedX", "1357"), nullptr, 10);
    // Parse kernel kind, result formatting and output path
    const std::string kind   = flag_opt(argc, argv, "--kind", "");
    const std::string out    = flag_opt(argc, argv, "--output", "");
    const std::string fmt    = flag_opt(argc, argv, "--format", "json");

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
    std::vector<float> hX((size_t)M * N);
    std::vector<float> hY((size_t)M * N, 0.f);
    std::vector<float> hY_ref((size_t)M * N, 0.f);

    // Fill X deterministically
    fill_vector_uniform(hX, seedX); 

    // Device buffers + stream; async copies
    float *dX=nullptr, *dY=nullptr;
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    // Determine memory sizing for matrices
    size_t sizeXY = sizeof(float) * (size_t)M * N;

    auto release_device = [&](){
        if (dX) cudaFree(dX), dX=nullptr;
        if (dY) cudaFree(dY), dY=nullptr;
        if (stream) cudaStreamDestroy(stream), stream=nullptr;
    };

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dX, sizeXY));
    CUDA_CHECK(cudaMalloc(&dY, sizeXY));
    // Copy filled matrix X to device
    CUDA_CHECK(cudaMemcpyAsync(dX, hX.data(), sizeXY, cudaMemcpyHostToDevice, stream));
    // Set output Y matrix values to 0
    CUDA_CHECK(cudaMemsetAsync(dY, 0, sizeXY, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Build a CPU reference result
    softmax_ref(hX, hY_ref, M, N);

    // Resolve kernel kind
    KernelKind k = parse_kernel(kind.c_str());

    // Run the kernel 'iters' times and time with CUDA events
    float ms = time_cuda_events(iters, stream, [&]{
        launch_kernel(k, dX, dY, M, N, stream);
    });

    // Copy kernel output C back to host for comparison with CPU reference.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(hY.data(), dY, sizeXY, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare the kernel output with the reference value
    double max_abs_err=0.0, rel_err=0.0;
    cmp_vectors(hY, hY_ref, max_abs_err, rel_err);

    // Softmax FLOP count: 5 * M * N
    // 5X = max, exp, max correction (add), sum, div 
    double gflops = (5.0 * (double)M * (double)N / (ms * 1e-3)) / 1e9;

    // Serialize one record and append to --output
    const char* kname = kernel_name(k);
    std::string record = (fmt == "json")
        ? make_json_record(kname, M, N, iters, ms, gflops, max_abs_err, rel_err)
        : make_csv_record (kname, M, N, iters, ms, gflops, max_abs_err, rel_err);

    // Append the benchmark record to the output file
    if (!append_file_line(out, record)) {
        std::fprintf(stderr, "error: failed to append to output file: %s\n", out.c_str());
        
        release_device();
        return 3;
    }

    release_device();
    return 0;
}