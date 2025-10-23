// softmax/kernels_torch.cu
// LibTorch Softmax reference (row-wise)

#include "kernels.hpp"

#include <assert.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h> 

static inline torch::TensorOptions device_f32() {
  return torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
}

void launch_torch(const float* X, float* Y, int M, int N, cudaStream_t s) {
    assert(torch::cuda::is_available());

    // No autograd overhead in benchmarking
    torch::InferenceMode guard;

    torch::Tensor tensorX = torch::from_blob(const_cast<float*>(X), {M, N}, device_f32());  
    torch::Tensor tensorY = torch::from_blob(Y, {M, N}, device_f32());  
    
    // Perform softmax on the tensor row-wise (dim=1)
    torch::Tensor out = torch::softmax(tensorX, 1);
    tensorY.copy_(out, /*non_blocking=*/true);

    // Fence torch stream so our timer stream s measures correctly
    auto torch_stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t done{};
    cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
    cudaEventRecord(done, torch_stream.stream());
    cudaStreamWaitEvent(s, done, 0);
    cudaEventDestroy(done);
}