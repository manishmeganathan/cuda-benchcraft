// softmax/kernels_torch.cu
// LibTorch Softmax reference (row-wise)

#include "kernels.hpp"

#include <assert.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h> 

static inline torch::TensorOptions device_f32() {
    return torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
}

static thread_local cudaEvent_t kFenceEvent = []() {
  cudaEvent_t e{};
  cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  return e;
}();

void launch_torch(const float* X, float* Y, int M, int N, cudaStream_t s) {
    // Confirm that CUDA execution for libtorch is available and disable autograd
    assert(torch::cuda::is_available());
    torch::InferenceMode no_autograd;

    // Wrap device buffers as torch tensors
    torch::Tensor tensorX = torch::from_blob(const_cast<float*>(X), {M, N}, device_f32());  
    torch::Tensor tensorY = torch::from_blob(Y, {M, N}, device_f32());  
    
    // Perform softmax on the tensor row-wise (dim=1)
    torch::Tensor out = torch::softmax(tensorX, 1);
    tensorY.copy_(out, /*non_blocking=*/true);

    // Get the torch stream so we can fence around it
    cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream();
    // Wait for the torch stream to complete so our 
    // timer stream s can time our GPU work correctly
    cudaEventRecord(kFenceEvent, torch_stream);
    cudaStreamWaitEvent(s, kFenceEvent, 0);
}