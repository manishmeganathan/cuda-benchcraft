// softmax/kernels_torch.cu
// LibTorch Softmax reference (row-wise)

#include "kernels.hpp"
#include "cudatools.hpp"

#include <assert.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h> 

static inline torch::TensorOptions device_f32() {
  return torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
}

float launch_torch(int iters, const float* X, float* Y, int M, int N) {
  // Confirm that CUDA execution for libtorch is available and disable autograd
  assert(torch::cuda::is_available());
  torch::InferenceMode no_autograd;

  // Reinterpret the device buffers for the matrices as torc tensors
  torch::Tensor tensorX = torch::from_blob(const_cast<float*>(X), {M, N}, device_f32());  
  torch::Tensor tensorY = torch::from_blob(Y, {M, N}, device_f32());  

  // Get the torch stream so we can time it
  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream();

  // Execute the softmax call (row-wise) with the timer
  return time_cuda_events(iters, torch_stream, [&]{
      torch::softmax_out(tensorY, tensorX, 1);
  });
}