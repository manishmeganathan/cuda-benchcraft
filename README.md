# GEMM Benchmarking with CUDA

A step-by-step CUDA GEMM Benchmark: from naive matrix multiplication to GEMM-class kernels.

We add one optimization per rung (coalescing, tiling, shared memory, warp shuffles,
double buffering, `cp.async`, split-K, WMMA), measure correctness vs cuBLAS, and
collect performance metrics (GFLOP/s + Nsight Compute) with a single Python tool.

> Target SMs: **120;90** (Hopper/Blackwell & Ada/Lovelace).  
> Toolchain tested: **CUDA 12.8+**, **GCC/G++ 13+**, **CMake 3.24+**.

## What is GEMM?
> *Source*: *https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html*

*GEMM (General Matrix Multiplication)* is defined as:
\[
C \leftarrow \alpha\,AB + \beta\,C
\]
with \(A\) and \(B\) as matrix inputs, \(\alpha\) and \(\beta\) scalars, and \(C\) a pre-existing matrix overwritten by the output. A plain matrix product \(AB\) is a GEMM with \(\alpha=1,\ \beta=0\).

GEMMs are a fundamental building block for many operations in neural networks—fully-connected layers, RNN/LSTM/GRU cells, and even convolutions via lowering.  

### Mathematic Bounds

For a GEMM with dimensions \(M \times K\) and \(K \times N\), the number of floating-point operations are:
\[
\text{FLOPs} = 2 \times M \times N \times K
\]
(one multiply + one add per inner-product term). We time only the **kernel** region.

GFLOP/s is computed as:
\[
\text{GFLOP/s} = \frac{2MNK}{time(s)} \div 10^9 
\]

This count is **independent of implementation** (naive vs tiled vs WMMA); different kernels only change **time**, not theoretical FLOPs.

### Correctness Checks

We compare each variant against cuBLAS and report the **max absolute difference** and the **max relative error**. The default tolerances for these are `abs ≤ 1e-5` and `rel ≤ 1e-5`.

These small differences occur in the first place because floating-point math isn't exact and rounding errors can accumulate differently even when compared against identical kernel implementations. When stacked over the fact that these different kernels accumulate their results in different orders, the rounding effects can be different.


