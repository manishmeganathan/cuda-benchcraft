# GEMM Benchmarking Engine

A benchmarking engine for GEMM Kernels. Each kernel aims to add one optimization to extract
more performance through better utilization of GPU architecture or memory management.

## Running GEMM Benchmarking
Run `make build` from the project root to build the the `bench_gemm` binary which will be stored under the `build` directory. The binary can run 1 kernel at a time and export benchmark metrics to a given file by appending to it. Supports exporting records as CSV or JSON (default). Supported kernels can be obtained with ``./build/bench_gemm --list``

All kernel executions are verified against a CuBLAS reference output.

### Example Run
```
./build/bench_gemm --kind Naive1D --output output.json
```

### Configuring Benchmark Parameters
Additional parameters for the benchmark can be configured as follows

| Parameter | Description | Default |
|------------|--------------|--------|
| `--iters` | benchmarking iterations | `10` |
| `--kind` | [*REQUIRED*] kernel to benchmark | ~ |
| `--M` | number of rows in $`A`$ and $`C`$| `1024` |
| `--N` | number of columns in $`A`$ and rows in $`B`$ | `1024` |
| `--K` | number of columns in $`B`$ and $`C`$ | `1024` |
| `--seedA` | generation seed for $`A`$ | `1234` |
| `--seedB` | generation seed for $`B`$ | `5678` |
| `--output` | [*REQUIRED*] file to record benchmark output | ~ |
| `--format` | format of record benchmark output `csv` or `json` | `json` |

## What is GEMM
*GEMM (General Matrix Multiplication)* is defined as:
$$
C = \alpha AB + \beta C'
$$


with $`A`$ and $`B`$ as matrix inputs, $`\alpha`$ and $`\beta`$ scalars, and $`C'`$ a pre-existing matrix overwritten by the output. A plain matrix product $`AB`$ is a GEMM with $`\alpha = 1, \beta=0`$.

GEMMs are a fundamental building block for many operations in neural networks—fully-connected layers, RNN/LSTM/GRU cells, and even convolutions via lowering.  

> *Read More*:   
*https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html*

### Maximum Theoretical Performance
For a GEMM between matrices $`A`$ and $`B`$ with row-major dimensions $`M * K`$ and $`K * N`$ respectively, the number of floating point operations are:
$$
\text FLOPs = 2 \times M \times \ N \times K
$$
This is derived from output matrix $`C`$ having $`M * N`$ elements, each derived from calculating a dot product of $`K`$ elements. Each dot operation is fused multiply-add operation, which by convention is measured as 2 floating point operations.

From this we compute the $`GFLOPS/s`$ for a given kernel's execution:
$$
\text GFLOPs/s = \frac{2MNK}{time(s)} \div 10^9
$$

### Correctness Checks
We compare each variant against cuBLAS and report the **max absolute difference** and the **max relative error**. The default tolerances for these are `abs ≤ 1e-5` and `rel ≤ 1e-5`.

These small differences occur in the first place because floating-point math isn't exact and rounding errors can accumulate differently even when compared against identical kernel implementations. When stacked over the fact that these different kernels accumulate their results in different orders, the rounding effects can be different.


