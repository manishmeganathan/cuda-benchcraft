# CUDA Benchmarking
A generalized CUDA Kernel Benchmarking framework. 

Benchcraft can prompt for benchmarking parameters and execute multiple kernel implementation for the same workload and generate comparison plot and profiler statistics

## Get Started
Run the following commands to build the project and begin a `benchcraft` session.
```
make build
make craft
```

The results of a benchmark session are stored under `results/*`

## Toolchain
> Target SMs: `86;90` (Ampere & Ada).  
> If your GPU does not support this architure, modify this value in `CMakeLists.txt` 