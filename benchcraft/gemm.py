#!/usr/bin/env python3

from . import system
from . import prompt
from . import analysis

from pathlib import Path
from typing import List, Dict, Tuple, Any

BIN_PATH = Path("build/bench_gemm") 

def bin_exists() -> bool:
    """ Confirms whether the GEMM benchmarking engine is available """
    if not BIN_PATH.exists():
        print(f"error: binary not found at {BIN_PATH}. build first with `make build`")
        return False
    
    if not BIN_PATH.is_file():
        print(f"error: path is not a file: {BIN_PATH}")
        return False

    return True

def list_kernels() -> List[str]:
    """ Return the list of available kernels for GEMM benchmarking """
    code, out, err = system.run_cmd([str(BIN_PATH), "--list"])
    if code != 0:
        raise RuntimeError(f"failed to list kernels (code={code}):\n{err.strip()}")
    
    return [ln.strip() for ln in out.splitlines() if ln.strip()]

def prompt_parameters() -> Tuple[Dict[str, Any], List[str]]:
    """ Prompt user for GEMM benchmarking parameters and kernel choices """
    try:
        kernel_list = list_kernels()
    except Exception as e:
        raise e

    if not kernel_list:
        raise RuntimeError("no kernels reported by bench_gemm")

    # Prompt gemm matrix parameters
    print("\n-- GEMM Benchmark Configuration --")
    params = {}
    params["M"] = prompt.prompt_int("M", 1024)
    params["N"] = prompt.prompt_int("N", 1024)
    params["K"] = prompt.prompt_int("K", 1024)
    print() # newline
    params["seedA"] = prompt.prompt_uint("Seed A", 1234)
    params["seedB"] = prompt.prompt_uint("Seed B", 5678)

    # Prompt kernel selection
    kernels = prompt.prompt_kernels(kernel_list)
    print("------------------------------------\n")

    return params, kernels

def generate_args(
    kernel: str, iters: int,
    M: int, N: int, K: int, 
    seedA: int, seedB: int, 
    output: Path
) -> List:
    return [
        str(BIN_PATH),
        "--kind", kernel,
        "--iters", str(iters),
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--seedA", str(seedA), "--seedB", str(seedB),
        "--format", "json",
        "--output", str(output),
    ]

def run_kernel(
    kernel: str, iterations: int,
    M: int, N: int, K: int, 
    seedA: int, seedB: int, 
    session_dir: Path,
    profile: bool,
) -> List[str]:
    """ 
    Run bench_gemm for the given kernel.
    Benchmark records are stored to {session_dir}/benchmarks.jsonl

    If profile = True, runs the benchmark through the nsys CLI and capture profiler stats.
    Return profiler artifacts as a list if this was done, otherwise returns an empty list.
    """
    benchmarks = session_dir / "benchmarks.jsonl"

    # Check if profiler is enabled
    if profile:
        base_path = session_dir / f"nsys-{kernel}"

        try:
            # Attempt to run the kernel through the nsys CLI, capture timeline and generate stats
            return analysis.profile_stats(base_path, generate_args(kernel, iterations, M, N, K, seedA, seedB, benchmarks))
        except Exception as e:
            raise RuntimeError(f"kernel '{kernel}' benchmark (profiled) failed: {e}")
        
    else:
        # Run the kernel directly on the benchmark engine
        code, out, err = system.run_cmd(generate_args(kernel, iterations, M, N, K, seedA, seedB, benchmarks))
        if code != 0:
            raise RuntimeError(f"kernel '{kernel}' benchmark failed:\n{err.strip() or out.strip()}")

        return []