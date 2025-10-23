#!/usr/bin/env python3

from . import system
from . import prompt
from . import analysis

from pathlib import Path
from typing import List, Dict, Tuple, Any

BIN_PATH = Path("build/bench_softmax") 

def bin_exists() -> bool:
    """ Confirms whether the Softmax benchmarking engine is available """
    return system.bin_exists(BIN_PATH)

def list_kernels() -> List[str]:
    """ Return the list of available kernels for Softmax benchmarking """
    return system.bin_list(BIN_PATH)

def prompt_parameters() -> Tuple[Dict[str, Any], List[str]]:
    """ Prompt user for Softmax benchmarking parameters and kernel choices """
    try:
        kernel_list = list_kernels()
    except Exception as e:
        raise e

    if not kernel_list:
        raise RuntimeError("no kernels reported by bench_softmax")

    # Prompt gemm matrix parameters
    print("\n-- Softmax Benchmark Configuration --")
    params = {}
    params["M"] = prompt.prompt_int("M", 1024)
    params["N"] = prompt.prompt_int("N", 1024)
    print() # newline
    params["seedX"] = prompt.prompt_uint("Seed X", 1357)

    # Prompt kernel selection
    kernels = prompt.prompt_kernels(kernel_list)
    print("------------------------------------\n")

    return params, kernels

def generate_args(
    kernel: str, iters: int,
    M: int, N: int, seedX: int, 
    output: Path
) -> List:
    return [
        str(BIN_PATH),
        "--kind", kernel,
        "--iters", str(iters),
        "--M", str(M), "--N", str(N),
        "--seedX", str(seedX),
        "--format", "json",
        "--output", str(output),
    ]

def run_kernel(
    kernel: str, iterations: int,
    M: int, N: int, seedX: int, 
    session_dir: Path,
    profile: bool,
) -> List[str]:
    """ 
    Run bench_softmax for the given kernel.
    Benchmark records are stored to {session_dir}/benchmarks.jsonl

    If profile = True, runs the benchmark through the nsys CLI and capture profiler stats.
    Return profiler artifacts as a list if this was done, otherwise returns an empty list.
    """
    benchmarks = session_dir / "benchmarks.jsonl"
    arguments = generate_args(kernel, iterations, M, N, seedX, benchmarks)

    # Check if profiler is enabled
    if profile:
        try:
            # Attempt to run the kernel through the nsys CLI, capture timeline and generate stats
            return analysis.profile_stats(session_dir / f"nsys-{kernel}", arguments)
        except Exception as e:
            raise RuntimeError(f"kernel '{kernel}' benchmark (profiled) failed: {e}")
        
    else:
        # Run the kernel directly on the benchmark engine
        code, out, err = system.run_cmd(arguments)
        if code != 0:
            raise RuntimeError(f"kernel '{kernel}' benchmark failed:\n{err.strip() or out.strip()}")

        return []