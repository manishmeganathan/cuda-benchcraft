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

def prompt_parameters(multisize: bool) -> Tuple[List[str], Dict[str, Any],  List[Tuple[int, int]]]:
    """ 
    Prompt user for Softmax benchmarking parameters.
    Returns the following:
    - A list of selected kernels
    - A dictionary of call parameters (includes M & N if single-size)
    - A list of matrix sizes (contains only 1 element if not multi-size) 
    """
    try:
        kernel_list = list_kernels()
    except Exception as e:
        raise e

    if not kernel_list:
        raise RuntimeError("no kernels reported by bench_softmax")

    # Prompt gemm matrix parameters
    print("\n-- Softmax Benchmark Configuration --")
    params: Dict[str, Any] = {}
    sizes: List[Tuple[int, int]] = []

    if multisize:
        print("Enter up to 10 size pairs as 'M,N' | 'M N' | 'M*N'")
        print("Press enter on a blank line to finish (at least one pair required).")

        while len(sizes) < 10:
            raw = input(f"Size #{len(sizes)+1}: ").strip()
            if not raw:
                if sizes:
                    break
                else:
                    print("  no sizes added yet — please enter at least one pair")
                    continue

            # Allow comma or space separated input; allow single value to use default N
            tokens = raw.replace(",", " ")
            tokens = tokens.replace("*", " ")
            tokens = tokens.split()

            try:
                if len(tokens) >= 2:
                    m = int(tokens[0])
                    n = int(tokens[1])
                else:
                    raise ValueError
                
                if m <= 0 or n <= 0:
                    raise ValueError
                
            except ValueError:
                print("  invalid size — enter positive integers like '2048,1024' or '2048 1024' or '2048*2048'")
                continue

            pair = (m, n)
            if pair in sizes:
                print("  duplicate pair ignored")
                continue

            sizes.append(pair)

    else:
        params["M"] = prompt.prompt_int("M", 1024)
        params["N"] = prompt.prompt_int("N", 1024)
        sizes = [(params["M"], params["M"])]

    print() # newline
    params["seedX"] = prompt.prompt_uint("Seed X", 1357)

    # Prompt kernel selection
    kernels = prompt.prompt_kernels(kernel_list)
    print("------------------------------------\n")

    return kernels, params, sizes

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
    profile: bool, **kwargs,
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

def run_kernel_multisize(
    kernel: str, iterations: int, seedX: int, 
    sizes: List[Tuple[int, int]], 
    session_dir: Path, **kwargs,
) -> List[str]:
    """ 
    Run bench_softmax for the given kernel across all given sizes combinations
    Benchmark records are stored to {session_dir}/benchmarks.jsonl

    Returns an empty list (matches return interface of run_kernel)
    """
    benchmarks = session_dir / "benchmarks.jsonl"

    for size in sizes:
        M, N = size
        args = generate_args(kernel, iterations, M, N, seedX, benchmarks)

        # Run the kernel directly on the benchmark engine
        code, out, err = system.run_cmd(args)
        if code != 0:
            raise RuntimeError(f"kernel '{kernel}' benchmark failed:\n{err.strip() or out.strip()}")

    return []