#!/usr/bin/env python3

import json
import matplotlib.pyplot as pyplot 

from . import system

from pathlib import Path
from typing import List, Dict, Any, Tuple

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

def run_kernel(
    kernel: str, iters: int,
    M: int, N: int, K: int, 
    seedA: int, seedB: int, 
    output_dir: Path,
) -> Dict[str, Any]:
    """ Run gemm_bench engine for the given kernel. Returns the benchmark record """
    benchmarks = output_dir / "benchmarks.jsonl"

    # Run the kernel on the benchmark engine
    code, out, err = system.run_cmd([
        str(BIN_PATH),
        "--kind", kernel,
        "--iters", str(iters),
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--seedA", str(seedA), "--seedB", str(seedB),
        "--format", "json",
        "--output", str(benchmarks),
    ])
    if code != 0:
        raise RuntimeError(f"gemm_bench failed (code={code}) for kernel '{kernel}':\n{err.strip() or out.strip()}")

    # Obtain the last line from the benchmark output
    return get_benchmark_last(benchmarks, kernel)

def run_kernel_with_profiling(
    kernel: str, iters: int,
    M: int, N: int, K: int, 
    seedA: int, seedB: int, 
    output_dir: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Run the gemm_bench engine for the given kernel through the nsys CLI. 
    Return the benchmark record and list of profiling artifacts
    """
    benchmarks = output_dir / "benchmarks.jsonl"
    nsys_stats = output_dir / f"nsys-stats-{kernel}"
    nsys_base  = output_dir / f"nsys-{kernel}"
    nsys_rep   = Path(str(nsys_base) + ".nsys-rep")

    print("debug: def")
    
    # Run the kernel on the benchmark engine
    # through the nsys CLI and record timeline
    code, out, err = system.run_cmd([
        "nsys", "profile",
        "-o", str(nsys_base),
        "--trace=cuda,cublas,osrt",
        "--sample=cpu",
        "--force-overwrite=true",
        str(BIN_PATH),
        "--kind", kernel,
        "--iters", str(iters),
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--seedA", str(seedA), "--seedB", str(seedB),
        "--format", "json", 
        "--output", str(benchmarks),
    ])
    if code != 0:
        raise RuntimeError(f"nsys(bench_gemm) failed (code={code}) for kernel '{kernel}':\n{err.strip() or out.strip()}")
    
    print("debug: ghi")

    # Obtain the last line from the benchmark output
    record = get_benchmark_last(benchmarks, kernel)

    reports = [
        "cuda_api_gpu_sum",
        "cuda_gpu_kern_gb_sum"
    ]

    # Generate stats from the timeline
    code, out, err = system.run_cmd([
        "nsys", "stats",
        "--report", ','.join(reports),
        "--format", "json",
        "--force-overwrite=true",
        "--force-export=true",
        "-o", str(nsys_stats),
        str(nsys_rep)
    ])
    if code != 0:
        raise RuntimeError(f"nsys stats failed (code={code}): {out.strip() or err.strip()}")
    
    print("debug: jkl")

    # Generate artifacts
    # Generate headlines

    return record, []

def get_benchmark_last(benchmarks: Path, kernel: str) -> Dict[str, Any]:
    # Check that benchmark file exists
    if not benchmarks.exists():
        raise FileNotFoundError(f"could not find benchmark output at '{benchmarks}'")

    record = None
    with benchmarks.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                pass

    if not record or record.get("name") != kernel:
        raise RuntimeError(f"could not parse output for kernel '{kernel}' from {benchmarks}")
    
    return record

def get_benchmarks_all(benchmarks: Path) -> List[Dict[str, Any]]:
    # Check that benchmark file exists
    if not benchmarks.exists():
        raise FileNotFoundError(f"could not find benchmark output at '{benchmarks}'")

    records: List[Dict[str, Any]] = []
    with benchmarks.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            
            records.append(record)

    return records

def plot_graphs(output_dir: Path) -> List[str]:
    """
    Create a single comparison bar chart of GFLOP/s for all kernels in the JSONL.
    Returns True if plot saved (and possibly shown), False if matplotlib unavailable.
    """
    # Obtain all benchmark records
    benchmarks = output_dir / "benchmarks.jsonl"
    records = get_benchmarks_all(benchmarks)

    plot_artifacts = []

    if not records:
        print("(!) No records found for plotting.")
        return plot_artifacts

    kernel = []
    gflops = []

    for record in records:
        kernel.append(record.get('name'))
        gflops.append(record.get('gflops'))

    # Plot kernel throughput in GFLOP/s
    throughput = pyplot.figure()    
    pyplot.bar(kernel, gflops)
    pyplot.ylabel("GFLOP/s")
    pyplot.title("Kernel Throughput")
    pyplot.tight_layout()

    throughput.savefig(output_dir / "throughput.png")
    pyplot.close(throughput)
    plot_artifacts.append("throughput.png")

    return plot_artifacts
