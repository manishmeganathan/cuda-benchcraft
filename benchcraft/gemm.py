#!/usr/bin/env python3

from . import system

from pathlib import Path
from typing import List

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
):
    """ Run bench_gemm for the given kernel """
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

def run_kernel_profile(
    kernel: str, iters: int,
    M: int, N: int, K: int, 
    seedA: int, seedB: int, 
    output_dir: Path,
) -> List[str]:
    """
    Run the bench_gemm for the given kernel through the nsys CLI. 
    Return a list of artifacts generated from the profiling
    """
    benchmarks = output_dir / "benchmarks.jsonl"
    nsys_stats = output_dir / f"nsys-stats-{kernel}"
    nsys_base  = output_dir / f"nsys-{kernel}"
    nsys_rep   = Path(str(nsys_base) + ".nsys-rep")
    
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

    # Generate profile artifacts
    artifacts = [f"{nsys_stats.name}_{report}" for report in reports]
    artifacts.append(Path(str(nsys_base) + ".sqlite").name)
    artifacts.append(nsys_rep.name)

    return artifacts
