#!/usr/bin/env python3

import math
import json
import matplotlib.pyplot as pyplot 

from . import system

from pathlib import Path
from typing import List, Tuple, Dict, Any

def get_benchmarks_records(session_dir: Path) -> List[Dict[str, Any]]:
    """ 
    Obtain all the benchmark records from session directory. 
    Makes no attempt to validate the schema of the records.
    """
    # Check that benchmarks.jsonl file exists
    benchmarks = session_dir / "benchmarks.jsonl"
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

def plot_throughput(session_dir: Path, multisize: bool = False) -> bool:
    """
    Create a throughput plot from benchmarks.jsonl records in `session_dir`.

    Behavior:
      - Parse all records and bucket by (M,N) size and kernel name.
      - If >1 size bucket exists but `multisize` is False -> print error and return False.
      - If `multisize` is True -> line plot: X = elements (M*N), Y = GFLOP/s, one line per kernel.
        For each (kernel, size) pair, the latest record (last occurrence) is used.
      - If only one size bucket exists -> bar chart with latest record per kernel.

    Saves: session_dir / "throughput.png"
    Returns: True on success, False otherwise.
    """
    # Obtain all benchmark records
    records: List[Dict[str, Any]] = get_benchmarks_records(session_dir)

    if not records:
        print("(!!) No records found for plotting.")
        return False

    # --- Build buckets ---
    # size_buckets: Dict[(M,N), Dict[kernel_name, record]]
    size_buckets: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    for rec in records:
        try:
            M = int(rec.get("M"))
            N = int(rec.get("N"))
            name = str(rec.get("name"))
            gflops = float(rec.get("gflops"))
        except (TypeError, ValueError):
            # Skip malformed entries
            continue
        if not name:
            continue
        key = (M, N)
        # Keep the latest record per (kernel,size) by overwriting as we iterate in order
        size_buckets.setdefault(key, {})[name] = rec

    if not size_buckets:
        print("(!!) No valid (M,N) records for plotting.")
        return False

    unique_sizes = list(size_buckets.keys())

    # --- Guard: multisize disabled but multiple sizes present ---
    if len(unique_sizes) > 1 and not multisize:
        sizes_str = ", ".join([f"{M}x{N}" for (M, N) in sorted(unique_sizes)])
        print(f"(!!) Multiple sizes present ({sizes_str}) but multisize=False. Aborting.")
        return False

    # --- Single-size path: bar chart (latest per kernel) ---
    if len(unique_sizes) == 1:
        (M, N) = unique_sizes[0]
        kernel_to_rec = size_buckets[(M, N)]

        # Collect latest record per kernel
        kernels = sorted(kernel_to_rec.keys())
        gflops_vals = [float(kernel_to_rec[k]["gflops"]) for k in kernels]

        fig = pyplot.figure()
        pyplot.bar(kernels, gflops_vals)
        pyplot.ylabel("GFLOP/s")
        pyplot.title(f"Kernel Throughput @ {M}×{N}")
        pyplot.tight_layout()

        out = session_dir / "throughput.png"
        fig.savefig(out)
        pyplot.close(fig)
        return True

    # --- Multisize path: line plot (X = M*N, Y = GFLOP/s), one line per kernel ---
    # Build per-kernel series: kernel -> List[(elements, gflops)]
    kernel_series: Dict[str, List[Tuple[int, float]]] = {}
    for (M, N), per_kernel in size_buckets.items():
        elements = M * N
        for kernel_name, rec in per_kernel.items():
            try:
                gflops = float(rec["gflops"])
            except (TypeError, ValueError, KeyError):
                continue
            kernel_series.setdefault(kernel_name, []).append((elements, gflops))

    if not kernel_series:
        print("(!!) No kernel series to plot.")
        return False

    # Sort each series by elements so the lines are monotonic in X
    for k in kernel_series:
        kernel_series[k].sort(key=lambda t: t[0])

    fig = pyplot.figure()
    for kernel_name, series in sorted(kernel_series.items()):
        xs = [e for (e, _) in series]
        ys = [g for (_, g) in series]
        pyplot.plot(xs, ys, marker="o", label=kernel_name)

    # Logarithmic X axis (base 2)
    pyplot.xscale("log", base=2)

    # Compute nice power-of-two tick marks covering the data
    all_x = [e for series in kernel_series.values() for (e, _) in series]
    emin, emax = min(all_x), max(all_x)

    lo_pow = max(1, math.floor(math.log2(max(emin, 1))))   # clamp lower pow ≥ 2^1
    hi_pow = min(30, math.ceil(math.log2(emax)))           # clamp upper pow ≤ 2^30
    ticks = [2**k for k in range(int(lo_pow), int(hi_pow) + 1)]

    pyplot.xticks(ticks, [f"2^{int(math.log2(t))}" for t in ticks])
    pyplot.xlim(2**lo_pow, 2**hi_pow)

    pyplot.xlabel("Elements (M×N, log₂ scale)")
    pyplot.ylabel("GFLOP/s")
    pyplot.title("Kernel Throughput vs Problem Size")
    pyplot.legend()
    pyplot.tight_layout()

    out = session_dir / "throughput.png"
    fig.savefig(out)
    pyplot.close(fig)
    return True

def profile_stats(base_path: Path, args: List) -> List[str]:
    """
    Run the nsys CLI with the given arguments. 
    Returns a list of artifacts generated from the profiling
    """
    # Generate base path for nsys related files
    stats_path = Path(str(base_path) + "-stats")
    rep_path = Path(str(base_path) + ".nsys-rep")
    sql_path = Path(str(base_path) + ".sqlite")

    # Run the benchmark engine through the nsys CLI and record timeline
    # args handles specific call and config
    code, out, err = system.run_cmd([
        "nsys", "profile",
        "-o", str(base_path),
        "--trace=osrt,cuda,cublas",
        "--force-overwrite=true",
        *args,
    ])
    if code != 0:
        raise RuntimeError(f"nsys profile failed (code={code}):\n{err.strip() or out.strip()}")

    # List of stat reports to generate from the timeline
    reports = [
        "cuda_api_gpu_sum",
        "cuda_gpu_kern_gb_sum",
        "cuda_gpu_kern_sum",
        "cuda_kern_exec_sum"
    ]

    # Generate stats from the timeline
    code, out, err = system.run_cmd([
        "nsys", "stats",
        "--report", ','.join(reports),
        "--format", "json",
        "--force-overwrite=true",
        "--force-export=true",
        "-o", str(stats_path),
        str(rep_path)
    ])
    if code != 0:
        raise RuntimeError(f"nsys stats failed (code={code}): {out.strip() or err.strip()}")

   # Generate profile artifacts
    artifacts = [f"{stats_path.name}_{report}" for report in reports]
    artifacts.append(sql_path.name)
    artifacts.append(rep_path.name)

    return artifacts
