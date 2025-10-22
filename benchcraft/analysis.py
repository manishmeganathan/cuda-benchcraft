#!/usr/bin/env python3

import json
import matplotlib.pyplot as pyplot 

from . import system

from pathlib import Path
from typing import List, Dict, Any

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

def plot_throughput(session_dir: Path) -> bool:
    """
    Create a single comparison bar chart of GFLOP/s for all kernels.
    Data is obtained from the benchmarks.jsonl file in the session directory.
    The plot is saved to the session directory as "throughtput.png"
    """
    # Obtain all benchmark records
    records = get_benchmarks_records(session_dir)

    if not records:
        print("(!!) No records found for plotting.")
        return False

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

    throughput.savefig(session_dir / "throughput.png")
    pyplot.close(throughput)

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
