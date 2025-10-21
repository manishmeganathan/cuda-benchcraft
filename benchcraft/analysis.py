#!/usr/bin/env python3

import json
import matplotlib.pyplot as pyplot 

from pathlib import Path
from typing import List, Dict, Any

def get_benchmarks_records(session_dir: Path) -> List[Dict[str, Any]]:
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
