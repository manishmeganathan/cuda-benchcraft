#!/usr/bin/env python3
"""
benchcraft.py — single-shot GEMM benchmarking session builder.

Requires:
  - Python 3.8+
  - Built CUDA binary at build/gemm_bench
  - NVIDIA GPU + CUDA drivers
  - (Optional) Nsight Systems CLI 'nsys' on PATH for profiling
  - (Optional) matplotlib for plotting
"""

import sys
import json
import shutil

from benchcraft import gemm
from benchcraft import prompt, system, analysis

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

ISO_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"

def main() -> int:
    print("== Benchcraft ==")

    # Check if binary exists
    if not gemm.bin_exists():
        return 1

    # Cache kernel list
    try:
        kernel_list = gemm.list_kernels()
    except Exception as e:
        print(str(e))
        return 1
    if not kernel_list:
        print("error: no kernels reported by the binary.")
        return 1

    # Check Nsight CLI
    nsys_ok = shutil.which("nsys") is not None

    # Prompt loop
    while True:
        print("\n-- GEMM Benchmark Configuration --")

        # Collect matrix parameters
        matrix_params = {}
        matrix_params["M"] = prompt.prompt_int("M", 1024)
        matrix_params["N"] = prompt.prompt_int("N", 1024)
        matrix_params["K"] = prompt.prompt_int("K", 1024)
        matrix_params["seedA"] = prompt.prompt_uint("\nSeed A", 1234)
        matrix_params["seedB"] = prompt.prompt_uint("Seed B", 5678)
        
        # Collect benchmark iterations and kernels
        iters = prompt.prompt_int("\nIterations", 10)
        kernels = prompt.prompt_kernels(kernel_list)

        # Collect optional benchmark features
        plotting = prompt.prompt_yes_no("\nEnable plotting?", default=False)
        profiling = False
        if nsys_ok:
            profiling = prompt.prompt_yes_no("Enable profiling?", default=False)
        else:
            print("(!!) nsys not found; profiling unavailable.")

        # Draft benchcraft session preview
        # Other parameters like 
        config: Dict[str, Any] = {
            "benchmark": {
                "profile": profiling,
                "plot": plotting,
                "iterations": iters,
                "matrix": matrix_params,
                "kernels": kernels[:],
            },
        }

        print("\n-- Preview (benchcraft.json) --")
        print(json.dumps(config, indent=2, sort_keys=False))

        if not prompt.prompt_yes_no("\nProceed with this configuration?", default=True):
            print("Okay, let's reconfigure from scratch.")
            continue # restart prompt loop

        break # exit prompt

    # Generate session ID with current timestamp
    session_id = f"benchmark-{datetime.now(timezone.utc).strftime(ISO_TIME_FMT)}"
    # Create session directory
    try:
        session_dir = Path("results") / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"error: failed to create session directory: {session_dir}\n{e}")
        return 1

    # Add build and system information
    config["build"] = system.discovery_build(gemm.BIN_PATH)
    config["system"] = system.discovery_system()

    # Add session ID and base artifacts to config
    config["benchmark"]["session_id"] = session_id
    config["artifacts"] = {
        "profile": [], "plots": [], 
        "benchmarks": "benchmarks.jsonl",
    }

    start_time = datetime.now()
    try:
        for kernel in kernels:
            print(f">> Benchmarking {kernel} ...")

            if profiling:
                artifacts = gemm.run_kernel_profile(kernel, iters, **matrix_params, output_dir=session_dir)
                config["artifacts"]["profile"].append(artifacts)

            else:
                gemm.run_kernel(kernel, iters, **matrix_params, output_dir=session_dir)

        # TODO: process records and artifacts and generate headlines

        if plotting:
            print(f">> Generating Plots ...")

            # Generate GFLOP/s Throughput plot
            if analysis.plot_throughput(session_dir):
                config["artifacts"]["plots"].append("throughput.png")
            else:
                print("(!!) throughput plot generation failed")

    except KeyboardInterrupt:
        print("\n^C received. cleaning up session directory.")

        if session_dir.exists(): shutil.rmtree(session_dir, ignore_errors=True)
        return 1

    except Exception as e:
        print(f"\nerror: {e}")
        print("(!!) aborting and cleaning up session directory.")

        if session_dir.exists(): shutil.rmtree(session_dir, ignore_errors=True)
        return 1
    
    finish_time = datetime.now()

    # Add timing information to config
    config["benchmark"]["time_start"] = start_time.strftime(ISO_TIME_FMT)
    config["benchmark"]["time_finish"] = finish_time.strftime(ISO_TIME_FMT)
    config["benchmark"]["duration"] = str(finish_time - start_time)

    try:
        benchcraft_json = session_dir / "benchcraft.json"
        print(f">> Writing benchcraft config ...")

        with benchcraft_json.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

    except Exception as e:
        print(f"(!!) warning: failed to write benchcraft.json: {e}")

    print(f">> Results saved to: {session_dir}")
    print(f"\n== Benchcraft Session Complete ==")

    return 0

if __name__ == "__main__":
    sys.exit(main())