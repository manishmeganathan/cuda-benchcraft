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

from benchcraft import prompt, system, analysis

from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone

ISO_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"

WORKLOADS = [
    "GEMM",
    "Softmax",
]


def main() -> int:
    print("== Benchcraft ==")

    # Generate session ID with current timestamp
    session_id = f"benchmark-{datetime.now(timezone.utc).strftime(ISO_TIME_FMT)}"
    # Generate session config with system discovery and base artifacts
    session_config: Dict[str, Any] = {
        "system": system.discovery_system(),
        "benchmark": {
            "session_id": session_id
        },
        "artifacts": {
            "profile": [], "plots": [], 
            "benchmarks": "benchmarks.jsonl",
        }
    }

    # Create session directory
    try:
        session_dir = Path("results") / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
    
    except Exception as e:
        print(f"\n(!!) error: failed to create session directory '{session_dir}': {e}")
        return 1

    try:
        # Prompt benchmarking iterations
        session_config["benchmark"]["iterations"] = prompt.prompt_int("\nBenchmarking Iterations", 10)

        # Prompt plotting enable
        session_config["benchmark"]["plot"] = prompt.prompt_yes_no("Enable Plotting?", default=False)

        # Prompt profiling enable (if nsys is available)
        if shutil.which("nsys") is not None:
            session_config["benchmark"]["profile"] = prompt.prompt_yes_no("Enable Profiling?", default=False)

        # Prompt for benchmarking workload
        menu = "Available Workloads: \n"
        for index, workload in enumerate(WORKLOADS):
            menu += f"[{index}] {workload}  "

        index = prompt.prompt_int(f"\n{menu}\nEnter Workload Index:", 0)
        if index >= len(WORKLOADS):
            raise ValueError("selection must be a valid index")

        # Collect arguments for kernel run
        kernel_run_args: Dict[str, Any] = {}
        kernel_run_args["session_dir"] = session_dir
        kernel_run_args["iterations"] = session_config["benchmark"]["iterations"]
        kernel_run_args["profile"] = session_config["benchmark"]["profile"]
        # Variables for the kernel list and runner func
        kernel_run_list = []
        kernel_run_func = None

        match workload := WORKLOADS[index]:
            case "GEMM":
                from benchcraft import gemm

                # Check if binary exists
                if not gemm.bin_exists(): raise FileNotFoundError("GEMM benchmarking binary is not available")
                # Add build information to session config
                session_config["build"] = system.discovery_build(gemm.BIN_PATH)

                # Prompt for gemm benchmark parameters and kernel selection
                gemm_params, gemm_kernels = gemm.prompt_parameters()
                
                # Use the user provided kernel selection
                kernel_run_list = gemm_kernels
                # Unpack the gemm args into the run args
                kernel_run_args = kernel_run_args | gemm_params
                # Set the run func from the gemm module
                kernel_run_func = gemm.run_kernel

            case "Softmax":
                from benchcraft import softmax

                # Check if binary exists
                if not softmax.bin_exists(): raise FileNotFoundError("Softmax benchmarking binary is not available")
                # Add build information to session config
                session_config["build"] = system.discovery_build(softmax.BIN_PATH)

                # Prompt for softmax benchmark parameters and kernel selection
                softmax_params, softmax_kernels = softmax.prompt_parameters()
                
                # Use the user provided kernel selection
                kernel_run_list = softmax_kernels
                # Unpack the softmax args into the run args
                kernel_run_args = kernel_run_args | softmax_params
                # Set the run func from the softmax module
                kernel_run_func = softmax.run_kernel

            case _:
                raise ValueError(f"unsupported workload: {workload}")

        # Abort if kernel runner was not set
        if kernel_run_func is None:
            raise RuntimeError("kernel runner function was not available")

        start_time = datetime.now()
        
        for kernel in kernel_run_list:
            print(f">> Benchmarking {kernel} ...")

            # Run the kernel benchmark (profiling is enabled if requested)
            artifacts = kernel_run_func(kernel, **kernel_run_args)
            # Attach profiling artifacts
            session_config["artifacts"]["profile"].append(artifacts)

        # TODO: process records and artifacts and generate headlines

        # Generate plots if enabled
        if session_config["benchmark"]["plot"]:
            print(f">> Generating Plots ...")

            # Plot GFLOP/s Throughput for all kernels
            if analysis.plot_throughput(session_dir):
                session_config["artifacts"]["plots"].append("throughput.png")
            else:
                print(">> WARNING: failed to generate throughput plot")

        finish_time = datetime.now()

        # Add timing information to session config
        session_config["benchmark"]["time_start"] = start_time.strftime(ISO_TIME_FMT)
        session_config["benchmark"]["time_finish"] = finish_time.strftime(ISO_TIME_FMT)
        session_config["benchmark"]["duration"] = str(finish_time - start_time)

    except KeyboardInterrupt:
        print(f"\nexit signal received")
        print(">> aborting benchmark. cleaning up session directory.")
        if session_dir.exists(): shutil.rmtree(session_dir, ignore_errors=True)
        return 1

    except Exception as e:
        print(f"\n(!!) ERROR: {e}")
        print(">> aborting benchmark. cleaning up session directory.")
        if session_dir.exists(): shutil.rmtree(session_dir, ignore_errors=True)
        return 1

    try:
        benchcraft_json = session_dir / "benchcraft.json"
        print(f">> Writing benchcraft config ...")

        with benchcraft_json.open("w", encoding="utf-8") as file:
            json.dump(session_config, file, indent=2)

    except Exception as e:
        print(f">> WARNING: failed to write benchcraft.json: {e}")

    print(f">> Results saved to: {session_dir}")
    print(f"\n== Benchcraft Session Complete ==")

    return 0

if __name__ == "__main__":
    sys.exit(main())