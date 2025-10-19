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
import hashlib
import platform
import subprocess
import matplotlib.pyplot as plt 

from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

ISO_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"
GEMM_BIN_PATH = Path("build/gemm_bench") 

def run_cmd(args: List[str], check: bool = False, capture: bool = True, text: bool = True) -> Tuple[int, str, str]:
    """ Run a subprocess and return (code, stdout, stderr) """
    try:
        proc = subprocess.run(args, check, capture_output=capture, text=text)

        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, args, proc.stdout, proc.stderr)
        
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    
    except FileNotFoundError as e:
        return 127, "", f"{e}"

def gemm_list_kernels() -> List[str]:
    """ Return the list of available kernels for GEMM benchmarking """
    code, out, err = run_cmd([str(GEMM_BIN_PATH), "--list"])
    if code != 0:
        raise RuntimeError(f"failed to list kernels (code={code}):\n{err.strip()}")
    
    return [ln.strip() for ln in out.splitlines() if ln.strip()]

def gemm_run_kernel(
    kernel: str, iters: int,
    M: int, N: int, K: int, 
    seedA: int, seedB: int, 
    output_dir: Path,
) -> Dict[str, Any]:
    """ Run gemm_bench engine for the given kernel. Returns the benchmark record """
    benchmarks = output_dir / "benchmarks.jsonl"

    # Run the kernel on the benchmark engine
    code, out, err = run_cmd([
        str(GEMM_BIN_PATH),
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

def gemm_run_kernel_with_profiling(
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
    
    # Run the kernel on the benchmark engine
    # through the nsys CLI and record timeline
    code, out, err = run_cmd([
        "nsys", "profile",
        "-o", str(nsys_base),
        "--trace=cuda,cublas,osrt",
        "--sample=cpu",
        "--force-overwrite=true",
        str(GEMM_BIN_PATH),
        "--kind", kernel,
        "--iters", str(iters),
        "--M", str(M), "--N", str(N), "--K", str(K),
        "--seedA", str(seedA), "--seedB", str(seedB),
        "--format", "json", 
        "--output", str(benchmarks),
    ])
    if code != 0:
        raise RuntimeError(f"nsys(gemm_bench) failed (code={code}) for kernel '{kernel}':\n{err.strip() or out.strip()}")

    # Obtain the last line from the benchmark output
    record = get_benchmark_last(benchmarks, kernel)

    reports = [
        "cuda_api_gpu_sum",
        "cuda_gpu_kern_gb_sum"
    ]

    # Generate stats from the timeline
    code, out, err = run_cmd([
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

    # Generate artifacts
    # Generate headlines

    return record, []

def gemm_bin_exists() -> bool:
    """ Confirms whether the GEMM benchmarking engine is available """
    if not GEMM_BIN_PATH.exists():
        print(f"error: binary not found at {GEMM_BIN_PATH}. build first with `make build`")
        return False
    
    if not GEMM_BIN_PATH.is_file():
        print(f"error: path is not a file: {GEMM_BIN_PATH}")
        return False

    return True

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

def discovery_cuda() -> Dict[str, Any]:
    """Discovers and returns CUDA build information"""
    cuda: Dict[str, Any] = {"archs": ""}

    # cuda.archs
    # Attempt to parse from CMakeCache.txt
    cm_cache = Path("build") / "CMakeCache.txt"
    if cm_cache.exists():
        try:
            with cm_cache.open("r", encoding="utf-8") as file:
                for raw in file:
                    line = raw.strip()
                    if not line or line.startswith("//"):
                        continue
                    if line.startswith("CMAKE_CUDA_ARCHITECTURES:"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            cuda["archs"] = parts[1].strip()
                        break
        except Exception:
            pass

    # cuda.nvcc
    # Attempt to parse from nvcc --version output
    code, out, _ = run_cmd(["nvcc", "--version"])
    cuda["nvcc"] = out.strip().splitlines()[-1].split()[-1] if code == 0 and out.strip() else ""

    # cuda.driver
    # Attempt to parse from nvidia-smi query output
    code, out, _ = run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    cuda["driver"] = out.strip() if code == 0 and out.strip() else ""

    return cuda

def discovery_system() -> Dict[str, Any]:
    """Discovers and returns system information"""
    system: Dict[str, Any] = {"cpu": "", "gpus": []}

    # system.os
    system["os"] = f"{platform.system()} {platform.release()}"

    # system.cpu
    # Attempt to parse from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
            for line in file:
                if "model name" in line:
                    system["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    # system.gpus
    # Attempt to parse from nvidia-smi query output
    code, out, _ = run_cmd(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"])
    if code == 0 and out.strip():
        for idx, line in enumerate(out.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            name, comp, mem = parts
            
            gpu = {"name": name, "compute_capacity": comp, "memory": mem}
            system["gpus"].append(gpu)

    return system

def discovery_build() -> Dict[str, Any]:
    """Discover and returns build information"""
    build: Dict[str, Any] = {}
     
    # build.hash
    # Generate SHA256 of gemm_bench binary
    try:
        digest = hashlib.sha256()
        # We develop the hash in 8mb chunks from the file
        with GEMM_BIN_PATH.open("rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                digest.update(chunk)

        hash = digest.hexdigest()

    except Exception:
        hash = ""

    build["hash"] = hash
    build["cuda"] = discovery_cuda()

    return build

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
    throughput = plt.figure()    
    plt.bar(kernel, gflops)
    plt.ylabel("GFLOP/s")
    plt.title("Kernel Throughput")
    plt.tight_layout()

    throughput.savefig(output_dir / "throughput.png")
    plt.close(throughput)
    plot_artifacts.append("throughput.png")

    return plot_artifacts

def prompt_int(prompt: str, default: int) -> int:
    while True:
        value = input(f"{prompt} (default: {default}): ").strip()
        if not value: return default
        
        try: return int(value)
        except ValueError:
            print("  please enter a valid integer.")

def prompt_uint(prompt: str, default: int) -> int:
    while True:
        value = prompt_int(prompt, default)
        if value < 0:
            print("  please enter a non-negative integer.")

        return value

def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    ask = "Y/n" if default else "y/N"

    while True:
        value = input(f"{prompt} ({ask}): ").strip().lower()
        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print("  please enter a valid response (y/n).")

def prompt_kernels(available: List[str]) -> List[str]:
    menu = "Available Kernels: \n"
    for index, kernel in enumerate(available):
        menu += f"[{index}] {kernel}  "

    selected: List[str] = []
    while True:
        index = input(f"\n{menu}\nEnter kernel index (blank to finish): ").strip()
        if not index: break
        
        try:
            kernel = available[int(index)]
        except (ValueError, IndexError):
            print("(!!) Selection must be a valid index")
            continue

        if kernel in selected:
            print("(!!) Already added!")
            continue
    
        selected.append(kernel)

        if not selected:
            print("(!!) You must select at least one kernel.")
            continue

    return selected
    
def main() -> int:
    print("== Benchcraft ==")

    # Check if binary exists
    if not gemm_bin_exists():
        return 1

    # Cache kernel list
    try:
        kernel_list = gemm_list_kernels()
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
        matrix_params["M"] = prompt_int("M", 1024)
        matrix_params["N"] = prompt_int("N", 1024)
        matrix_params["K"] = prompt_int("K", 1024)
        matrix_params["seedA"] = prompt_uint("\nSeed A", 1234)
        matrix_params["seedB"] = prompt_uint("Seed B", 5678)
        
        # Collect benchmark iterations and kernels
        iters = prompt_int("\nIterations", 10)
        kernels = prompt_kernels(kernel_list)

        # Collect optional benchmark features
        plotting = prompt_yes_no("\nEnable plotting?", default=False)
        profiling = False
        if nsys_ok:
            profiling = prompt_yes_no("Enable profiling?", default=False)
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

        if not prompt_yes_no("\nProceed with this configuration?", default=True):
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
    config["build"] = discovery_build()
    config["system"] = discovery_system()

    # Add session ID and base artifacts to config
    config["benchmark"]["session_id"] = session_id
    config["artifacts"] = {"profile": [], "benchmarks": "benchmarks.jsonl"}

    start_time = datetime.now()
    try:
        for kernel in kernels:
            print(f">> Benchmarking {kernel} ...")

            if profiling:
                record, profile_artifacts = gemm_run_kernel_with_profiling(kernel, iters, **matrix_params, output_dir=session_dir)
                config["artifacts"]["profile"].append(profile_artifacts)

            else:
                record = gemm_run_kernel(kernel, iters, **matrix_params, output_dir=session_dir)

            # TODO: process record(s) and artifacts and generate headlines

        if plotting:
            print(f">> Generating Plots ...")
            config["artifacts"]["plots"] = plot_graphs(session_dir)

    except KeyboardInterrupt:
        print("\n^C received. cleaning up session directory.")

        if session_dir.exists(): shutil.rmtree(session_dir, ignore_errors=True)
        return 1

    except Exception as e:
        print(f"\nerror: {e}")
        print("aborting and cleaning up session directory.")

        if session_dir.exists(): shutil.rmtree(session_dir, ignore_errors=True)
        return 1
    
    finish_time = datetime.now()

    # Add timing information to config
    config["benchmark"]["time_start"] = start_time.strftime(ISO_TIME_FMT)
    config["benchmark"]["time_finish"] = finish_time.strftime(ISO_TIME_FMT)
    config["benchmark"]["duration"] = str(finish_time - start_time)

    try:
        benchcraft = session_dir / "benchcraft.json"
        print(f">> Writing benchcraft config ...")

        with benchcraft.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

    except Exception as e:
        print(f"(!!) warning: failed to write benchcraft.json: {e}")

    print(f">> Closing benchcraft session. Results saved to: {session_dir}")

    print(f"\n== Benchcraft Session Complete ==")
    return 0

if __name__ == "__main__":
    sys.exit(main())