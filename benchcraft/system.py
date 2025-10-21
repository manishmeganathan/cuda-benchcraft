#!/usr/bin/env python3

import hashlib
import platform
import subprocess

from pathlib import Path
from typing import List, Dict, Any, Tuple

def run_cmd(args: List[str], check: bool = False, capture: bool = True, text: bool = True) -> Tuple[int, str, str]:
    """ Run a subprocess and return (code, stdout, stderr) """
    try:
        proc = subprocess.run(args, check, capture_output=capture, text=text)

        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, args, proc.stdout, proc.stderr)
        
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    
    except FileNotFoundError as e:
        return 127, "", f"{e}"

def discovery_cuda() -> Dict[str, Any]:
    """ Discovers and returns CUDA build information """
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
    """ Discovers and returns system information """
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

def discovery_build(bin: Path) -> Dict[str, Any]:
    """ Discover and returns build information """
    build: Dict[str, Any] = {}
     
    # build.hash
    # Generate SHA256 of gemm_bench binary
    try:
        digest = hashlib.sha256()
        # We develop the hash in 8mb chunks from the file
        with bin.open("rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                digest.update(chunk)

        hash = digest.hexdigest()

    except Exception:
        hash = ""

    build["hash"] = hash
    build["cuda"] = discovery_cuda()

    return build