#!/usr/bin/env python3

from typing import List

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
    opt = "Y/n" if default else "y/N"

    while True:
        value = input(f"{prompt} ({opt}): ").strip().lower()
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
        if not index: 
            if selected: break
            else:
                print("(!!) You must select at least one kernel.")
                continue
        
        try:
            kernel = available[int(index)]
        except (ValueError, IndexError):
            print("(!!) Selection must be a valid index")
            continue

        if kernel in selected:
            print("(!!) Already added!")
            continue

        selected.append(kernel)     

    return selected