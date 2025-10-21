#!/usr/bin/env python3

from typing import List

def prompt_int(prompt: str, default: int) -> int:
    """ Prompt user for an integer """
    while True:
        value = input(f"{prompt} (default: {default}): ").strip()
        if not value: return default
        
        try: return int(value)
        except ValueError:
            print("  please enter a valid integer.")

def prompt_uint(prompt: str, default: int) -> int:
    """ Prompt user for a positive integer """
    while True:
        value = prompt_int(prompt, default)
        if value < 0:
            print("  please enter a non-negative integer.")

        return value

def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """ Prompt user for a yes/no question """
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
    """ 
    Prompt loop to collect a list of kernels to benchmark.
    Expects user to input the index for the kernel. 
    Selection must contain aleast one option
    """
    menu = "Available Kernels: \n"
    for index, kernel in enumerate(available):
        menu += f"[{index}] {kernel}  "

    selected: List[str] = []
    while True:
        index = input(f"\n{menu}\nEnter Kernel Index (blank to finish): ").strip()
        if not index: 
            if selected: break
            else:
                print("  no kernels selected")
                continue
        
        try:
            kernel = available[int(index)]
        except (ValueError, IndexError):
            print("  selection must be a valid index")
            continue

        if kernel in selected:
            print("  kernel already selected")
            continue

        selected.append(kernel)     

    return selected