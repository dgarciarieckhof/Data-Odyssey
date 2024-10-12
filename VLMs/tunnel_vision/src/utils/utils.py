import gc
import torch
import GPUtil
from typing import List

# -------------------
# gpu utilization
def get_device() -> torch.device:
    """
    Returns the appropriate device (GPU if available, otherwise CPU).
    Returns:
        torch.device: The device to use (CPU or GPU).
    """
    if torch.cuda.is_available():
        print("GPU is available")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        return 'cuda:0'
    else:
        print("GPU is not available, using CPU")
        return 'cpu'

def print_gpu_stats() -> None:
    """
    Prints the statistics of all available GPUs.
    Returns:
        None
    """
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Load: {gpu.load * 100:.2f}%")
        print(f"GPU Free Memory: {gpu.memoryFree / 1024:.2f}GB")
        print(f"GPU Used Memory: {gpu.memoryUsed / 1024:.2f}GB")
        print(f"GPU Total Memory: {gpu.memoryTotal / 1024:.2f}GB")
        print(f"GPU Temperature: {gpu.temperature:.2f} °C")
        print(f"GPU UUID: {gpu.uuid}")

def clear_gpu(vars: List[object]) -> None:
    """
    Clears the GPU memory by deleting specified variables, emptying the cache, and running garbage collection.
    Args:
        vars (List[object]): List of variables to delete.
    Returns:
        None
    """
    for var in vars:
        del var
    torch.cuda.empty_cache()
    gc.collect()
