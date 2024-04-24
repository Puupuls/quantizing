import torch
from pynvml import *

def get_gpu_utilization():
    total_memory_used = 0
    for device in range(torch.cuda.device_count()):
        memory_used = torch.cuda.max_memory_allocated(device)
        total_memory_used += memory_used
    return total_memory_used
