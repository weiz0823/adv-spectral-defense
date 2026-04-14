import torch
import torch.backends.cudnn
from packaging import version

try:
    import pynvml

    has_pynvml = True
except ImportError:
    has_pynvml = False


def select_cuda_device_auto_pynvml():
    """Auto-select using pynvml. Based on `torch.cuda.list_gpu_processes()`."""
    if not has_pynvml:
        raise RuntimeError(
            "You need to install `pynvml` to automatically select GPU. "
            "Either manually specify one GPU to use, or `pip install pynvml`."
        )
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    if torch.cuda.device_count() < deviceCount:
        return 0
    best_mem = 0
    best_idx = -1
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if info.free > best_mem:
            best_mem, best_idx = info.free, i
    return best_idx


def cuda_optimization(level=2):
    """Setup CUDA optimization.

    Level:
        - 2:
            - `torch.backends.cudnn.enabled = True`
            - `torch.backends.cudnn.benchmark = True`

    Args:
        level (int): optimization level, similar to -O in cpp. Default: 2.
    """
    if not torch.cuda.is_available():
        return
    if level >= 2:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def get_device(use_cuda=True, device: int | str | None = 0):
    """Get device.

    use_cuda: bool, use CUDA or CPU.
    device: int | str | None
        None: select `torch.cuda.current_device()`
        "auto": auto-select based on memory info
        int: cuda device number
    """
    if use_cuda and torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        elif device == "auto":
            device = select_cuda_device_auto_pynvml()
        cuda_optimization(2)
        dev = torch.device(f"cuda:{device}")
    elif (
        use_cuda
        and version.parse(torch.__version__) >= version.parse("2.0")
        and torch.backends.mps.is_available()
    ):
        dev = torch.device("mps:0")
    else:
        dev = torch.device("cpu")
    return dev
