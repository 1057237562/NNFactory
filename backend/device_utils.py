"""Device detection and utility module for multi-backend support.

Provides safe detection of available compute devices:
CUDA (NVIDIA), ROCm (AMD), XPU (Intel), MPS (Apple), CPU fallback.
All functions are safe to call on CPU-only machines.
"""

from typing import Any

import torch


def _get_torch_version() -> Any:
    """Safely get the torch.version module (may not be in Pyright stubs)."""
    import torch.version as _tv  # noqa: F401
    return getattr(torch, 'version', None)


def is_cuda_available() -> tuple[bool, str]:
    """Check if CUDA is available and determine the vendor.

    Returns:
        Tuple of (available: bool, vendor: str).
        Vendor is "nvidia" for CUDA, "amd" for ROCm, or "" if not available.
    """
    if not torch.cuda.is_available():
        return False, ""

    tv = _get_torch_version()
    if tv is not None and tv.hip is not None:
        return True, "amd"

    if tv is not None and tv.cuda is not None:
        return True, "nvidia"

    return True, ""


def is_rocm_available() -> bool:
    """Check if AMD ROCm is available.

    Returns:
        True if CUDA is available with HIP (ROCm backend), False otherwise.
    """
    tv = _get_torch_version()
    return torch.cuda.is_available() and tv is not None and tv.hip is not None


def is_xpu_available() -> bool:
    """Check if Intel XPU is available.

    XPU support is built into PyTorch (IPEX was retired and merged).

    Returns:
        True if an XPU device is available, False otherwise.
    """
    try:
        return torch.xpu.is_available()
    except (AttributeError, RuntimeError):
        return False


def is_mps_available() -> bool:
    """Check if Apple Metal Performance Shaders (MPS) is available.

    Returns:
        True if MPS backend is available, False otherwise.
    """
    try:
        return torch.backends.mps.is_available()
    except (AttributeError, RuntimeError):
        return False


def get_device_name(device_type: str) -> str:
    """Return a human-readable name for the given device type.

    Args:
        device_type: One of "cuda", "rocm", "xpu", "mps", "cpu".

    Returns:
        Human-readable device name string.
    """
    names = {
        "cuda": "NVIDIA GPU",
        "rocm": "AMD ROCm GPU",
        "xpu": "Intel XPU GPU",
        "mps": "Apple MPS",
        "cpu": "CPU",
    }
    return names.get(device_type, "Unknown device")


def resolve_device(device_type: str) -> str:
    """Map frontend device type string to torch device string.

    Args:
        device_type: One of "cuda", "rocm", "xpu", "mps", "cpu".

    Returns:
        torch-compatible device string ("cuda", "xpu", "mps", "cpu").
    """
    mapping = {
        "rocm": "cuda",
        "xpu": "xpu",
        "mps": "mps",
        "cuda": "cuda",
        "cpu": "cpu",
    }
    return mapping.get(device_type, "cpu")


def get_device_detection_code(device_type: str) -> str:
    """Return Python code for device detection in generated __main__ block.

    Args:
        device_type: One of "cuda", "rocm", "xpu", "mps", "cpu".

    Returns:
        A Python assignment statement string like:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    code_map = {
        "cuda": "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "rocm": "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "xpu": "device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')",
        "mps": "device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')",
        "cpu": "device = torch.device('cpu')",
    }
    return code_map.get(device_type, "device = torch.device('cpu')")
