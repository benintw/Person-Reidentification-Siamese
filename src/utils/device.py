"""
Utility functions for device management in PyTorch.

This module provides functionality to determine and select the appropriate
compute device (CPU, CUDA GPU, or Apple MPS) for model training and inference.
"""

import torch


def get_device(device_name: str | None = None) -> torch.device:

    if device_name is not None:
        return torch.device(device_name)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
