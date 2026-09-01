from pathlib import Path

import torch
from safetensors.torch import load_file


def load_checkpoint(path, map_location="cpu"):
    """Load tensor-only checkpoint data without enabling pickle execution."""

    if Path(path).suffix.lower() == ".safetensors":
        if map_location is None:
            device = "cpu"
        elif isinstance(map_location, (str, torch.device)):
            device = str(map_location)
        else:
            raise TypeError(
                "safetensors map_location must be a device string or torch.device"
            )
        return load_file(str(path), device=device)
    return torch.load(path, map_location=map_location, weights_only=True)
