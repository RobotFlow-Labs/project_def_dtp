"""Backend auto-detection: CUDA > MLX > CPU."""

from __future__ import annotations

import importlib


def detect_backend() -> str:
    """Return the best available backend name."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    try:
        importlib.import_module("mlx")
        return "mlx"
    except ImportError:
        pass
    return "cpu"
