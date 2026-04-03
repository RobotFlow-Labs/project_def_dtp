"""GPU-accelerated objective computations for DTP-Attack.

All trace operations use batched torch tensors on CUDA, eliminating
Python loops from the inner attack iteration.
"""

from __future__ import annotations

import torch


class CudaObjectives:
    """Stateless vectorized objectives that operate on GPU tensors.

    All methods accept tensors of shape ``(B, T, 2)`` where B is the
    batch dimension (number of concurrent attack cases), T is the
    sequence length, and 2 is (x, y).
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)

    def ade(
        self,
        predict: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        """Average Displacement Error — Euclidean, shape (B,)."""
        # (B, T, 2) -> (B, T) -> (B,)
        return torch.norm(predict - future, dim=-1).mean(dim=-1)

    def fde(
        self,
        predict: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        """Final Displacement Error — Euclidean at last timestep, shape (B,)."""
        return torch.norm(predict[:, -1] - future[:, -1], dim=-1)

    def trace_distance(
        self,
        trace_a: torch.Tensor,
        trace_b: torch.Tensor,
    ) -> torch.Tensor:
        """Mean Euclidean distance between two trace tensors, shape (B,)."""
        return torch.norm(trace_a - trace_b, dim=-1).mean(dim=-1)

    def directional_offset(
        self,
        observe: torch.Tensor,
        predict: torch.Tensor,
        future: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Vectorized directional offset, shape (B,).

        Parameters
        ----------
        observe : (B, LI, 2)
        predict : (B, LO, 2)
        future  : (B, LO, 2)
        mode    : one of left, right, front, rear
        """
        # Reference point is last observed position: (B, 1, 2)
        ref = observe[:, -1:, :]

        # Direction vectors from reference to each future step: (B, LO, 2)
        direction = future - ref
        norms = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
        tangent = direction / norms  # (B, LO, 2)

        # Normal is 90-degree rotation of tangent
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)  # (B, LO, 2)

        # Deviation of prediction from ground truth
        delta = predict - future  # (B, LO, 2)

        if mode == "left":
            projection = (delta * normal).sum(dim=-1)  # (B, LO)
        elif mode == "right":
            projection = -(delta * normal).sum(dim=-1)
        elif mode == "front":
            projection = -(delta * tangent).sum(dim=-1)
        elif mode == "rear":
            projection = (delta * tangent).sum(dim=-1)
        else:
            raise ValueError(f"unsupported directional mode: {mode}")

        return projection.mean(dim=-1)  # (B,)

    def perturbation_mse(self, perturbation: torch.Tensor) -> torch.Tensor:
        """Mean squared perturbation magnitude, shape (B,)."""
        return (perturbation**2).sum(dim=-1).mean(dim=-1)

    def batch_criterion(
        self,
        values: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Binary adversarial criterion, shape (B,) bool."""
        return values > threshold
