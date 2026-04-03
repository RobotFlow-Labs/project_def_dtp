"""GPU-resident trajectory predictor for DTP-Attack campaigns.

Simulates a realistic black-box predictor using a lightweight neural
network that maps observed trajectories to future predictions on GPU.
When real Grip++/Trajectron++ checkpoints are available, swap this
for the real model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryPredictorNet(nn.Module):
    """Lightweight trajectory prediction MLP for attack evaluation.

    Architecture matches the complexity level of a real predictor
    to produce realistic GPU memory and compute profiles.
    Input:  (B, LI, 2)
    Output: (B, LO, 2)
    """

    def __init__(
        self,
        obs_length: int = 4,
        pred_length: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.obs_length = obs_length
        self.pred_length = pred_length

        layers: list[nn.Module] = []
        in_dim = obs_length * 2
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, pred_length * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, observe: torch.Tensor) -> torch.Tensor:
        """Map (B, LI, 2) -> (B, LO, 2)."""
        B = observe.shape[0]
        x = observe.reshape(B, -1)
        out = self.net(x)
        return out.reshape(B, self.pred_length, 2)


class GpuPredictor:
    """Wraps a neural net predictor for use in CudaBoundaryWalker.

    Manages the model lifecycle, provides the predict_fn callable,
    and tracks query count.
    """

    def __init__(
        self,
        obs_length: int = 4,
        pred_length: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 4,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.model = TrajectoryPredictorNet(
            obs_length=obs_length,
            pred_length=pred_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device).eval()
        self.query_count = 0

        # Initialize with reasonable weights (simulate trained model)
        with torch.no_grad():
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def __call__(self, observe: torch.Tensor) -> torch.Tensor:
        """Black-box predict: (B, LI, 2) -> (B, LO, 2)."""
        self.query_count += 1
        with torch.no_grad():
            return self.model(observe)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def vram_mb(self) -> float:
        return sum(p.nbytes for p in self.model.parameters()) / 1024**2


class EnsemblePredictor:
    """Ensemble of N predictors to increase GPU memory footprint.

    Runs each model independently and averages predictions.
    This simulates the real scenario where Grip++ and Trajectron++
    are both loaded and queried per attack step.
    """

    def __init__(
        self,
        n_models: int = 5,
        obs_length: int = 4,
        pred_length: int = 12,
        hidden_dim: int = 512,
        num_layers: int = 6,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.predictors = [
            GpuPredictor(
                obs_length=obs_length,
                pred_length=pred_length,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=device,
            )
            for _ in range(n_models)
        ]

    def __call__(self, observe: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([p(observe) for p in self.predictors])
        return preds.mean(dim=0)

    def total_params(self) -> int:
        return sum(p.param_count() for p in self.predictors)

    def total_vram_mb(self) -> float:
        return sum(p.vram_mb() for p in self.predictors)
