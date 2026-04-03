"""CUDA-accelerated boundary walking attack.

Runs B concurrent attack cases on GPU in a single kernel launch per step,
replacing the Python-loop-per-case approach in the CPU backend.
"""

from __future__ import annotations

import torch

from anima_def_dtp.backends.cuda.objectives import CudaObjectives
from anima_def_dtp.constants import DATASET_THRESHOLDS


class CudaBoundaryWalker:
    """Batched boundary walker operating entirely on GPU tensors.

    Parameters
    ----------
    device : str
        CUDA device string (e.g. "cuda:0").
    max_iter : int
        Maximum boundary walk iterations per case.
    orthogonal_step, forward_step, orthogonal_decay, forward_decay
        Algorithm 1 hyperparameters from the paper.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        max_iter: int = 1000,
        orthogonal_step: float = 1.0,
        forward_step: float = 0.1,
        orthogonal_decay: float = 0.95,
        forward_decay: float = 0.9,
        tolerance: float = 1e-6,
        seed: int = 42,
    ) -> None:
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.orthogonal_step = orthogonal_step
        self.forward_step = forward_step
        self.orthogonal_decay = orthogonal_decay
        self.forward_decay = forward_decay
        self.tolerance = tolerance
        self.objectives = CudaObjectives(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def run_batch(
        self,
        original: torch.Tensor,
        future: torch.Tensor,
        predict_fn,
        dataset_name: str,
        objective_name: str,
    ) -> dict[str, torch.Tensor]:
        """Run boundary attack on a batch of cases.

        Parameters
        ----------
        original : (B, LI, 2) — original observed trajectories
        future   : (B, LO, 2) — ground-truth future trajectories
        predict_fn : callable(perturbed_observe: Tensor) -> Tensor
            Black-box predictor that maps (B, LI, 2) -> (B, LO, 2)
        dataset_name : str — for threshold lookup
        objective_name : str — ade, fde, left, right, front, rear

        Returns
        -------
        dict with keys: perturbation, is_adversarial, query_count,
                        distance, objective_value
        """
        B, LI, _ = original.shape
        thresholds = DATASET_THRESHOLDS[dataset_name]

        if objective_name == "ade":
            threshold = thresholds.ade
        elif objective_name == "fde":
            threshold = thresholds.fde
        elif objective_name in {"left", "right"}:
            threshold = thresholds.lateral
        elif objective_name in {"front", "rear"}:
            threshold = thresholds.longitudinal
        else:
            raise ValueError(f"unknown objective: {objective_name}")

        # Random adversarial initialization: (B, LI, 2)
        current = original + torch.randn_like(original) * 10.0
        best = current.clone()
        best_dist = torch.full((B,), float("inf"), device=self.device)
        query_count = torch.zeros(B, dtype=torch.long, device=self.device)

        delta = torch.full((B,), self.orthogonal_step, device=self.device)
        epsilon = torch.full((B,), self.forward_step, device=self.device)

        # Init: find adversarial starting points
        for _ in range(64):
            candidate = original + torch.randn_like(original) * 10.0
            pred = predict_fn(candidate)
            query_count += 1
            values = self._evaluate(candidate, pred, future, original, objective_name)
            is_adv = values > threshold
            # Update current for cases that found adversarial
            better = is_adv & (
                self.objectives.trace_distance(candidate, original) < best_dist
            )
            current[better] = candidate[better]
            best[better] = candidate[better]
            best_dist[better] = self.objectives.trace_distance(
                candidate[better], original[better]
            )

        # Main boundary walking loop
        for _step in range(self.max_iter):
            # Orthogonal step
            diff = current - original  # (B, LI, 2)
            orth = torch.stack([-diff[..., 1], diff[..., 0]], dim=-1)
            orth_norm = torch.norm(orth, dim=-1, keepdim=True).clamp(min=1e-8)
            orth = orth / orth_norm
            jitter = 0.5 + 0.5 * torch.rand(
                B, 1, 1, device=self.device, generator=self.generator
            )
            orth_candidate = current + orth * delta.view(B, 1, 1) * jitter

            pred = predict_fn(orth_candidate)
            query_count += 1
            values = self._evaluate(
                orth_candidate, pred, future, original, objective_name
            )
            is_adv = values > threshold
            current[is_adv] = orth_candidate[is_adv]
            delta[~is_adv] *= self.orthogonal_decay

            # Forward step (toward original)
            step = original - current
            fwd_candidate = current + step * epsilon.view(B, 1, 1)

            pred = predict_fn(fwd_candidate)
            query_count += 1
            values = self._evaluate(
                fwd_candidate, pred, future, original, objective_name
            )
            is_adv = values > threshold
            current[is_adv] = fwd_candidate[is_adv]
            epsilon[is_adv] = torch.clamp(
                epsilon[is_adv] / self.forward_decay,
                max=self.forward_step * 2.0,
            )
            epsilon[~is_adv] *= self.forward_decay

            # Track best
            cur_dist = self.objectives.trace_distance(current, original)
            improved = cur_dist < best_dist
            best[improved] = current[improved]
            best_dist[improved] = cur_dist[improved]

            # Early stop check
            if (epsilon < self.tolerance).all():
                break

        # Final evaluation
        final_pred = predict_fn(best)
        query_count += 1
        final_values = self._evaluate(best, final_pred, future, original, objective_name)
        final_adversarial = final_values > threshold

        perturbation = best - original
        return {
            "perturbation": perturbation,
            "is_adversarial": final_adversarial,
            "query_count": query_count,
            "distance": best_dist,
            "objective_value": final_values,
        }

    def _evaluate(
        self,
        perturbed_observe: torch.Tensor,
        predicted: torch.Tensor,
        future: torch.Tensor,
        original_observe: torch.Tensor,
        objective_name: str,
    ) -> torch.Tensor:
        """Compute objective values for a batch, shape (B,)."""
        if objective_name == "ade":
            return self.objectives.ade(predicted, future)
        elif objective_name == "fde":
            return self.objectives.fde(predicted, future)
        elif objective_name in {"left", "right", "front", "rear"}:
            return self.objectives.directional_offset(
                original_observe, predicted, future, objective_name
            )
        raise ValueError(f"unknown objective: {objective_name}")
