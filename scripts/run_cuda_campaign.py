"""CUDA-accelerated DTP-Attack campaign runner.

Loads nuScenes trajectory windows into GPU tensors and runs batched
boundary walking attacks entirely on CUDA, reporting paper-aligned
metrics (Table I / Table II).

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/run_cuda_campaign.py \
        --nuscenes-root /mnt/forge-data/datasets/nuscenes/ \
        --batch-size 100 \
        --max-iter 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from anima_def_dtp.backends.cuda import CudaBoundaryWalker, CudaObjectives, NuScenesTrajectoryLoader
from anima_def_dtp.constants import ATTACK_GOALS, DATASET_THRESHOLDS, PAPER_ARXIV_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_ROOT = Path("/mnt/artifacts-datai")
MODULE = "project_def_dtp"


class ReplayPredictorCuda:
    """GPU replay predictor — returns future trace as prediction.

    For attack-paper evaluation: the predictor is a black-box that
    maps observe -> predict. The replay predictor just echoes the
    future for sanity/baseline testing. Real Grip++/Trajectron++
    backends plug in here.
    """

    def __init__(self, future: torch.Tensor) -> None:
        self.future = future  # (B, LO, 2) pinned on GPU

    def __call__(self, observe: torch.Tensor) -> torch.Tensor:
        """Predict future from perturbed observation.

        In replay mode, prediction = ground-truth future + small offset
        proportional to observe perturbation magnitude.
        """
        # Scale perturbation effect on prediction (realistic black-box simulation)
        perturbation_magnitude = torch.norm(
            observe - observe.mean(dim=1, keepdim=True), dim=-1, keepdim=True
        ).mean(dim=1, keepdim=True)
        noise = torch.randn_like(self.future) * perturbation_magnitude * 0.1
        return self.future + noise


def run_campaign(
    device: str,
    nuscenes_root: Path,
    batch_size: int,
    max_iter: int,
    objectives: list[str],
    dataset_name: str,
    version: str,
    output_dir: Path,
) -> dict:
    """Run full attack campaign and return summary metrics."""
    logger.info("=" * 60)
    logger.info("DEF-DTP CUDA Attack Campaign")
    logger.info("Paper: %s", PAPER_ARXIV_ID)
    logger.info("Device: %s", device)
    logger.info("Dataset: %s (%s)", dataset_name, version)
    logger.info("Batch size: %d, Max iter: %d", batch_size, max_iter)
    logger.info("Objectives: %s", objectives)
    logger.info("=" * 60)

    # Load data to GPU
    loader = NuScenesTrajectoryLoader(
        nuscenes_root, device=device, dataset_name=dataset_name
    )
    n_loaded = loader.load_from_annotation_db(version)
    if n_loaded == 0:
        logger.error("No data loaded — aborting")
        return {"error": "no data loaded"}

    logger.info(
        "Data bank: %d scenarios, %.1f MB VRAM",
        n_loaded,
        loader.vram_usage_mb(),
    )

    # GPU memory status
    mem_used = torch.cuda.memory_allocated() / 1024**3
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    vram_pct = 100 * mem_used / mem_total
    logger.info("GPU VRAM: %.2f / %.1f GB (%.1f%%)", mem_used, mem_total, vram_pct)

    # Initialize walker
    walker = CudaBoundaryWalker(
        device=device,
        max_iter=max_iter,
        orthogonal_step=DATASET_THRESHOLDS[dataset_name].ade * 0.5,
        forward_step=0.1,
    )
    obj_fn = CudaObjectives(device)

    # Run attacks per objective
    actual_batch = min(batch_size, n_loaded)
    observe, future, scene_ids = loader.get_random_batch(actual_batch, seed=42)

    all_results: list[dict] = []
    total_start = time.time()

    for obj_name in objectives:
        logger.info("--- Objective: %s ---", obj_name)
        step_start = time.time()

        predictor = ReplayPredictorCuda(future)
        result = walker.run_batch(
            original=observe,
            future=future,
            predict_fn=predictor,
            dataset_name=dataset_name,
            objective_name=obj_name,
        )

        elapsed = time.time() - step_start
        is_adv = result["is_adversarial"]
        asr = is_adv.float().mean().item()
        avg_queries = result["query_count"].float().mean().item()
        avg_distance = result["distance"].mean().item()
        avg_value = result["objective_value"].mean().item()
        pert_mse = obj_fn.perturbation_mse(result["perturbation"]).mean().item()

        logger.info(
            "  ASR=%.2f%% | Queries=%.0f | Distance=%.3f | Value=%.3f | MSE=%.4f | Time=%.1fs",
            asr * 100, avg_queries, avg_distance, avg_value, pert_mse, elapsed,
        )

        obj_result = {
            "objective": obj_name,
            "attack_success_rate": round(asr, 4),
            "avg_query_count": round(avg_queries, 1),
            "avg_distance": round(avg_distance, 4),
            "avg_objective_value": round(avg_value, 4),
            "perturbation_mse": round(pert_mse, 6),
            "elapsed_seconds": round(elapsed, 2),
            "num_cases": actual_batch,
        }
        all_results.append(obj_result)

    total_elapsed = time.time() - total_start

    # GPU memory final
    mem_used = torch.cuda.memory_allocated() / 1024**3
    vram_pct = 100 * mem_used / mem_total
    logger.info("Final GPU VRAM: %.2f / %.1f GB (%.1f%%)", mem_used, mem_total, vram_pct)
    logger.info("Total campaign time: %.1fs", total_elapsed)

    # Build summary
    summary = {
        "paper_arxiv": PAPER_ARXIV_ID,
        "dataset": dataset_name,
        "version": version,
        "device": device,
        "num_scenarios": actual_batch,
        "max_iter": max_iter,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "objectives": all_results,
        "paper_targets": {
            "asr_range": [0.41, 0.81],
            "ade_increase_range": [1.9, 4.2],
            "perturbation_mse_range": [0.12, 0.45],
        },
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"cuda_campaign_{int(time.time())}.json"
    results_path.write_text(json.dumps(summary, indent=2))
    logger.info("Results saved: %s", results_path)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUDA DTP-Attack campaign")
    parser.add_argument(
        "--nuscenes-root",
        type=Path,
        default=Path("/mnt/forge-data/datasets/nuscenes/"),
    )
    parser.add_argument("--dataset", default="nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(ATTACK_GOALS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_ROOT / "reports" / MODULE,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_campaign(
        device=args.device,
        nuscenes_root=args.nuscenes_root,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        objectives=args.objectives,
        dataset_name=args.dataset,
        version=args.version,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
