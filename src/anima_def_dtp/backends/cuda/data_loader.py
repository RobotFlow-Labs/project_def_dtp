"""Pre-load nuScenes trajectory data into GPU tensors.

Reads the nuScenes dataset (via nuscenes-devkit or raw JSON) and builds
a bank of scenario windows as pre-allocated GPU tensors, eliminating
per-case CPU-to-GPU transfer during attack campaigns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from anima_def_dtp.constants import DATASET_THRESHOLDS

logger = logging.getLogger(__name__)


class NuScenesTrajectoryLoader:
    """Load nuScenes prediction windows as GPU tensor banks.

    Parameters
    ----------
    nuscenes_root : Path
        Path to the nuScenes dataset root (e.g. /mnt/forge-data/datasets/nuscenes/).
    device : str
        CUDA device string.
    obs_length : int
        Number of observed timesteps (paper: 4 for nuScenes).
    pred_length : int
        Number of predicted timesteps (paper: 12 for nuScenes).
    """

    def __init__(
        self,
        nuscenes_root: str | Path,
        *,
        device: str = "cuda",
        obs_length: int | None = None,
        pred_length: int | None = None,
        dataset_name: str = "nuscenes",
    ) -> None:
        self.root = Path(nuscenes_root)
        self.device = torch.device(device)
        thresholds = DATASET_THRESHOLDS[dataset_name]
        self.obs_length = obs_length or thresholds.obs_length
        self.pred_length = pred_length or thresholds.pred_length
        self.dataset_name = dataset_name

        self._observe_bank: torch.Tensor | None = None
        self._future_bank: torch.Tensor | None = None
        self._scene_ids: list[str] = []

    @property
    def num_scenarios(self) -> int:
        return len(self._scene_ids)

    def load_from_prediction_windows(self, windows_dir: str | Path) -> int:
        """Load pre-generated prediction window JSON files into GPU tensors.

        Each JSON file should have the repo-style format with observe_trace
        and future_trace per object.

        Returns the number of scenarios loaded.
        """
        windows_path = Path(windows_dir)
        observe_list: list[list[list[float]]] = []
        future_list: list[list[list[float]]] = []
        scene_ids: list[str] = []

        for json_file in sorted(windows_path.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
                objects = data.get("objects", {})
                for obj_id, obj in objects.items():
                    obs = obj.get("observe_trace", [])
                    fut = obj.get("future_trace", [])
                    if len(obs) >= self.obs_length and len(fut) >= self.pred_length:
                        observe_list.append(obs[: self.obs_length])
                        future_list.append(fut[: self.pred_length])
                        scene_ids.append(f"{json_file.stem}_{obj_id}")
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping %s: %s", json_file, exc)
                continue

        if not observe_list:
            logger.warning("No valid scenarios found in %s", windows_dir)
            return 0

        self._observe_bank = torch.tensor(
            observe_list, dtype=torch.float32, device=self.device
        )
        self._future_bank = torch.tensor(
            future_list, dtype=torch.float32, device=self.device
        )
        self._scene_ids = scene_ids
        logger.info(
            "Loaded %d scenarios to %s (%.1f MB VRAM)",
            len(scene_ids),
            self.device,
            (self._observe_bank.nbytes + self._future_bank.nbytes) / 1024**2,
        )
        return len(scene_ids)

    def load_from_annotation_db(self, version: str = "v1.0-mini") -> int:
        """Extract trajectory windows from nuScenes annotation database.

        Uses the sample_annotation.json to build observe+future windows
        for each instance across consecutive samples.
        """
        meta_dir = self.root / "nuscenes" / version
        if not meta_dir.exists():
            meta_dir = self.root / version
        if not meta_dir.exists():
            logger.warning("nuScenes metadata not found at %s", meta_dir)
            return 0

        ann_path = meta_dir / "sample_annotation.json"
        sample_path = meta_dir / "sample.json"
        if not ann_path.exists() or not sample_path.exists():
            logger.warning("Missing annotation or sample JSON in %s", meta_dir)
            return 0

        samples = json.loads(sample_path.read_text())
        annotations = json.loads(ann_path.read_text())

        # Build sample timeline: token -> timestamp
        sample_ts = {s["token"]: s["timestamp"] for s in samples}

        # Group annotations by instance
        instance_anns: dict[str, list[dict]] = {}
        for ann in annotations:
            inst = ann["instance_token"]
            instance_anns.setdefault(inst, []).append(ann)

        # Sort each instance's annotations by sample timestamp
        for inst in instance_anns:
            instance_anns[inst].sort(key=lambda a: sample_ts.get(a["sample_token"], 0))

        total_steps = self.obs_length + self.pred_length
        observe_list: list[list[list[float]]] = []
        future_list: list[list[list[float]]] = []
        scene_ids: list[str] = []

        for inst, anns in instance_anns.items():
            if len(anns) < total_steps:
                continue
            # Extract (x, y) from translation field
            positions = [[a["translation"][0], a["translation"][1]] for a in anns]
            # Slide windows
            for start in range(0, len(positions) - total_steps + 1, self.obs_length):
                obs = positions[start : start + self.obs_length]
                fut = positions[start + self.obs_length : start + total_steps]
                observe_list.append(obs)
                future_list.append(fut)
                scene_ids.append(f"{inst}_{start}")

        if not observe_list:
            logger.warning("No valid windows extracted from %s", meta_dir)
            return 0

        self._observe_bank = torch.tensor(
            observe_list, dtype=torch.float32, device=self.device
        )
        self._future_bank = torch.tensor(
            future_list, dtype=torch.float32, device=self.device
        )
        self._scene_ids = scene_ids
        logger.info(
            "Loaded %d trajectory windows from nuScenes %s to %s (%.1f MB VRAM)",
            len(scene_ids),
            version,
            self.device,
            (self._observe_bank.nbytes + self._future_bank.nbytes) / 1024**2,
        )
        return len(scene_ids)

    def get_batch(
        self, indices: list[int] | torch.Tensor | None = None, batch_size: int = 64
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Return a batch of (observe, future, scene_ids) from the GPU bank.

        If indices is None, returns the first batch_size scenarios.
        """
        if self._observe_bank is None or self._future_bank is None:
            raise RuntimeError("No data loaded. Call load_from_* first.")
        if indices is None:
            indices = list(range(min(batch_size, self.num_scenarios)))
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        return (
            self._observe_bank[indices],
            self._future_bank[indices],
            [self._scene_ids[i] for i in indices.cpu().tolist()],
        )

    def get_random_batch(
        self, batch_size: int = 64, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Return a random batch from the GPU bank."""
        if self.num_scenarios == 0:
            raise RuntimeError("No data loaded.")
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        indices = torch.randperm(self.num_scenarios, generator=gen)[:batch_size]
        return self.get_batch(indices.to(self.device), batch_size)

    def vram_usage_mb(self) -> float:
        """Current VRAM usage of the tensor bank in MB."""
        if self._observe_bank is None or self._future_bank is None:
            return 0.0
        return (self._observe_bank.nbytes + self._future_bank.nbytes) / 1024**2
