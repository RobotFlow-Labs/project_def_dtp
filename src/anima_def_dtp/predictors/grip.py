"""GRIP / Grip++ adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anima_def_dtp.predictors.base import RepoPredictorAdapter


class GripAdapter(RepoPredictorAdapter):
    """Adapter for the upstream GRIP interface used in the paper repo."""

    predictor_name = "grip"

    def load_backend(
        self,
        *,
        checkpoint_path: str | Path | None,
        device: str,
    ) -> Any:
        try:
            from prediction.model.GRIP.interface import GRIPInterface
        except ImportError as exc:  # pragma: no cover - external dependency path
            raise RuntimeError(
                "GRIPInterface is unavailable. Mount the upstream GRIP code and "
                "its dependencies, or inject a fake backend for local tests."
            ) from exc
        if checkpoint_path is None:
            raise RuntimeError("GripAdapter requires checkpoint_path when not injected.")
        return GRIPInterface(
            self.obs_length,
            self.pred_length,
            pre_load_model=str(checkpoint_path),
            device=device,
        )
