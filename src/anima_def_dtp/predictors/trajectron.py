"""Trajectron++ adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anima_def_dtp.predictors.base import RepoPredictorAdapter


class TrajectronAdapter(RepoPredictorAdapter):
    """Adapter for upstream Trajectron++ map and non-map interfaces."""

    predictor_name = "trajectron"

    def __init__(
        self,
        *,
        obs_length: int,
        pred_length: int,
        use_map: bool = False,
        backend: Any | None = None,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.use_map = use_map
        super().__init__(
            obs_length=obs_length,
            pred_length=pred_length,
            backend=backend,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    def load_backend(
        self,
        *,
        checkpoint_path: str | Path | None,
        device: str,
    ) -> Any:
        try:
            from prediction.model.Trajectron.interface import TrajectronInterface
        except ImportError as exc:  # pragma: no cover - external dependency path
            raise RuntimeError(
                "TrajectronInterface is unavailable. Mount the upstream "
                "Trajectron++ code and its dependencies, or inject a fake "
                "backend for local tests."
            ) from exc
        if checkpoint_path is None:
            raise RuntimeError(
                "TrajectronAdapter requires checkpoint_path when not injected."
            )
        return TrajectronInterface(
            self.obs_length,
            self.pred_length,
            pre_load_model=str(checkpoint_path),
            use_map=self.use_map,
            device=device,
        )
