"""Predictor protocol used by the attack engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from anima_def_dtp.data import prediction_bundle_from_repo_dict, scenario_window_to_repo_dict
from anima_def_dtp.types import PredictionBundle, ScenarioWindow

BackendCallable = Callable[[dict[str, Any]], Any]


@runtime_checkable
class PredictorAdapter(Protocol):
    """Minimal protocol for a black-box trajectory predictor."""

    obs_length: int
    pred_length: int

    def predict(
        self,
        window: ScenarioWindow,
        target_object_id: str,
        perturbation: list[list[float]] | None = None,
    ) -> PredictionBundle:
        """Return observed, future, and predicted traces."""


class RepoPredictorAdapter:
    """Base class for wrapping upstream repo-style predictor interfaces."""

    predictor_name = "unknown"

    def __init__(
        self,
        *,
        obs_length: int,
        pred_length: int,
        backend: BackendCallable | Any | None = None,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.backend = backend or self.load_backend(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device

    def load_backend(
        self,
        *,
        checkpoint_path: str | Path | None,
        device: str,
    ) -> BackendCallable | Any:
        raise RuntimeError(
            f"{self.predictor_name} backend is not configured. "
            "Inject a backend for local tests, or wire the upstream predictor "
            "stack and checkpoint path before using this adapter."
        )

    def prepare_input(
        self,
        window: ScenarioWindow,
        *,
        target_object_id: str,
        perturbation: list[list[float]] | None,
    ) -> dict[str, Any]:
        return scenario_window_to_repo_dict(
            window,
            target_object_id=target_object_id,
            perturbation=perturbation,
        )

    def invoke_backend(self, backend_input: dict[str, Any]) -> Any:
        if callable(self.backend):
            return self.backend(backend_input)
        if hasattr(self.backend, "run"):
            return self.backend.run(backend_input)
        raise TypeError(
            f"{self.predictor_name} backend must be callable or expose .run(...)"
        )

    def normalize_output(self, backend_output: Any) -> PredictionBundle:
        if isinstance(backend_output, PredictionBundle):
            return backend_output
        if isinstance(backend_output, tuple):
            backend_output = backend_output[0]
        if isinstance(backend_output, Mapping) and "objects" in backend_output:
            return prediction_bundle_from_repo_dict(dict(backend_output))
        raise TypeError(
            f"{self.predictor_name} backend returned unsupported output type: "
            f"{type(backend_output)!r}"
        )

    def predict(
        self,
        window: ScenarioWindow,
        target_object_id: str,
        perturbation: list[list[float]] | None = None,
    ) -> PredictionBundle:
        backend_input = self.prepare_input(
            window,
            target_object_id=target_object_id,
            perturbation=perturbation,
        )
        backend_output = self.invoke_backend(backend_input)
        return self.normalize_output(backend_output)
