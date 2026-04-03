"""Replay predictor — returns future trace as prediction (for dry-runs and tests)."""

from __future__ import annotations

from anima_def_dtp.types import PredictionBundle, ScenarioWindow


class ReplayAdapter:
    """Local dry-run predictor that replays the future trace as the prediction."""

    def __init__(self, *, obs_length: int, pred_length: int) -> None:
        self.obs_length = obs_length
        self.pred_length = pred_length

    def predict(
        self,
        window: ScenarioWindow,
        target_object_id: str,
        perturbation: list[list[float]] | None = None,
    ) -> PredictionBundle:
        observe = {
            obj_id: [list(point) for point in obj.observe_trace]
            for obj_id, obj in window.objects.items()
        }
        if perturbation is not None:
            observe[target_object_id] = [
                [point[0] + delta[0], point[1] + delta[1]]
                for point, delta in zip(
                    observe[target_object_id],
                    perturbation,
                    strict=True,
                )
            ]
        future = {
            obj_id: [list(point) for point in obj.future_trace]
            for obj_id, obj in window.objects.items()
        }
        return PredictionBundle(
            observe_traces=observe,
            future_traces=future,
            predict_traces=future,
        )
