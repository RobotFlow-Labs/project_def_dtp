"""Binary adversarial criteria from the paper."""

from __future__ import annotations

from anima_def_dtp.constants import DATASET_THRESHOLDS
from anima_def_dtp.objectives import ade, directional_offset, fde
from anima_def_dtp.types import PredictionBundle


class AdversarialCriterion:
    """Threshold an objective into a binary decision."""

    def __init__(self, dataset_name: str):
        if dataset_name not in DATASET_THRESHOLDS:
            raise KeyError(f"unknown dataset: {dataset_name}")
        self.thresholds = DATASET_THRESHOLDS[dataset_name]

    def evaluate(
        self,
        bundle: PredictionBundle,
        target_object_id: str,
        objective_name: str,
    ) -> tuple[float, bool]:
        predict_trace = bundle.predict_traces[target_object_id]
        future_trace = bundle.future_traces[target_object_id]
        observe_trace = bundle.observe_traces[target_object_id]
        if objective_name == "ade":
            value = ade(predict_trace, future_trace)
            threshold = self.thresholds.ade
        elif objective_name == "fde":
            value = fde(predict_trace, future_trace)
            threshold = self.thresholds.fde
        elif objective_name in {"left", "right"}:
            value = directional_offset(observe_trace, predict_trace, future_trace, objective_name)
            threshold = self.thresholds.lateral
        elif objective_name in {"front", "rear"}:
            value = directional_offset(observe_trace, predict_trace, future_trace, objective_name)
            threshold = self.thresholds.longitudinal
        else:
            raise ValueError(f"unsupported objective: {objective_name}")
        return value, value > threshold
