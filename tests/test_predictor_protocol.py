from dataclasses import dataclass

from anima_def_dtp.predictors.base import PredictorAdapter
from anima_def_dtp.types import PredictionBundle, ScenarioWindow


@dataclass
class DummyPredictor:
    obs_length: int = 2
    pred_length: int = 2

    def predict(
        self,
        window: ScenarioWindow,
        target_object_id: str,
        perturbation=None,
    ) -> PredictionBundle:
        base = window.objects[target_object_id]
        observe = [list(point) for point in base.observe_trace]
        if perturbation is not None:
            observe = [
                [point[0] + delta[0], point[1] + delta[1]]
                for point, delta in zip(observe, perturbation, strict=True)
            ]
        return PredictionBundle(
            observe_traces={target_object_id: observe},
            future_traces={target_object_id: [list(point) for point in base.future_trace]},
            predict_traces={target_object_id: [list(point) for point in base.future_trace]},
        )


def test_dummy_predictor_matches_protocol() -> None:
    predictor: PredictorAdapter = DummyPredictor()
    assert predictor.obs_length == 2
