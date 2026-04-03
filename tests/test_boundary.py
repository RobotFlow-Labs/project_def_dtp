from dataclasses import dataclass

from anima_def_dtp.attack.boundary import BoundaryWalker
from anima_def_dtp.criteria import AdversarialCriterion
from anima_def_dtp.types import ObjectTrack, PredictionBundle, ScenarioWindow


@dataclass
class SyntheticPredictor:
    obs_length: int = 2
    pred_length: int = 2

    def predict(
        self,
        window: ScenarioWindow,
        target_object_id: str,
        perturbation=None,
    ) -> PredictionBundle:
        track = window.objects[target_object_id]
        observe = [list(point) for point in track.observe_trace]
        if perturbation is not None:
            observe = [
                [point[0] + delta[0], point[1] + delta[1]]
                for point, delta in zip(observe, perturbation, strict=True)
            ]
        mean_lat = sum(point[1] for point in observe) / len(observe)
        future = [list(point) for point in track.future_trace]
        predict = [[point[0], point[1] + mean_lat] for point in future]
        return PredictionBundle(
            observe_traces={target_object_id: observe},
            future_traces={target_object_id: future},
            predict_traces={target_object_id: predict},
        )


def test_boundary_walker_keeps_adversarial_point() -> None:
    window = ScenarioWindow(
        observe_length=2,
        predict_length=2,
        time_step=0.5,
        objects={
            "1": ObjectTrack(
                object_id="1",
                observe_trace=[[0.0, 0.0], [1.0, 0.0]],
                future_trace=[[2.0, 0.0], [3.0, 0.0]],
            )
        },
    )
    predictor = SyntheticPredictor()
    criterion = AdversarialCriterion("apolloscape")
    walker = BoundaryWalker(max_iter=16)
    result = walker.run(window, predictor, criterion, "1", "left")
    assert result.is_adversarial is True
    assert result.query_count > 0
    assert result.metrics["left"] > 2.0
