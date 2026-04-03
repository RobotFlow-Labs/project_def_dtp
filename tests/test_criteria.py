from anima_def_dtp.criteria import AdversarialCriterion
from anima_def_dtp.types import PredictionBundle


def _bundle():
    return PredictionBundle(
        observe_traces={"1": [[0.0, 0.0], [1.0, 0.0]]},
        future_traces={"1": [[2.0, 0.0], [3.0, 0.0]]},
        predict_traces={"1": [[2.0, 3.0], [3.0, 3.0]]},
    )


def test_lateral_threshold_triggers_attack() -> None:
    criterion = AdversarialCriterion("apolloscape")
    value, is_adversarial = criterion.evaluate(_bundle(), "1", "left")
    assert value > 2.0
    assert is_adversarial is True
