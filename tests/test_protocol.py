import pytest

from anima_def_dtp.evaluation.protocol import ProtocolCase, run_protocol
from anima_def_dtp.types import AttackResult, ObjectTrack, ScenarioWindow


def _case() -> ProtocolCase:
    return ProtocolCase(
        case_id="case-1",
        dataset_name="apolloscape",
        target_object_id="1",
        window=ScenarioWindow(
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
        ),
    )


def test_query_budget_is_respected() -> None:
    def attacker(**kwargs):
        return AttackResult(
            target_object_id="1",
            objective=kwargs["objective_name"],
            is_adversarial=True,
            query_count=5,
            distance_to_original=1.0,
            perturbation=[[0.0, 0.0], [0.1, 0.0]],
            metrics={kwargs["objective_name"]: 3.0},
        )

    summary = run_protocol([_case()], predictor=object(), attacker=attacker, objectives=["ade"])
    assert summary.total_runs == 1
    assert summary.max_query_count == 5


def test_query_budget_violation_raises() -> None:
    def attacker(**kwargs):
        return AttackResult(
            target_object_id="1",
            objective=kwargs["objective_name"],
            is_adversarial=True,
            query_count=1001,
            distance_to_original=1.0,
            perturbation=[[0.0, 0.0], [0.1, 0.0]],
            metrics={kwargs["objective_name"]: 3.0},
        )

    with pytest.raises(ValueError, match="query budget exceeded"):
        run_protocol([_case()], predictor=object(), attacker=attacker, objectives=["ade"])
