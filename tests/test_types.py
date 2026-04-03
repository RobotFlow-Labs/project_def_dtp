import pytest

from anima_def_dtp.constants import DATASET_THRESHOLDS
from anima_def_dtp.types import ObjectTrack, ScenarioWindow


def test_window_lengths_match_constants() -> None:
    thresholds = DATASET_THRESHOLDS["apolloscape"]
    track = ObjectTrack(
        object_id="1",
        observe_trace=[[float(i), float(i)] for i in range(thresholds.obs_length)],
        future_trace=[[float(i), float(i)] for i in range(thresholds.pred_length)],
    )
    window = ScenarioWindow(
        observe_length=thresholds.obs_length,
        predict_length=thresholds.pred_length,
        time_step=0.5,
        objects={"1": track},
    )
    assert window.objects["1"].object_id == "1"


def test_window_rejects_bad_lengths() -> None:
    track = ObjectTrack(
        object_id="1",
        observe_trace=[[0.0, 0.0]],
        future_trace=[[0.0, 0.0]],
    )
    with pytest.raises(ValueError):
        ScenarioWindow(observe_length=2, predict_length=1, time_step=0.5, objects={"1": track})
