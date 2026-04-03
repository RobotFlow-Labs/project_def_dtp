from anima_def_dtp.predictors.trajectron import TrajectronAdapter
from anima_def_dtp.types import ObjectTrack, ScenarioWindow


def _window() -> ScenarioWindow:
    return ScenarioWindow(
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


def test_trajectron_adapter_returns_prediction_bundle() -> None:
    def fake_backend(repo_input):
        obj = repo_input["objects"]["1"]
        obj["predict_trace"] = [[2.0, -1.0], [3.0, -1.0]]
        return repo_input, None

    adapter = TrajectronAdapter(
        obs_length=2,
        pred_length=2,
        backend=fake_backend,
        use_map=True,
    )
    bundle = adapter.predict(_window(), target_object_id="1")
    assert bundle.predict_traces["1"][0] == [2.0, -1.0]
