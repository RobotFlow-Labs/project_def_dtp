from anima_def_dtp.data.windows import input_data_by_attack_step, scenario_window_from_repo_dict


def _repo_like_case():
    return {
        "observe_length": 3,
        "predict_length": 2,
        "time_step": 0.5,
        "objects": {
            "1": {
                "type": 1,
                "observe_trace": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
                "observe_feature": [[0.0], [0.1], [0.2], [0.3], [0.4]],
            }
        },
    }


def test_attack_step_slice_lengths() -> None:
    sliced = input_data_by_attack_step(
        _repo_like_case(),
        obs_length=3,
        pred_length=2,
        attack_step=0,
    )
    window = scenario_window_from_repo_dict(sliced)
    assert len(window.objects["1"].observe_trace) == 3
    assert len(window.objects["1"].future_trace) == 2
