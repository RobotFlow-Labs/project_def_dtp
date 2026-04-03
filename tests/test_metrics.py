from anima_def_dtp.evaluation.metrics import (
    fde_distance,
    miss_rate,
    offroad_rate,
    perturbation_mse,
)


def test_fde_matches_last_point_distance() -> None:
    predict = [[1.0, 1.0], [3.0, 4.0]]
    future = [[1.0, 1.0], [0.0, 0.0]]
    assert fde_distance(predict, future) == 25.0


def test_miss_rate_thresholds_fde() -> None:
    predict = [[1.0, 1.0], [3.0, 4.0]]
    future = [[1.0, 1.0], [0.0, 0.0]]
    assert miss_rate(predict, future, 10.0) == 1.0


def test_offroad_and_perturbation_mse() -> None:
    assert offroad_rate([True, False, True]) == 2 / 3
    assert perturbation_mse([[1.0, 0.0], [0.0, 1.0]]) == 1.0
