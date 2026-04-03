import math

from anima_def_dtp.objectives import ade, directional_offset, fde


def test_ade_zero_for_identical_traces() -> None:
    trace = [[0.0, 0.0], [1.0, 1.0]]
    assert ade(trace, trace) == 0.0


def test_ade_returns_mean_euclidean_distance() -> None:
    predict = [[1.0, 0.0], [2.0, 0.0]]
    future = [[0.0, 0.0], [0.0, 0.0]]
    # distances: 1.0 and 2.0, mean = 1.5
    assert ade(predict, future) == 1.5


def test_fde_uses_last_point_euclidean() -> None:
    # last points: (2, 3) vs (1, 1) -> sqrt(1+4) = sqrt(5)
    assert fde([[0.0, 0.0], [2.0, 3.0]], [[0.0, 0.0], [1.0, 1.0]]) == math.sqrt(5.0)


def test_directional_offset_positive_for_leftward_deviation() -> None:
    observe = [[0.0, 0.0], [1.0, 0.0]]
    future = [[2.0, 0.0], [3.0, 0.0]]
    predict = [[2.0, 1.0], [3.0, 1.0]]
    assert directional_offset(observe, predict, future, "left") > 0
