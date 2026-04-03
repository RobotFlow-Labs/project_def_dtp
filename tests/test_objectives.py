from anima_def_dtp.objectives import ade, directional_offset, fde


def test_ade_zero_for_identical_traces() -> None:
    trace = [[0.0, 0.0], [1.0, 1.0]]
    assert ade(trace, trace) == 0.0


def test_fde_uses_last_point_only() -> None:
    assert fde([[0.0, 0.0], [2.0, 3.0]], [[0.0, 0.0], [1.0, 1.0]]) == 5.0


def test_directional_offset_positive_for_leftward_deviation() -> None:
    observe = [[0.0, 0.0], [1.0, 0.0]]
    future = [[2.0, 0.0], [3.0, 0.0]]
    predict = [[2.0, 1.0], [3.0, 1.0]]
    assert directional_offset(observe, predict, future, "left") > 0
