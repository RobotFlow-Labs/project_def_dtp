"""Attack objectives from the DTP-Attack paper."""

from __future__ import annotations

import math
from collections.abc import Sequence

Trace = Sequence[Sequence[float]]


def _mean_squared_displacement(trace_a: Trace, trace_b: Trace) -> float:
    if len(trace_a) != len(trace_b):
        raise ValueError("trace lengths must match")
    if not trace_a:
        return 0.0
    total = 0.0
    for point_a, point_b in zip(trace_a, trace_b, strict=True):
        total += (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
    return total / len(trace_a)


def ade(predict_trace: Trace, future_trace: Trace) -> float:
    """Average displacement error used in the paper."""

    return _mean_squared_displacement(predict_trace, future_trace)


def fde(predict_trace: Trace, future_trace: Trace) -> float:
    """Final displacement error used in the paper."""

    if len(predict_trace) != len(future_trace):
        raise ValueError("trace lengths must match")
    if not predict_trace:
        return 0.0
    dx = predict_trace[-1][0] - future_trace[-1][0]
    dy = predict_trace[-1][1] - future_trace[-1][1]
    return dx * dx + dy * dy


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    if norm == 0:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def directional_offset(
    observe_trace: Trace,
    predict_trace: Trace,
    future_trace: Trace,
    mode: str,
) -> float:
    """Directional objective used for intention misclassification."""

    if not predict_trace or len(predict_trace) != len(future_trace):
        raise ValueError("predict_trace and future_trace must be non-empty and aligned")
    total = 0.0
    previous = observe_trace[-1]
    direction_sign = 1.0
    for predicted, future in zip(predict_trace, future_trace, strict=True):
        direction = _normalize((future[0] - previous[0], future[1] - previous[1]))
        tangent = direction
        normal = (-direction[1], direction[0])
        delta = (predicted[0] - future[0], predicted[1] - future[1])
        if mode == "left":
            direction_sign = 1.0
            basis = normal
        elif mode == "right":
            direction_sign = -1.0
            basis = normal
        elif mode == "front":
            direction_sign = -1.0
            basis = tangent
        elif mode == "rear":
            direction_sign = 1.0
            basis = tangent
        else:
            raise ValueError(f"unsupported directional objective: {mode}")
        total += direction_sign * (delta[0] * basis[0] + delta[1] * basis[1])
        previous = future
    return total / len(predict_trace)
