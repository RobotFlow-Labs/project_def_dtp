"""Attack objectives from the DTP-Attack paper."""

from __future__ import annotations

import math
from collections.abc import Sequence

Trace = Sequence[Sequence[float]]


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def ade(predict_trace: Trace, future_trace: Trace) -> float:
    """Average Displacement Error — mean Euclidean distance per timestep."""

    if len(predict_trace) != len(future_trace):
        raise ValueError("trace lengths must match")
    if not predict_trace:
        return 0.0
    total = sum(
        _euclidean(p, f) for p, f in zip(predict_trace, future_trace, strict=True)
    )
    return total / len(predict_trace)


def fde(predict_trace: Trace, future_trace: Trace) -> float:
    """Final Displacement Error — Euclidean distance at the last timestep."""

    if len(predict_trace) != len(future_trace):
        raise ValueError("trace lengths must match")
    if not predict_trace:
        return 0.0
    return _euclidean(predict_trace[-1], future_trace[-1])


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
    """Directional objective used for intention misclassification.

    The heading direction is derived from the last observed position to
    each future ground-truth point (instantaneous heading per step),
    matching the reference repo's implementation.
    """

    if not predict_trace or len(predict_trace) != len(future_trace):
        raise ValueError("predict_trace and future_trace must be non-empty and aligned")
    total = 0.0
    reference = observe_trace[-1]
    for predicted, future in zip(predict_trace, future_trace, strict=True):
        direction = _normalize((future[0] - reference[0], future[1] - reference[1]))
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
        reference = future
    return total / len(predict_trace)
