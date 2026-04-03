"""Benchmark metrics used by the DEF-DTP evaluation harness."""

from __future__ import annotations

from collections.abc import Sequence

from anima_def_dtp.objectives import ade, fde

Trace = Sequence[Sequence[float]]


def fde_distance(predict_trace: Trace, future_trace: Trace) -> float:
    """Alias the paper's final displacement metric for evaluation code."""

    return fde(predict_trace, future_trace)


def miss_rate(
    predict_trace: Trace,
    future_trace: Trace,
    threshold: float,
) -> float:
    """Binary miss-rate metric based on FDE thresholding."""

    return float(fde_distance(predict_trace, future_trace) > threshold)


def offroad_rate(offroad_flags: Sequence[bool | int | float]) -> float:
    """Average off-road ratio from a sequence of per-step indicators."""

    if not offroad_flags:
        return 0.0
    total = sum(float(flag) for flag in offroad_flags)
    return total / len(offroad_flags)


def perturbation_mse(perturbation: Trace) -> float:
    """Mean squared perturbation magnitude across all attack timesteps."""

    if not perturbation:
        return 0.0
    total = 0.0
    for point in perturbation:
        total += point[0] ** 2 + point[1] ** 2
    return total / len(perturbation)


def summarize_degradation(
    predict_trace: Trace,
    future_trace: Trace,
    *,
    miss_threshold: float,
) -> dict[str, float]:
    """Convenience summary used by the protocol layer and report generation."""

    return {
        "ade": ade(predict_trace, future_trace),
        "fde": fde_distance(predict_trace, future_trace),
        "mr": miss_rate(predict_trace, future_trace, miss_threshold),
    }
