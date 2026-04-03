"""Baseline placeholders required by the paper reproduction interface."""

from __future__ import annotations

from typing import Any


def _stub_result(name: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "baseline": name,
        "status": "stub",
        "details": kwargs,
    }


def run_pso_baseline(**kwargs: Any) -> dict[str, Any]:
    """Placeholder for the PSO baseline runner used in parity reports."""

    return _stub_result("pso", **kwargs)


def run_sba_baseline(**kwargs: Any) -> dict[str, Any]:
    """Placeholder for the SBA baseline runner used in parity reports."""

    return _stub_result("sba", **kwargs)
