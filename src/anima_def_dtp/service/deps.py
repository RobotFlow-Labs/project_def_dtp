"""Dependency injection for the DEF-DTP API."""

from __future__ import annotations

from functools import lru_cache

from anima_def_dtp.cli import create_predictor
from anima_def_dtp.config import get_settings
from anima_def_dtp.types import ScenarioWindow


@lru_cache(maxsize=1)
def get_engine():
    from anima_def_dtp.attack.engine import DtpAttackEngine

    return DtpAttackEngine(get_settings())


def resolve_predictor(predictor_name: str, window: ScenarioWindow):
    """Build a predictor adapter by name.  Falls back to replay for tests."""
    return create_predictor(
        predictor_name,
        window,
        checkpoint=None,
        device="auto",
    )
