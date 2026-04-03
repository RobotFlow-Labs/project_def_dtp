"""Paper-aligned constants for DEF-DTP."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV_ID = "2603.26462"
PAPER_TITLE = "DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction"


@dataclass(frozen=True)
class DatasetThresholds:
    obs_length: int
    pred_length: int
    ade: float
    fde: float
    lateral: float
    longitudinal: float


DATASET_THRESHOLDS: dict[str, DatasetThresholds] = {
    "nuscenes": DatasetThresholds(
        obs_length=4,
        pred_length=12,
        ade=7.5,
        fde=17.5,
        lateral=2.0,
        longitudinal=3.0,
    ),
    "apolloscape": DatasetThresholds(
        obs_length=6,
        pred_length=6,
        ade=3.5,
        fde=7.5,
        lateral=2.0,
        longitudinal=3.0,
    ),
}

ATTACK_GOALS = ("ade", "fde", "left", "right", "front", "rear")
