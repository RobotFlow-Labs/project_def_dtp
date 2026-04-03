"""Request / response models for the DEF-DTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from anima_def_dtp.types import AttackResult


class AttackRequest(BaseModel):
    """POST /attack request body."""

    dataset_name: Literal["nuscenes", "apolloscape"] = "nuscenes"
    predictor_name: Literal["replay", "grip", "trajectron", "trajectron_map"] = "replay"
    objective_name: Literal["ade", "fde", "left", "right", "front", "rear"] = "ade"
    target_object_id: str | None = None
    window: dict  # raw repo-style JSON window


class AttackResponse(BaseModel):
    """POST /attack response body."""

    target_object_id: str
    objective: str
    is_adversarial: bool
    query_count: int
    distance_to_original: float
    perturbation: list[list[float]]
    metrics: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result: AttackResult) -> AttackResponse:
        return cls(**result.model_dump())


class EvaluateRequest(BaseModel):
    """POST /evaluate request body."""

    dataset_name: Literal["nuscenes", "apolloscape"] = "nuscenes"
    predictor_name: Literal["replay", "grip", "trajectron", "trajectron_map"] = "replay"
    objective_name: Literal["ade", "fde", "left", "right", "front", "rear"] = "ade"
    target_object_id: str | None = None
    window: dict


class EvaluateResponse(BaseModel):
    """POST /evaluate response body."""

    target_object_id: str
    objective: str
    metric_value: float
    is_adversarial: bool


class HealthResponse(BaseModel):
    """GET /healthz response body."""

    status: str = "ok"
    module: str = "def-dtp"
    version: str = ""
