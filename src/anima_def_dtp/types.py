"""Typed scenario contracts for DEF-DTP."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ObjectTrack(BaseModel):
    """One traffic participant across observe and future horizons."""

    model_config = ConfigDict(extra="allow")

    object_id: str
    object_type: int = 0
    observe_trace: list[list[float]]
    future_trace: list[list[float]]
    observe_mask: list[int] = Field(default_factory=list)
    future_mask: list[int] = Field(default_factory=list)
    observe_feature: list[list[float]] = Field(default_factory=list)
    future_feature: list[list[float]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_shapes(self) -> ObjectTrack:
        if self.observe_trace and len({len(row) for row in self.observe_trace}) != 1:
            raise ValueError("observe_trace rows must share one width")
        if self.future_trace and len({len(row) for row in self.future_trace}) != 1:
            raise ValueError("future_trace rows must share one width")
        if self.observe_mask and len(self.observe_mask) != len(self.observe_trace):
            raise ValueError("observe_mask length must match observe_trace")
        if self.future_mask and len(self.future_mask) != len(self.future_trace):
            raise ValueError("future_mask length must match future_trace")
        return self


class ScenarioWindow(BaseModel):
    """One prediction case aligned with the paper notation."""

    observe_length: int
    predict_length: int
    time_step: float
    objects: dict[str, ObjectTrack]
    map_name: str | None = None
    scene_name: str | None = None

    @model_validator(mode="after")
    def _validate_object_lengths(self) -> ScenarioWindow:
        for obj_id, track in self.objects.items():
            if len(track.observe_trace) != self.observe_length:
                raise ValueError(f"{obj_id} observe length mismatch")
            if len(track.future_trace) != self.predict_length:
                raise ValueError(f"{obj_id} future length mismatch")
        return self


class PredictionBundle(BaseModel):
    """Predictor output used by the attack engine."""

    observe_traces: dict[str, list[list[float]]]
    future_traces: dict[str, list[list[float]]]
    predict_traces: dict[str, list[list[float]]]


class AttackResult(BaseModel):
    """Serializable attack result for a single case."""

    target_object_id: str
    objective: str
    is_adversarial: bool
    query_count: int
    distance_to_original: float
    perturbation: list[list[float]]
    metrics: dict[str, float] = Field(default_factory=dict)
