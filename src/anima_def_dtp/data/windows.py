"""Offline JSON window slicing and format conversion between repo dicts and typed models.

The reference repo (eclipse-bot/DTP-Attack) stores scenario data as nested dicts
with long observe traces.  This module slices them into fixed-length windows that
match the paper's LI / LO per dataset and converts between the repo dict format
and the typed :class:`ScenarioWindow` / :class:`PredictionBundle` models.
"""

from __future__ import annotations

import copy
from typing import Any

from anima_def_dtp.types import ObjectTrack, PredictionBundle, ScenarioWindow


# ---------------------------------------------------------------------------
# Repo dict  →  typed models
# ---------------------------------------------------------------------------

def scenario_window_from_repo_dict(data: dict[str, Any]) -> ScenarioWindow:
    """Convert a repo-style JSON case dict into a :class:`ScenarioWindow`.

    Expected keys in *data*:
      - ``observe_length``, ``predict_length``, ``time_step``
      - ``objects``: ``{id: {observe_trace, future_trace, type?, ...}}``
    """
    objects: dict[str, ObjectTrack] = {}
    for obj_id, obj in data.get("objects", {}).items():
        objects[obj_id] = ObjectTrack(
            object_id=str(obj_id),
            object_type=obj.get("type", obj.get("object_type", 0)),
            observe_trace=obj["observe_trace"],
            future_trace=obj["future_trace"],
            observe_mask=obj.get("observe_mask", []),
            future_mask=obj.get("future_mask", []),
            observe_feature=obj.get("observe_feature", []),
            future_feature=obj.get("future_feature", []),
        )
    return ScenarioWindow(
        observe_length=data["observe_length"],
        predict_length=data["predict_length"],
        time_step=data.get("time_step", 0.5),
        objects=objects,
        map_name=data.get("map_name"),
        scene_name=data.get("scene_name"),
    )


def prediction_bundle_from_repo_dict(data: dict[str, Any]) -> PredictionBundle:
    """Extract a :class:`PredictionBundle` from a repo-style predictor output dict.

    The dict must have an ``objects`` key mapping object IDs to dicts with
    ``observe_trace``, ``future_trace``, and ``predict_trace``.
    """
    observe: dict[str, list[list[float]]] = {}
    future: dict[str, list[list[float]]] = {}
    predict: dict[str, list[list[float]]] = {}
    for obj_id, obj in data.get("objects", {}).items():
        observe[obj_id] = obj.get("observe_trace", [])
        future[obj_id] = obj.get("future_trace", [])
        predict[obj_id] = obj.get("predict_trace", obj.get("future_trace", []))
    return PredictionBundle(
        observe_traces=observe,
        future_traces=future,
        predict_traces=predict,
    )


# ---------------------------------------------------------------------------
# Typed models  →  repo dict
# ---------------------------------------------------------------------------

def scenario_window_to_repo_dict(
    window: ScenarioWindow,
    *,
    target_object_id: str | None = None,
    perturbation: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Convert a :class:`ScenarioWindow` to a repo-style dict, optionally
    applying *perturbation* to *target_object_id*'s observe trace.
    """
    objects: dict[str, dict[str, Any]] = {}
    for obj_id, track in window.objects.items():
        observe = [list(p) for p in track.observe_trace]
        if perturbation is not None and obj_id == target_object_id:
            observe = [
                [p[0] + d[0], p[1] + d[1]]
                for p, d in zip(observe, perturbation, strict=True)
            ]
        objects[obj_id] = {
            "type": track.object_type,
            "observe_trace": observe,
            "future_trace": [list(p) for p in track.future_trace],
            "observe_mask": list(track.observe_mask),
            "future_mask": list(track.future_mask),
            "observe_feature": [list(f) for f in track.observe_feature],
            "future_feature": [list(f) for f in track.future_feature],
        }
    return {
        "observe_length": window.observe_length,
        "predict_length": window.predict_length,
        "time_step": window.time_step,
        "objects": objects,
        "map_name": window.map_name,
        "scene_name": window.scene_name,
    }


# ---------------------------------------------------------------------------
# Attack-step slicing
# ---------------------------------------------------------------------------

def input_data_by_attack_step(
    data: dict[str, Any],
    obs_length: int,
    pred_length: int,
    attack_step: int,
) -> dict[str, Any]:
    """Slice a repo-style case dict at *attack_step* into a fixed-length window.

    For each object the long ``observe_trace`` is split:
      - observe: ``[attack_step : attack_step + obs_length]``
      - future:  ``[attack_step + obs_length : attack_step + obs_length + pred_length]``

    Returns a **new** dict ready for :func:`scenario_window_from_repo_dict`.
    """
    out = copy.deepcopy(data)
    out["observe_length"] = obs_length
    out["predict_length"] = pred_length
    for obj_id, obj in out.get("objects", {}).items():
        trace = obj.get("observe_trace", [])
        start = attack_step
        mid = start + obs_length
        end = mid + pred_length
        obj["observe_trace"] = trace[start:mid]
        obj["future_trace"] = trace[mid:end]
        if "observe_feature" in obj and obj["observe_feature"]:
            feat = obj["observe_feature"]
            obj["observe_feature"] = feat[start:mid]
            obj["future_feature"] = feat[mid:end]
    return out
