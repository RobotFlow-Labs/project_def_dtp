"""Smoke tests for the DEF-DTP FastAPI endpoints."""

from fastapi.testclient import TestClient

from anima_def_dtp.service.api import app

client = TestClient(app)

WINDOW = {
    "observe_length": 2,
    "predict_length": 2,
    "time_step": 0.5,
    "objects": {
        "1": {
            "type": 0,
            "observe_trace": [[0.0, 0.0], [1.0, 0.0]],
            "future_trace": [[2.0, 0.0], [3.0, 0.0]],
        }
    },
}


def test_healthz() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["module"] == "def-dtp"


def test_evaluate_returns_result() -> None:
    resp = client.post(
        "/evaluate",
        json={
            "dataset_name": "apolloscape",
            "predictor_name": "replay",
            "objective_name": "ade",
            "window": WINDOW,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "is_adversarial" in data
    assert data["target_object_id"] == "1"


def test_attack_returns_adversarial_result() -> None:
    resp = client.post(
        "/attack",
        json={
            "dataset_name": "apolloscape",
            "predictor_name": "replay",
            "objective_name": "left",
            "window": WINDOW,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_count"] > 0
    assert isinstance(data["perturbation"], list)


def test_evaluate_rejects_malformed_window() -> None:
    resp = client.post(
        "/evaluate",
        json={
            "dataset_name": "apolloscape",
            "predictor_name": "replay",
            "objective_name": "ade",
            "window": {"bad": "data"},
        },
    )
    assert resp.status_code == 422


def test_evaluate_rejects_empty_objects() -> None:
    resp = client.post(
        "/evaluate",
        json={
            "dataset_name": "apolloscape",
            "predictor_name": "replay",
            "objective_name": "ade",
            "window": {
                "observe_length": 2,
                "predict_length": 2,
                "time_step": 0.5,
                "objects": {},
            },
        },
    )
    assert resp.status_code == 422


def test_attack_rejects_unknown_target() -> None:
    resp = client.post(
        "/attack",
        json={
            "dataset_name": "apolloscape",
            "predictor_name": "replay",
            "objective_name": "ade",
            "target_object_id": "nonexistent",
            "window": WINDOW,
        },
    )
    assert resp.status_code == 404


def test_evaluate_rejects_invalid_dataset() -> None:
    resp = client.post(
        "/evaluate",
        json={
            "dataset_name": "fake_dataset",
            "predictor_name": "replay",
            "objective_name": "ade",
            "window": WINDOW,
        },
    )
    # Literal validation rejects at Pydantic level -> 422
    assert resp.status_code == 422
